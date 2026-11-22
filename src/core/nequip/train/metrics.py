from copy import deepcopy
from hashlib import sha1
from typing import Union, Sequence, Tuple, Optional

import yaml

import torch
import torch.distributed as dist

from nequip.data import AtomicDataDict
from torch_runstats import RunningStats, Reduction

from ._loss import find_loss_function
from ._key import ABBREV

metrics_to_reduction = {
    "mae": Reduction.MEAN,
    "mean": Reduction.MEAN,
    "rmse": Reduction.RMS,
}


def _prod(x):
    out = 1
    for a in x:
        out *= a
    return out


class AdditiveStats:
    """Additive sufficient statistics with a RunningStats-compatible interface.

    Accumulates detached float64 error sums and valid-element counts per bin
    (MEAN) or squared-error sums (RMS, square root only after aggregation).
    Used in distributed mode so that one final all-reduce of the packed
    sums/counts yields the exact global statistic; local ``current_result``
    and batch returns mirror ``RunningStats`` (zero-count bins give 0.0).
    """

    def __init__(
        self,
        dim=1,
        reduce_dims=tuple(),
        reduction: Reduction = Reduction.MEAN,
        ignore_nan: bool = False,
        n_bins: int = 1,
    ):
        if isinstance(dim, int):
            dim = (dim,)
        self._in_dim = tuple(dim)
        self._reduce_dims = tuple(sorted(set(reduce_dims)))
        if len(self._reduce_dims) > 0 and (
            min(self._reduce_dims) < 0 or max(self._reduce_dims) >= len(self._in_dim)
        ):
            raise ValueError(f"Invalid dimension indexes in reduce_dims {reduce_dims}")
        if reduction not in (Reduction.MEAN, Reduction.RMS):
            raise NotImplementedError(
                f"Reduction {reduction} is not supported in distributed mode"
            )
        self._reduction = reduction
        self.ignore_nan = ignore_nan
        self._out_dim = tuple(
            d for i, d in enumerate(self._in_dim) if i not in self._reduce_dims
        )
        self._reduction_factor = _prod(
            d for i, d in enumerate(self._in_dim) if i in self._reduce_dims
        )
        self._n_bins = n_bins
        self.reset()

    def reset(self):
        self._sum = torch.zeros((self._n_bins,) + self._out_dim, dtype=torch.float64)
        self._count = torch.zeros((self._n_bins,) + self._out_dim, dtype=torch.float64)

    def to(self, device=None, dtype=None):
        self._sum = self._sum.to(dtype=dtype, device=device)
        self._count = self._count.to(device=device)

    @property
    def dim(self):
        return self._in_dim

    @property
    def output_dim(self):
        return self._out_dim

    @property
    def reduce_dims(self):
        return self._reduce_dims

    @property
    def reduction(self):
        return self._reduction

    @property
    def n_bins(self):
        return self._n_bins

    def _finalize(self, sums, counts):
        average = torch.nan_to_num(sums / counts, nan=0.0)
        if self._reduction == Reduction.RMS:
            average = average.sqrt()
        return average

    def accumulate_batch(
        self, batch: torch.Tensor, accumulate_by=None, materialize=True
    ):
        if batch.shape == self._in_dim:
            batch = batch.unsqueeze(0)
        v = batch.detach().to(torch.float64)
        if self._reduction == Reduction.RMS:
            v = v.square()

        not_nan = (v == v).to(torch.float64) if self.ignore_nan else None
        if self.ignore_nan:
            v = torch.where(torch.isnan(v), torch.zeros_like(v), v)

        if len(self._reduce_dims) > 0:
            rd = tuple(i + 1 for i in self._reduce_dims)
            new_sum = v.sum(dim=rd)
            new_count = not_nan.sum(dim=rd) if self.ignore_nan else None
        else:
            new_sum = v
            new_count = not_nan

        if accumulate_by is None:
            new_sum = new_sum.sum(dim=0, keepdim=True)
            if self.ignore_nan:
                new_count = new_count.sum(dim=0, keepdim=True)
            else:
                new_count = torch.full(
                    (1,) + self._out_dim,
                    batch.shape[0] * self._reduction_factor,
                    dtype=torch.float64,
                    device=new_sum.device,
                )
            self._sum += new_sum.to(self._sum.device)
            self._count += new_count.to(self._count.device)
            return self._finalize(new_sum, new_count) if materialize else None

        bins = accumulate_by.reshape(-1).long()
        sums = torch.zeros(
            (self._n_bins,) + self._out_dim,
            dtype=torch.float64,
            device=new_sum.device,
        )
        sums.index_add_(0, bins, new_sum)
        if self.ignore_nan:
            counts = torch.zeros_like(sums)
            counts.index_add_(0, bins, new_count)
        else:
            counts = torch.bincount(bins, minlength=self._n_bins).to(torch.float64)
            counts = counts.reshape((-1,) + (1,) * len(self._out_dim))
            counts = counts * self._reduction_factor
            counts = counts.expand((self._n_bins,) + self._out_dim)
        self._sum += sums.to(self._sum.device)
        self._count += counts.to(self._count.device)
        # mirror RunningStats: batch result only spans bins seen in this batch
        if not materialize:
            return None
        n_batch_bins = int(bins.max().item()) + 1 if bins.numel() > 0 else 0
        return self._finalize(sums[:n_batch_bins], counts[:n_batch_bins])

    def current_result(self):
        return self._finalize(self._sum, self._count)


class Trace(torch.nn.Module):
    """
    Custom Metric function to compute the Trace of a tensor.
    Computes Trace(pred[key])
    """

    def __init__(self, scale=1.0, units=""):
        super().__init__()
        if not 0 < scale < float("inf"):
            raise ValueError("Trace scale must be finite and positive")
        self.scale = float(scale)
        self.units = units

    def get_name(self, name):
        return f"{name}_{self.units}" if self.units else name

    def forward(self, pred: dict, ref: dict, key: str, mean: bool = False):
        # 1. 获取张量 (预期形状 [Batch, 3, 3] 或 [N_atoms, 3, 3])
        if key not in pred:
            # Disabled evidence heads still have configured trace metrics.
            # Their zero must follow the actual predictions: a synthetic CPU
            # fallback conflicts with running statistics moved to the GPU.
            for value in pred.values():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    return value.new_zeros(1)
            # Preserve the empty-prediction behavior when no dtype is available.
            return torch.zeros(1)

        tensor = pred[key]  # [..., 3, 3]

        # 2. 计算迹: 对角线元素求和
        # dim1=-2, dim2=-1 对应最后两个维度 (3x3)
        trace = torch.diagonal(tensor, dim1=-2, dim2=-1).sum(
            -1
        )  # 结果形状 [Batch] 或 [N_atoms]
        trace = trace * self.scale

        # 3. 兼容接口
        # Metrics 类后续会自己处理 RunningStats (求平均等)，
        # 所以这里我们只需要返回每个样本的 trace 值即可。
        if mean:
            return trace.mean()
        else:
            return trace


class Metrics:
    """Computes all mae, rmse needed for report

    Args:
        components: a list or a tuples of definition.

    Example:

    ```
    components = [(key1, "rmse"), (key2, "mse")]
    ```

    You can also offer more details with a dictionary. The keys can be any keys for RunningStats or

    report_per_component (bool): if True, report the mean on each component (equivalent to mean(axis=0) in numpy),
                                 otherwise, take the mean across batch and all components for vector data.
    functional: the function to compute the error. It has to be exactly the same as the one defined in torch.nn.
                Callables are also allowed.
                default: "L1Loss"
    functional_kwargs: keyword arguments for the functional. Trace accepts
                       scale (positive multiplier) and units (column suffix).
    PerSpecies: whether to compute the estimation for each species or not

    the keys are case-sensitive.


    ```
    components = (
        (
            AtomicDataDict.FORCE_KEY,
            "rmse",
            {"PerSpecies": True, "functional": "L1Loss", "dim": 3},
        ),
        (AtomicDataDict.FORCE_KEY, "mae", {"dim": 3}),
    )
    ```

    """

    def __init__(
        self,
        components: Sequence[Union[Tuple[str, str], Tuple[str, str, dict]]],
        distributed: bool = False,
        species_count: Optional[int] = None,
    ):

        self._distributed = distributed
        self._species_count = species_count
        if distributed:
            for component in components:
                _, _, params = Metrics.parse(component)
                if params.get("PerSpecies", False) and species_count is None:
                    raise RuntimeError(
                        "distributed per-species metrics require species_count "
                        "so every rank allocates a rank-consistent schema"
                    )

        self.running_stats = {}
        self.params = {}
        self.funcs = {}
        self.kwargs = {}
        for component in components:

            key, reduction, params = Metrics.parse(component)

            params["PerSpecies"] = params.get("PerSpecies", False)
            params["PerAtom"] = params.get("PerAtom", False)

            param_hash = Metrics.hash_component(component)

            functional = params.get("functional", "L1Loss")

            # default is to flatten the array

            if key not in self.running_stats:
                self.running_stats[key] = {}
                self.funcs[key] = {}
                self.kwargs[key] = {}
                self.params[key] = {}

            # store for initialization
            kwargs = deepcopy(params)
            kwargs.pop("functional", "L1Loss")
            functional_kwargs = kwargs.pop("functional_kwargs", {})
            kwargs.pop("PerSpecies")
            kwargs.pop("PerAtom")

            # by default, report a scalar that is mae and rmse over all component
            self.kwargs[key][param_hash] = dict(
                reduction=metrics_to_reduction.get(reduction, reduction),
            )
            self.kwargs[key][param_hash].update(kwargs)
            self.params[key][param_hash] = (reduction, params)
            if functional == "Trace":
                self.funcs[key][param_hash] = Trace(**functional_kwargs)
            else:
                self.funcs[key][param_hash] = find_loss_function(
                    functional, functional_kwargs
                )

    def init_runstat(self, params, error: torch.Tensor, n_bins: int = 1):
        """
        Initialize Runstat Counter based on the shape of the error matrix

        Args:
        params (dict): dictionary of additional arguments
        error (torch.Tensor): error matrix
        """

        kwargs = deepcopy(params)
        # automatically define the dimensionality
        if "dim" not in kwargs:
            kwargs["dim"] = error.shape[1:]

        if "reduce_dims" not in kwargs:
            if not kwargs.pop("report_per_component", False):
                kwargs["reduce_dims"] = tuple(range(len(error.shape) - 1))

        if self._distributed:
            rs = AdditiveStats(
                dim=kwargs["dim"],
                reduce_dims=kwargs.get("reduce_dims", tuple()),
                reduction=kwargs["reduction"],
                ignore_nan=kwargs.get("ignore_nan", False),
                n_bins=n_bins,
            )
        else:
            rs = RunningStats(**kwargs)
        rs.to(device=error.device)
        return rs

    @staticmethod
    def hash_component(component):
        buffer = yaml.dump(component).encode("ascii")
        return sha1(buffer).hexdigest()

    @staticmethod
    def parse(component):
        # parse the input list
        reduction = "mae"
        params = {}

        if isinstance(component, str):
            key = component
        elif len(component) == 1:
            key = component[0]
        elif len(component) == 2:
            key, reduction = component
        elif len(component) == 3:
            key, reduction, _params = component
            params = {k: deepcopy(v) for k, v in _params.items()}
        else:
            raise ValueError(
                f"tuple should have a max length of 3 but {len(component)} is given"
            )
        return key, reduction, params

    def __call__(self, pred: dict, ref: dict, materialize=True):

        metrics = {}
        N = None
        for key in self.funcs.keys():
            for param_hash, kwargs in self.kwargs[key].items():
                func = self.funcs[key][param_hash]
                error = func(
                    pred=pred,
                    ref=ref,
                    key=key,
                    mean=False,
                )

                _, params = self.params[key][param_hash]
                per_species = params["PerSpecies"]
                per_atom = params["PerAtom"]

                # initialize the internal run_stat base on the error shape
                if param_hash not in self.running_stats[key]:
                    n_bins = 1
                    if self._distributed and per_species:
                        n_bins = self._species_count
                    self.running_stats[key][param_hash] = self.init_runstat(
                        params=kwargs, error=error, n_bins=n_bins
                    )

                stat = self.running_stats[key][param_hash]

                params = {}
                if per_species:
                    # TO DO, this needs OneHot component. will need to be decoupled
                    params = {
                        "accumulate_by": pred[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
                    }
                if per_atom:
                    if N is None:
                        N = torch.bincount(ref[AtomicDataDict.BATCH_KEY]).unsqueeze(-1)
                    error_N = error / N
                else:
                    error_N = error

                if stat.dim == () and not per_species:
                    if isinstance(stat, AdditiveStats):
                        metrics[(key, param_hash)] = stat.accumulate_batch(
                            error_N.flatten(), materialize=materialize, **params
                        )
                    else:
                        metrics[(key, param_hash)] = stat.accumulate_batch(
                            error_N.flatten(), **params
                        )
                else:
                    if isinstance(stat, AdditiveStats):
                        metrics[(key, param_hash)] = stat.accumulate_batch(
                            error_N, materialize=materialize, **params
                        )
                    else:
                        metrics[(key, param_hash)] = stat.accumulate_batch(
                            error_N, **params
                        )

        return metrics if materialize else None

    def reset(self):
        for stats in self.running_stats.values():
            for stat in stats.values():
                stat.reset()

    def to(self, device):
        for stats in self.running_stats.values():
            for stat in stats.values():
                stat.to(device=device)

    def current_result(self):

        metrics = {}
        for key, stats in self.running_stats.items():
            for reduction, stat in stats.items():
                metrics[(key, reduction)] = stat.current_result()
        return metrics

    def global_result(self):
        """All-reduce additive statistics once and finalize global metrics.

        Rank 0 broadcasts the metric schema first so ranks with zero local
        batches allocate identical zero accumulators and join the same packed
        all-reduce; every rank then finalizes identical values locally.
        """
        if dist.get_rank() == 0:
            layout = [
                (key, param_hash, tuple(stat._sum.shape), stat.reduction.value)
                for key, stats in self.running_stats.items()
                for param_hash, stat in stats.items()
            ]
        else:
            layout = None
        obj = [layout]
        bcast_kwargs = {}
        if dist.get_backend() != "gloo":
            bcast_kwargs["device"] = torch.device("cuda", torch.cuda.current_device())
        dist.broadcast_object_list(obj, src=0, **bcast_kwargs)
        layout = obj[0]

        stats_list = []
        for key, param_hash, shape, reduction in layout:
            stat = self.running_stats.get(key, {}).get(param_hash)
            if stat is None:
                # rank with zero local batches: allocate the broadcast schema
                stat = AdditiveStats(
                    dim=shape[1:],
                    reduction=Reduction(reduction),
                    n_bins=shape[0],
                )
                self.running_stats.setdefault(key, {})[param_hash] = stat
            stats_list.append(((key, param_hash), stat))

        device = (
            torch.device("cpu")
            if dist.get_backend() == "gloo"
            else torch.device("cuda", torch.cuda.current_device())
        )
        pieces = [stat._sum.reshape(-1).to(device) for _, stat in stats_list] + [
            stat._count.reshape(-1).to(device) for _, stat in stats_list
        ]
        packed = (
            torch.cat(pieces)
            if pieces
            else torch.zeros(0, dtype=torch.float64, device=device)
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)

        results = {}
        offset = 0
        finalized = []
        for _, stat in stats_list:
            numel = stat._sum.numel()
            finalized.append(packed[offset : offset + numel].reshape(stat._sum.shape))
            offset += numel
        for ((key, param_hash), stat), sums in zip(stats_list, finalized):
            counts = packed[offset : offset + sums.numel()].reshape(sums.shape)
            offset += sums.numel()
            results[(key, param_hash)] = stat._finalize(sums, counts)
        return results

    def flatten_metrics(self, metrics, type_names=None):

        flat_dict = {}
        skip_keys = []
        for k, value in metrics.items():

            key, param_hash = k
            reduction, params = self.params[key][param_hash]

            short_name = ABBREV.get(key, key)
            if hasattr(self.funcs[key][param_hash], "get_name"):
                short_name = self.funcs[key][param_hash].get_name(short_name)

            per_atom = params["PerAtom"]
            suffix = "/N" if per_atom else ""
            item_name = f"{short_name}{suffix}_{reduction}"

            stat = self.running_stats[key][param_hash]
            per_species = params["PerSpecies"]

            if per_species:
                if stat.output_dim == tuple():
                    if type_names is None:
                        type_names = [i for i in range(len(value))]
                    for id_ele, v in enumerate(value):
                        if type_names is not None:
                            flat_dict[f"{type_names[id_ele]}_{item_name}"] = v.item()
                        else:
                            flat_dict[f"{id_ele}_{item_name}"] = v.item()

                    flat_dict[f"psavg_{item_name}"] = value.mean().item()
                else:
                    for id_ele, vec in enumerate(value):
                        ele = type_names[id_ele]
                        for idx, v in enumerate(vec):
                            name = f"{ele}_{item_name}_{idx}"
                            flat_dict[name] = v.item()
                            skip_keys.append(name)

            else:
                if stat.output_dim == tuple():
                    # a scalar
                    flat_dict[item_name] = value.item()
                else:
                    # a vector
                    for idx, v in enumerate(value.flatten()):
                        flat_dict[f"{item_name}_{idx}"] = v.item()
        return flat_dict, skip_keys
