import logging
from typing import Union, List

import torch.nn
import torch
import torch.distributed as dist
from ._loss import find_loss_function, PerSpeciesLoss
from ._key import ABBREV
from torch_runstats import RunningStats, Reduction


class Loss:
    """
    assemble loss function based on key(s) and coefficient(s)

    Args:
        coeffs (dict, str): keys with coefficient and loss function name

    Example input dictionaries

    ```python
    'total_energy'
    ['total_energy', 'forces']
    {'total_energy': 1.0}
    {'total_energy': (1.0)}
    {'total_energy': (1.0, 'MSELoss'), 'forces': (1.0, 'L1Loss', param_dict)}
    {'total_energy': (1.0, user_define_callables), 'force': (1.0, 'L1Loss', param_dict)}
    {'total_energy': (1.0, 'MSELoss'),
     'force': (1.0, 'Weighted_L1Loss', param_dict)}
    ```

    The loss function can be a loss class name that is exactly the same (case sensitive) to the ones defined in torch.nn.
    It can also be a user define class type that
        - takes "reduction=none" as init argument
        - uses prediction tensor and reference tensor for its call functions,
        - outputs a vector with the same shape as pred/ref

    """

    def __init__(
        self,
        coeffs: Union[dict, str, List[str]],
        coeff_schedule: str = "constant",
        ldetach_method: bool = False,
        lauto_loss: bool = False,
    ):

        self.coeff_schedule = coeff_schedule
        self.ldetach_method = ldetach_method
        self.lauto_loss = lauto_loss
        self.coeffs = {}
        self.funcs = {}
        self.keys = []

        mseloss = find_loss_function("MSELoss", {})
        if isinstance(coeffs, str):
            self.coeffs[coeffs] = 1.0
            self.funcs[coeffs] = mseloss
        elif isinstance(coeffs, list):
            for key in coeffs:
                self.coeffs[key] = 1.0
                self.funcs[key] = mseloss
        elif isinstance(coeffs, dict):
            for key, value in coeffs.items():
                logging.debug(f" parsing {key} {value}")
                coeff = 1.0
                func = "MSELoss"
                func_params = {}
                if isinstance(value, (float, int)):
                    coeff = value
                elif isinstance(value, str) or callable(value):
                    func = value
                elif isinstance(value, (list, tuple)):
                    # list of [func], [func, param], [coeff, func], [coeff, func, params]
                    if isinstance(value[0], (float, int)):
                        coeff = value[0]
                        if len(value) > 1:
                            func = value[1]
                        if len(value) > 2:
                            func_params = value[2]
                    else:
                        func = value[0]
                        if len(value) > 1:
                            func_params = value[1]
                else:
                    raise NotImplementedError(
                        f"expected float, list or tuple, but get {type(value)}"
                    )
                logging.debug(f" parsing {coeff} {func}")
                self.coeffs[key] = coeff
                self.funcs[key] = find_loss_function(
                    func,
                    func_params,
                )
        else:
            raise NotImplementedError(
                f"loss_coeffs can only be str, list and dict. got {type(coeffs)}"
            )

        for key, coeff in self.coeffs.items():
            self.coeffs[key] = torch.as_tensor(coeff, dtype=torch.get_default_dtype())
            self.keys += [key]

        if self.lauto_loss:
            print("Use Automatic Weighted Loss")
            self.awl = AutomaticWeightedLoss(len(self.keys))
        elif self.ldetach_method:
            print("Use Detached Loss")
        else:
            print("Use Original Loss")

    def __call__(self, pred: dict, ref: dict, skip_key=None):

        loss = 0.0
        contrib = {}
        if self.lauto_loss:
            loss_list = []
        for key in self.coeffs:
            if key == skip_key:
                continue
            _loss = self.funcs[key](
                pred=pred,
                ref=ref,
                key=key,
                mean=True,
            )
            contrib[key] = _loss
            _loss = self.coeffs[key] * _loss
            if self.ldetach_method:
                _loss = _loss / (_loss.detach() + 1e-8)
            elif self.lauto_loss:
                loss_list.append(_loss)
            loss = loss + _loss
        contrib["awl"] = loss
        if self.lauto_loss:
            loss, _ = self.awl(loss_list)
        return loss, contrib

    def sum_and_count(self, pred: dict, ref: dict, skip_key=None):
        """Unreduced additive numerator/count per loss component.

        Returns ``(numerators, counts)`` dictionaries keyed like ``coeffs``;
        numerators keep their autograd graph, counts are detached scalars.
        The single-process ``__call__`` contract is unchanged.
        """
        numerators = {}
        counts = {}
        for key in self.coeffs:
            if key == skip_key:
                continue
            func = self.funcs[key]
            if not hasattr(func, "sum_and_count"):
                raise NotImplementedError(
                    f"loss function for key '{key}' ({type(func).__name__}) does "
                    "not implement the additive numerator/count contract "
                    "required for distributed training"
                )
            numerator, count = func.sum_and_count(pred=pred, ref=ref, key=key)
            numerators[key] = numerator
            counts[key] = count
        return numerators, counts

    def assert_distributed_compatible(self):
        """Fail fast for reductions without a globally additive contract."""
        for key, func in self.funcs.items():
            if isinstance(func, PerSpeciesLoss):
                raise NotImplementedError(
                    f"per-species loss for key '{key}' averages equally over "
                    "locally present species, which has no globally additive "
                    "contract; it is not supported in distributed training"
                )
            if not hasattr(func, "sum_and_count"):
                raise NotImplementedError(
                    f"loss function for key '{key}' ({type(func).__name__}) does "
                    "not implement the additive numerator/count contract "
                    "required for distributed training"
                )


def distributed_objective(loss_instance: Loss, pred: dict, ref: dict, world_size: int):
    """Global-correct local objective for distributed training.

    All-reduces each component's detached valid-element count, then scales
    the local error sums so that gradient averaging (native DDP or the manual
    all-reduce path) yields the concatenated global objective:
    ``L_r = sum_k a_k * W * S_rk / (sum_j C_jk)``. Returns
    ``(loss, numerators, counts)``; numerators keep their autograd graph and
    counts are the rank-local detached counts for statistics accumulation.
    """
    numerators, counts = loss_instance.sum_and_count(pred=pred, ref=ref)
    keys = list(numerators.keys())
    count_vec = torch.stack([counts[k].detach() for k in keys]).to(torch.float64)
    dist.all_reduce(count_vec, op=dist.ReduceOp.SUM)
    zero = (count_vec <= 0).nonzero().flatten().tolist()
    if zero:
        raise RuntimeError(
            "distributed loss: globally zero valid-element count for required "
            f"target(s): {[keys[i] for i in zero]}"
        )
    scale = float(world_size) / count_vec
    loss = None
    for i, key in enumerate(keys):
        term = loss_instance.coeffs[key] * (
            numerators[key] * scale[i].to(numerators[key].dtype)
        )
        loss = term if loss is None else loss + term
    return loss, numerators, counts


class LossStat:
    """
    The class that accumulate the loss function values over all batches
    for each loss component.

    Args:

    keys (null): redundant argument

    """

    def __init__(self, loss_instance=None):
        self.loss_stat = {
            "total": RunningStats(
                dim=tuple(), reduction=Reduction.MEAN, ignore_nan=False
            )
        }
        self.ignore_nan = {}
        if loss_instance is not None:
            for key, func in loss_instance.funcs.items():
                self.ignore_nan[key] = (
                    func.ignore_nan if hasattr(func, "ignore_nan") else False
                )

    def __call__(self, loss, loss_contrib, materialize=True):
        """
        Args:

        loss (torch.Tensor): the value of the total loss function for the current batch
        loss (Dict(torch.Tensor)): the dictionary which contain the loss components
        """

        results = {}

        total = self.loss_stat["total"].accumulate_batch(loss)
        if materialize:
            results["loss"] = total.item()

        # go through each component
        for k, v in loss_contrib.items():

            # initialize for the 1st batch
            if k not in self.loss_stat:
                self.loss_stat[k] = RunningStats(
                    dim=tuple(),
                    reduction=Reduction.MEAN,
                    ignore_nan=self.ignore_nan.get(k, False),
                )
                device = v.get_device()
                self.loss_stat[k].to(device="cpu" if device == -1 else device)

            value = self.loss_stat[k].accumulate_batch(v)
            if materialize:
                results["loss_" + ABBREV.get(k, k)] = value.item()
        return results

    def reset(self):
        """
        Reset all the counters to zero
        """

        for v in self.loss_stat.values():
            v.reset()

    def to(self, device):
        for v in self.loss_stat.values():
            v.to(device=device)

    def current_result(self):
        results = {
            "loss_" + ABBREV.get(k, k): v.current_result().item()
            for k, v in self.loss_stat.items()
            if k != "total"
        }
        results["loss"] = self.loss_stat["total"].current_result().item()
        return results


class DistributedLossStat:
    """Additive numerator/count accumulator for distributed loss statistics.

    Drop-in for ``LossStat`` when a process group is initialized: accumulates
    detached float64 error sums and valid-element counts per component across
    batches; ``global_result()`` performs a single packed all-reduce and
    finalizes component means and the fixed-weight total from those means.
    """

    def __init__(self, loss_instance, device=None):
        self.keys = list(loss_instance.keys)
        self.coeffs = {k: float(loss_instance.coeffs[k]) for k in self.keys}
        self._device = device if device is not None else torch.device("cpu")
        self.reset()

    def reset(self):
        self._sums = {
            k: torch.zeros((), dtype=torch.float64, device=self._device)
            for k in self.keys
        }
        self._counts = {
            k: torch.zeros((), dtype=torch.float64, device=self._device)
            for k in self.keys
        }

    def to(self, device):
        self._device = device
        for stat in (self._sums, self._counts):
            for k in stat:
                stat[k] = stat[k].to(device)

    def accumulate(self, numerators, counts, materialize=True):
        """Accumulate one batch; return rank-local batch means for logging."""
        results = {}
        total = 0.0
        component_means = {}
        for key in self.keys:
            s = numerators[key].detach().to(device=self._device, dtype=torch.float64)
            c = counts[key].detach().to(device=self._device, dtype=torch.float64)
            self._sums[key] += s
            self._counts[key] += c
            if materialize:
                component_means[key] = (s / c.clamp_min(1.0)).item()
                total += self.coeffs[key] * component_means[key]
        if not materialize:
            return {}
        results["loss"] = total
        for key in self.keys:
            results["loss_" + ABBREV.get(key, key)] = component_means[key]
        results["loss_awl"] = total
        return results

    def global_result(self):
        """Single packed all-reduce of sums/counts; finalize global means."""
        packed = torch.stack(
            [self._sums[k] for k in self.keys] + [self._counts[k] for k in self.keys]
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        n = len(self.keys)
        total = 0.0
        component_means = {}
        for i, key in enumerate(self.keys):
            count = packed[n + i]
            if count <= 0:
                raise RuntimeError(
                    "distributed loss: globally zero valid-element count for "
                    f"required target '{key}'"
                )
            component_means[key] = (packed[i] / count).item()
            total += self.coeffs[key] * component_means[key]
        results = {"loss_" + ABBREV.get(k, k): component_means[k] for k in self.keys}
        results["loss_awl"] = total
        results["loss"] = total
        return results


class AutomaticWeightedLoss(torch.nn.Module):
    """automatically weighted multi-task loss
    Params:
        num: int, the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl([loss1, loss2])
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(
                1 + self.params[i] ** 2
            )
        coefficient = 0.5 / (self.params[: len(x)] ** 2)
        return loss_sum, coefficient
