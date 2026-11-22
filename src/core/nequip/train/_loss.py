import inspect
import logging
from collections.abc import Mapping

import torch
import torch.nn
from torch_runstats.scatter import scatter, scatter_mean

from nequip.data import AtomicDataDict
from nequip.utils import instantiate_from_cls_name, load_callable


def _promote(prediction: torch.Tensor, reference: torch.Tensor):
    """Cast a floating prediction/reference pair to their promoted dtype.

    With ``model_dtype`` different from ``default_dtype``, e.g. float32
    predictions meet float64 targets. Most torch losses promote in forward,
    but some (``mse_loss``, ``huber_loss``) cannot backpropagate mixed dtypes.
    The cast is differentiable, so gradients return in the prediction dtype.
    Same-dtype pairs are returned unchanged.
    """
    if (
        prediction.dtype == reference.dtype
        or not prediction.is_floating_point()
        or not reference.is_floating_point()
    ):
        return prediction, reference
    dtype = torch.promote_types(prediction.dtype, reference.dtype)
    return prediction.to(dtype), reference.to(dtype)


def _base_keys(base):
    # a ``Batch``/``Data`` iterates over (key, value) pairs and exposes its
    # field names through the ``keys`` property; dicts iterate over keys
    if isinstance(base, Mapping):
        return iter(base)
    keys = base.keys
    return iter(keys() if callable(keys) else keys)


class _OverlayMapping(Mapping):
    """Read-only mapping view of ``base`` with some fields replaced.

    ``base`` is a data dict or a ``Batch``; nothing is copied. Missing keys
    raise ``KeyError`` (also for a ``Batch``, whose own ``[]`` returns
    ``None``), so ``get``, ``items``, ``keys``, ``values``, ``in`` and ``len``
    behave like on a dict.
    """

    def __init__(self, base, overrides: dict):
        self._base = base
        self._overrides = dict(overrides)

    def __getitem__(self, key):
        if key in self._overrides:
            return self._overrides[key]
        if key in self._base:
            return self._base[key]
        raise KeyError(key)

    def __contains__(self, key):
        return key in self._overrides or key in self._base

    def __iter__(self):
        yield from self._overrides
        for key in _base_keys(self._base):
            if key not in self._overrides:
                yield key

    def __len__(self):
        return sum(1 for _ in self)


def _promote_key(pred, ref, key: str):
    """``_promote`` for ``pred[key]`` and ``ref[key]``.

    Returns the inputs themselves when no cast is needed, otherwise read-only
    mapping views with the promoted ``key``.
    """
    prediction, reference = _promote(pred[key], ref[key])
    if prediction is pred[key] and reference is ref[key]:
        return pred, ref
    return (
        _OverlayMapping(pred, {key: prediction}),
        _OverlayMapping(ref, {key: reference}),
    )


class SimpleLoss:
    """wrapper to compute weighted loss function

    Args:

    func_name (str or class): any loss function defined in torch.nn that
        takes "reduction=none" as init argument, uses prediction tensor,
        and reference tensor for its call functions, and outputs a vector
        with the same shape as pred/ref. A class with the same contract, or
        its fully qualified dotted import path, is also accepted.
    params (str): arguments needed to initialize the function above

    Return:

    if mean is True, return a scalar; else return the error matrix of each entry
    """

    def __init__(self, func_name: str, params: dict = {}):
        self.ignore_nan = params.get("ignore_nan", False)
        func, _ = instantiate_from_cls_name(
            torch.nn,
            class_name=func_name,
            prefix="",
            positional_args=dict(reduction="none"),
            optional_args=params,
            all_args={},
        )
        # plain class name, used below to special-case some torch.nn losses
        self.func_name = (
            func_name
            if isinstance(func_name, str) and "." not in func_name
            else type(func).__name__
        )
        self.func = func

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
    ):
        pred, ref = _promote_key(pred, ref, key)
        # zero the nan entries
        has_nan = self.ignore_nan and torch.isnan(ref[key].mean())
        if has_nan:
            not_nan = (ref[key] == ref[key]).int()
            if self.func_name == "CosineEmbeddingLoss":
                loss = (
                    self.func(
                        pred[key],
                        torch.nan_to_num(ref[key], nan=0.0),
                        torch.Tensor([1]),
                    )
                    * not_nan
                )
            else:
                loss = (
                    self.func(pred[key], torch.nan_to_num(ref[key], nan=0.0)) * not_nan
                )
            if mean:
                return loss.sum() / not_nan.sum()
            else:
                return loss
        else:
            if self.func_name == "CosineEmbeddingLoss":
                loss = self.func(
                    pred[key], ref[key], torch.Tensor([1]).to(pred[key].device)
                )
            else:
                loss = self.func(pred[key], ref[key])
            if mean:
                return loss.mean()
            else:
                return loss

    def sum_and_count(self, pred: dict, ref: dict, key: str):
        """Additive (error_sum, valid_count) contract for distributed use.

        ``error_sum`` keeps its autograd graph; ``count`` is a detached
        scalar. ``error_sum / count`` reproduces the ``mean=True`` result.
        """
        pred, ref = _promote_key(pred, ref, key)
        has_nan = self.ignore_nan and torch.isnan(ref[key].mean())
        if has_nan:
            not_nan = (ref[key] == ref[key]).int()
            if self.func_name == "CosineEmbeddingLoss":
                loss = (
                    self.func(
                        pred[key],
                        torch.nan_to_num(ref[key], nan=0.0),
                        torch.Tensor([1]),
                    )
                    * not_nan
                )
            else:
                loss = (
                    self.func(pred[key], torch.nan_to_num(ref[key], nan=0.0)) * not_nan
                )
            return loss.sum(), not_nan.sum().detach()
        else:
            if self.func_name == "CosineEmbeddingLoss":
                loss = self.func(
                    pred[key], ref[key], torch.Tensor([1]).to(pred[key].device)
                )
            else:
                loss = self.func(pred[key], ref[key])
            return loss.sum(), torch.as_tensor(loss.numel(), device=loss.device)


class PerAtomLoss(SimpleLoss):
    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
    ):
        pred, ref = _promote_key(pred, ref, key)
        # zero the nan entries
        has_nan = self.ignore_nan and torch.isnan(ref[key].sum())
        N = torch.bincount(ref[AtomicDataDict.BATCH_KEY])
        # N = N.reshape((-1, 1))
        if has_nan:
            not_nan = (ref[key] == ref[key]).int()
            loss = self.func(pred[key], torch.nan_to_num(ref[key], nan=0.0))
            N = N.reshape([N.shape[0]] + [1] * (loss.ndim - 1))
            loss = loss * not_nan / N
            if self.func_name == "MSELoss":
                loss = loss / N
            assert loss.shape == pred[key].shape  # [atom, dim]
            if mean:
                return loss.sum() / not_nan.sum().clamp_min(1)
            else:
                return loss
        else:
            loss = self.func(pred[key], ref[key])
            len_size = len(loss.shape)
            for i in range(len_size - 1):
                N = N.unsqueeze(-1)
            loss = loss / N
            if self.func_name == "MSELoss":
                loss = loss / N
            assert loss.shape == pred[key].shape  # [atom, dim]
            if mean:
                return loss.mean()
            else:
                return loss

    def sum_and_count(self, pred: dict, ref: dict, key: str):
        """Additive (error_sum, valid_count) with per-frame atom normalization.

        Counts are the unclamped valid-element counts so that they remain
        additive across ranks; clamping happens only at finalization.
        """
        pred, ref = _promote_key(pred, ref, key)
        has_nan = self.ignore_nan and torch.isnan(ref[key].sum())
        N = torch.bincount(ref[AtomicDataDict.BATCH_KEY])
        if has_nan:
            not_nan = (ref[key] == ref[key]).int()
            loss = self.func(pred[key], torch.nan_to_num(ref[key], nan=0.0))
            N = N.reshape([N.shape[0]] + [1] * (loss.ndim - 1))
            loss = loss * not_nan / N
            if self.func_name == "MSELoss":
                loss = loss / N
            assert loss.shape == pred[key].shape
            return loss.sum(), not_nan.sum().detach()
        else:
            loss = self.func(pred[key], ref[key])
            len_size = len(loss.shape)
            for i in range(len_size - 1):
                N = N.unsqueeze(-1)
            loss = loss / N
            if self.func_name == "MSELoss":
                loss = loss / N
            assert loss.shape == pred[key].shape
            return loss.sum(), torch.as_tensor(loss.numel(), device=loss.device)


class PerSpeciesLoss(SimpleLoss):
    """Compute loss for each species and average among the same species
    before summing them up.

    Args same as SimpleLoss
    """

    def sum_and_count(self, pred: dict, ref: dict, key: str):
        raise NotImplementedError(
            f"PerSpeciesLoss for key '{key}' averages equally over locally "
            "present species, which has no globally additive numerator/count "
            "contract; it is not supported in distributed training"
        )

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
    ):
        if not mean:
            raise NotImplementedError("Cannot handle this yet")

        pred, ref = _promote_key(pred, ref, key)
        has_nan = self.ignore_nan and torch.isnan(ref[key].mean())

        if has_nan:
            not_nan = (ref[key] == ref[key]).int()
            per_atom_loss = (
                self.func(pred[key], torch.nan_to_num(ref[key], nan=0.0)) * not_nan
            )
        else:
            per_atom_loss = self.func(pred[key], ref[key])

        reduce_dims = tuple(i + 1 for i in range(len(per_atom_loss.shape) - 1))

        spe_idx = pred[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        if has_nan:
            if len(reduce_dims) > 0:
                per_atom_loss = per_atom_loss.sum(dim=reduce_dims)
            assert per_atom_loss.ndim == 1

            per_species_loss = scatter(per_atom_loss, spe_idx, dim=0)

            assert per_species_loss.ndim == 1  # [type]

            N = scatter(not_nan, spe_idx, dim=0)
            N = N.sum(reduce_dims)
            N = N.reciprocal()
            N_species = ((N == N).int()).sum()
            assert N.ndim == 1  # [type]

            per_species_loss = (per_species_loss * N).sum() / N_species

            return per_species_loss

        else:

            if len(reduce_dims) > 0:
                per_atom_loss = per_atom_loss.mean(dim=reduce_dims)
            assert per_atom_loss.ndim == 1

            # offset species index by 1 to use 0 for nan
            _, inverse_species_index = torch.unique(spe_idx, return_inverse=True)

            per_species_loss = scatter_mean(per_atom_loss, inverse_species_index, dim=0)
            assert per_species_loss.ndim == 1  # [type]

            return per_species_loss.mean()


def _frame_index(ref: dict, rows: int, what: str):
    """Map the leading dimension of a per-atom or per-frame field to frames.

    Returns ``(n_frames, frame_of_row, per_atom)``. Every frame has at least
    one atom, so ``rows == n_atoms == n_frames`` only when each frame has a
    single atom, in which case both interpretations coincide.
    """
    # only ``in`` and ``[]``: ``ref`` may be a dict or a ``Batch``
    batch = ref[AtomicDataDict.BATCH_KEY] if AtomicDataDict.BATCH_KEY in ref else None
    if batch is None:
        n_atoms = ref[AtomicDataDict.POSITIONS_KEY].shape[0]
        batch = torch.zeros(
            n_atoms,
            dtype=torch.long,
            device=ref[AtomicDataDict.POSITIONS_KEY].device,
        )
        n_frames = 1
    else:
        batch = batch.reshape(-1).to(torch.long)
        n_atoms = batch.shape[0]
        n_frames = int(batch.max()) + 1 if n_atoms > 0 else 0
    if rows == n_atoms:
        return n_frames, batch, True
    if rows == n_frames:
        return n_frames, torch.arange(n_frames, device=batch.device), False
    raise ValueError(
        f"{what} has leading dimension {rows}, which matches neither the number "
        f"of atoms ({n_atoms}) nor the number of frames ({n_frames}); per-frame "
        "losses support per-atom and per-frame fields only"
    )


class PerFrameLoss(SimpleLoss):
    """Frame-weighted loss: the per-frame element mean, summed over frames.

    For every frame ``f`` the element-wise loss (``|pred - ref|`` for
    ``PerFrameL1Loss``) is averaged over all of the frame's elements (its
    atoms times the field components for a per-atom field, the components for
    a per-frame field), giving ``l_f``. The additive contract is
    ``S = sum_f l_f`` and ``C = number of frames``, so the single-process
    value is the mean of ``l_f`` over frames: every frame has the same weight
    regardless of its number of atoms.

    In distributed training the applied gradient is
    ``coeff * grad(sum_f l_f) / (global number of frames)``. A coefficient
    equal to the global number of frames per optimizer step
    (``world_size * batch_size``) therefore applies the *sum* of the
    per-frame gradients.

    Optional ``params`` (third entry of a ``loss_coeffs`` item):

    - ``mask_field``: boolean per-frame or per-atom field, read from the
      prediction (e.g. a model output) and otherwise from the reference data.
      With a per-atom mask a frame's mean runs over its selected atoms only.
      A frame without selected elements adds an exact zero to ``S`` but still
      counts one in ``C``; it stays in the autograd graph with a zero
      gradient, like ``0 * pred.sum()``. Masked-out elements are compared
      with the (detached) prediction instead of their target, so their loss
      and gradient are exactly zero for any element-wise loss, even when the
      targets of unlabeled frames are NaN placeholders.
    - ``ref_key``: reference field compared with ``pred[key]``; defaults to
      ``key``. Default metrics are derived from the loss keys, so set
      ``metrics_components`` explicitly when ``ref_key`` differs from ``key``.
    - ``ignore_nan``: NaN reference elements are left out of the frame means.
    """

    def __init__(self, func_name: str, params: dict = {}):
        super().__init__(func_name, params)
        self.mask_field = params.get("mask_field", None)
        self.ref_key = params.get("ref_key", None)

    def _row_mask(
        self,
        pred: dict,
        ref: dict,
        n_rows: int,
        key_per_atom: bool,
        frame_of_row: torch.Tensor,
    ):
        if self.mask_field in pred:
            mask = pred[self.mask_field]
        elif self.mask_field in ref:
            mask = ref[self.mask_field]
        else:
            raise KeyError(
                f"mask_field '{self.mask_field}' is neither in the predictions nor "
                "in the data"
            )
        mask = mask.reshape(mask.shape[0], -1)
        if mask.shape[1] != 1:
            raise ValueError(
                f"mask_field '{self.mask_field}' must hold one value per frame or "
                f"per atom, got shape {tuple(mask.shape)}"
            )
        mask = mask[:, 0].to(torch.bool)
        _, _, mask_per_atom = _frame_index(
            ref, mask.shape[0], f"mask_field '{self.mask_field}'"
        )
        if mask.shape[0] == n_rows:
            return mask
        if key_per_atom and not mask_per_atom:
            # per-frame mask on a per-atom field: broadcast to the atoms
            return mask[frame_of_row]
        raise ValueError(
            f"a per-atom mask_field '{self.mask_field}' cannot select frames of "
            "a per-frame loss field"
        )

    def per_frame(self, pred: dict, ref: dict, key: str):
        """Return the per-frame means ``l_f`` (zero for masked-out frames)."""
        ref_key = key if self.ref_key is None else self.ref_key
        prediction, target = _promote(pred[key], ref[ref_key])
        shape = torch.broadcast_shapes(prediction.shape, target.shape)
        n_rows = shape[0]
        n_frames, frame_of_row, per_atom = _frame_index(ref, n_rows, f"'{key}'")
        keep = None
        if self.mask_field is not None:
            keep = self._row_mask(pred, ref, n_rows, per_atom, frame_of_row)
            # masked-out rows compare the prediction with itself, so any
            # element-wise loss and its gradient are exactly zero there, also
            # for NaN placeholder targets; selected rows are untouched
            keep_rows = keep.reshape([n_rows] + [1] * (len(shape) - 1))
            target = torch.where(keep_rows, target, prediction.detach())
        if self.ignore_nan:
            valid = target == target
            target = torch.nan_to_num(target, nan=0.0)
        loss = self.func(prediction, target)
        loss = loss.reshape(n_rows, -1)
        if self.ignore_nan:
            valid = valid.reshape(n_rows, -1)
            loss = loss * valid
            row_count = valid.sum(dim=1).to(loss.dtype)
        else:
            row_count = torch.full(
                (n_rows,), loss.shape[1], dtype=loss.dtype, device=loss.device
            )
        row_sum = loss.sum(dim=1)

        if keep is not None:
            # where() keeps masked rows in the graph with an exact zero value
            row_sum = torch.where(keep, row_sum, torch.zeros_like(row_sum))
            row_count = torch.where(keep, row_count, torch.zeros_like(row_count))

        frame_sum = torch.zeros(n_frames, dtype=loss.dtype, device=loss.device)
        frame_sum = frame_sum.index_add(0, frame_of_row, row_sum)
        frame_count = torch.zeros(n_frames, dtype=loss.dtype, device=loss.device)
        frame_count = frame_count.index_add(0, frame_of_row, row_count)
        return frame_sum / frame_count.clamp_min(1)

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
    ):
        frame_loss = self.per_frame(pred, ref, key)
        if mean:
            return frame_loss.mean()
        return frame_loss

    def sum_and_count(self, pred: dict, ref: dict, key: str):
        """Additive ``(sum of per-frame means, number of frames)``."""
        frame_loss = self.per_frame(pred, ref, key)
        return frame_loss.sum(), torch.as_tensor(
            frame_loss.shape[0], device=frame_loss.device
        )


def find_loss_function(name: str, params):
    """
    Search for loss functions in this module

    If the name starts with PerSpecies, PerAtom or PerFrame, return the
    corresponding wrapper around the torch.nn loss named by the rest of the
    string (e.g. ``PerFrameL1Loss``). A fully qualified dotted import path
    (e.g. ``"my_package.losses.MyLoss"``) is resolved first and then handled
    like the object it names: a class is an element-wise loss with the
    torch.nn contract (``reduction="none"``), any other callable is used
    directly as a loss object with the ``(pred, ref, key, mean)`` signature
    (and ``sum_and_count`` for distributed training).
    """

    wrapper_list = dict(
        perspecies=PerSpeciesLoss,
        peratom=PerAtomLoss,
        perframe=PerFrameLoss,
    )

    if isinstance(name, str) and "." in name:
        # plain strings keep configs loadable with ``yaml.safe_load``
        name = load_callable(name)

    if isinstance(name, str):
        for key in wrapper_list:
            if name.lower().startswith(key):
                logging.debug(f"create loss instance {wrapper_list[key]}")
                return wrapper_list[key](name[len(key) :], params)
        return SimpleLoss(name, params)
    elif inspect.isclass(name):
        return SimpleLoss(name, params)
    elif callable(name):
        return name
    else:
        raise NotImplementedError(f"{name} Loss is not implemented")
