"""Spin moment conditioning (SpinScale) and paramagnetic coordination helper (Mw).

- ``SpinScale`` applies a multiplicative per-species mask and reference moment
  :math:`\\tilde{\\mathbf{s}}_i = (m_{Z_i} / s^{\\mathrm{ref}}_{Z_i}) \\mathbf{s}_i`.
  Because nonmagnetic species carry ``mask_mag[Z] == 0.0`` linearly before any
  nonlinear operation, :math:`\\partial^k E / \\partial \\mathbf{s}_i^k \\equiv 0`
  identically to all derivative orders :math:`k \\ge 1`.
- ``paramagnetic_coordination`` / ``ParamagneticCoordination`` computes the
  spin-independent magnetic neighbor coordination :math:`M_{w,i}`, distinguishing
  demagnetized paramagnetic environments (:math:`\\mathbf{s} \\to 0`) from
  nonmagnetic neighborhoods while contributing exact zero magnetic torque.
"""

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .wigner import _ordinary_tensors

__all__ = [
    "SpinScale",
    "paramagnetic_coordination",
    "ParamagneticCoordination",
]


_SPIN_SCALE_F64_CACHE = {}


def _cached_f64_1d(values: Tuple[float, ...]) -> torch.Tensor:
    t = _SPIN_SCALE_F64_CACHE.get(values)
    if t is None:
        with _ordinary_tensors():
            t = torch.tensor(values, dtype=torch.float64)
        _SPIN_SCALE_F64_CACHE[values] = t
    return t


_cached_f64_1d._dynamo_marked_constant = True


class SpinScale(nn.Module):
    r"""Condition raw atomic spin vectors by per-species magnetic mask and reference magnitude.

    .. math::

        \tilde{\mathbf{s}}_i = \frac{m_{Z_i}}{s^{\mathrm{ref}}_{Z_i}} \, \mathbf{s}_i

    Buffers are registered in ``torch.get_default_dtype()`` for zero FP64->FP32
    runtime casts in FP32 training, while ``_apply()`` re-materializes exact
    53-bit FP64 tables whenever converted to ``float64``.

    Parameters
    ----------
    num_types : int
        Number of real atom species.
    use_spin : sequence of bool or float
        Per-species magnetic flags (length ``num_types``). ``True`` / nonzero marks a
        magnetic species (``mask_mag = 1.0``); ``False`` / ``0`` marks a nonmagnetic
        species (``mask_mag = 0.0``).
    spin_reference : sequence of float or float, optional
        Strictly positive per-species reference spin magnitudes :math:`s^{\mathrm{ref}}_Z`
        in dataset units (e.g. Bohr magnetons). Defaults to ``1.0`` for all species.
    include_padding_type : bool
        If ``True``, appends an extra trailing nonmagnetic padding entry at index
        ``num_types`` (default ``False``).
    """

    def __init__(
        self,
        num_types: int,
        use_spin: Sequence[Union[bool, int, float]],
        spin_reference: Optional[Union[float, Sequence[float]]] = None,
        include_padding_type: bool = False,
    ):
        super().__init__()
        if not (
            isinstance(num_types, int)
            and not isinstance(num_types, bool)
            and num_types >= 1
        ):
            raise ValueError(f"num_types must be a positive integer, got {num_types!r}")
        if len(use_spin) != num_types:
            raise ValueError(
                f"use_spin must have length num_types={num_types}, got {len(use_spin)}"
            )
        self.num_types = num_types
        self.include_padding_type = bool(include_padding_type)

        mask_list = [1.0 if bool(flag) else 0.0 for flag in use_spin]
        if self.include_padding_type:
            mask_list.append(0.0)
        table_len = len(mask_list)

        if spin_reference is None:
            ref_list = [1.0] * table_len
        elif isinstance(spin_reference, (int, float)) and not isinstance(
            spin_reference, bool
        ):
            ref_val = float(spin_reference)
            if ref_val <= 0.0:
                raise ValueError(
                    f"spin_reference must be strictly positive, got {ref_val}"
                )
            ref_list = [ref_val] * table_len
        else:
            ref_list = [float(v) for v in spin_reference]
            if len(ref_list) == num_types and self.include_padding_type:
                ref_list.append(1.0)
            if len(ref_list) != table_len:
                raise ValueError(
                    f"spin_reference length {len(ref_list)} does not match expected {table_len}"
                )
            if any(v <= 0.0 for v in ref_list):
                raise ValueError(
                    f"All spin_reference entries must be strictly positive, got {ref_list}"
                )

        self._mask_mag_f64: Tuple[float, ...] = tuple(mask_list)
        self._spin_ref_f64: Tuple[float, ...] = tuple(ref_list)
        self._combined_f64: Tuple[float, ...] = tuple(
            m / r for m, r in zip(mask_list, ref_list)
        )

        default_dtype = torch.get_default_dtype()
        self.register_buffer("mask_mag", torch.tensor(mask_list, dtype=default_dtype))
        self.register_buffer(
            "spin_reference", torch.tensor(ref_list, dtype=default_dtype)
        )
        self.register_buffer(
            "_combined_scale",
            torch.tensor(self._combined_f64, dtype=default_dtype),
            persistent=False,
        )

    def _apply(self, fn):
        ret = super()._apply(fn)
        if self.mask_mag.dtype == torch.float64:
            self.mask_mag.data.copy_(
                _cached_f64_1d(self._mask_mag_f64).to(device=self.mask_mag.device)
            )
        if self.spin_reference.dtype == torch.float64:
            self.spin_reference.data.copy_(
                _cached_f64_1d(self._spin_ref_f64).to(device=self.spin_reference.device)
            )
        if self._combined_scale.dtype == torch.float64:
            self._combined_scale.data.copy_(
                _cached_f64_1d(self._combined_f64).to(
                    device=self._combined_scale.device
                )
            )
        return ret

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        raw_mask = state_dict.get(prefix + "mask_mag")
        raw_ref = state_dict.get(prefix + "spin_reference")
        mask_ok = (
            isinstance(raw_mask, torch.Tensor) and raw_mask.shape == self.mask_mag.shape
        )
        ref_ok = (
            isinstance(raw_ref, torch.Tensor)
            and raw_ref.shape == self.spin_reference.shape
        )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        mask_src = raw_mask if mask_ok else self.mask_mag
        ref_src = raw_ref if ref_ok else self.spin_reference
        self._mask_mag_f64 = tuple(
            float(v)
            for v in mask_src.detach().to(dtype=torch.float64, device="cpu").tolist()
        )
        self._spin_ref_f64 = tuple(
            float(v)
            for v in ref_src.detach().to(dtype=torch.float64, device="cpu").tolist()
        )
        self._combined_f64 = tuple(
            (m / r if r != 0.0 else 0.0)
            for m, r in zip(self._mask_mag_f64, self._spin_ref_f64)
        )
        self._combined_scale.data.copy_(
            torch.tensor(
                self._combined_f64,
                dtype=self._combined_scale.dtype,
                device=self._combined_scale.device,
            )
        )

    def set_spin_reference(
        self, spin_reference: Union[torch.Tensor, Sequence[float]]
    ) -> None:
        """Update per-species reference spin magnitudes (strictly positive)."""
        ref = torch.as_tensor(
            spin_reference, dtype=torch.float64, device=self.spin_reference.device
        )
        if ref.ndim != 1:
            raise ValueError(f"spin_reference must be 1D, got shape {tuple(ref.shape)}")
        if ref.shape[0] == self.num_types and self.include_padding_type:
            ref = torch.cat([ref, ref.new_ones(1)], dim=0)
        if ref.shape[0] != self.spin_reference.shape[0]:
            raise ValueError(
                f"Expected spin_reference length {self.spin_reference.shape[0]}, got {ref.shape[0]}"
            )
        if bool((ref <= 0.0).any().item()):
            raise ValueError("All spin_reference values must be strictly positive")
        ref_vals = tuple(float(v) for v in ref.detach().cpu().tolist())
        self._spin_ref_f64 = ref_vals
        self._combined_f64 = tuple(
            m / r for m, r in zip(self._mask_mag_f64, self._spin_ref_f64)
        )
        self.spin_reference.copy_(ref.to(dtype=self.spin_reference.dtype))
        self._combined_scale.copy_(
            torch.tensor(
                self._combined_f64,
                dtype=self._combined_scale.dtype,
                device=self._combined_scale.device,
            )
        )

    @torch.no_grad()
    def calibrate_reference(
        self,
        spins: torch.Tensor,
        atom_types: torch.Tensor,
        min_ref: float = 1e-3,
    ) -> torch.Tensor:
        """Estimate per-species RMS spin magnitude from data for magnetic types."""
        norms2 = (spins.to(dtype=torch.float64) ** 2).sum(dim=-1)
        types = atom_types.to(dtype=torch.long)
        new_ref = list(self._spin_ref_f64)
        for t_idx in range(self.num_types):
            if self._mask_mag_f64[t_idx] == 0.0:
                new_ref[t_idx] = 1.0
                continue
            sel = types == t_idx
            if bool(sel.any().item()):
                rms = torch.sqrt(norms2[sel].mean()).item()
                if rms > min_ref:
                    new_ref[t_idx] = rms
        self.set_spin_reference(new_ref)
        return self.spin_reference.clone()

    def is_magnetic(self, atom_types: torch.Tensor) -> torch.Tensor:
        """Return per-atom float magnetic mask ``m_{Z_i}`` of shape ``(...,)``."""
        idx = (
            atom_types if atom_types.dtype == torch.long else atom_types.to(torch.long)
        )
        return self.mask_mag.index_select(0, idx.reshape(-1)).reshape(atom_types.shape)

    def forward(self, spins: torch.Tensor, atom_types: torch.Tensor) -> torch.Tensor:
        """Apply ``(mask_mag[Z_i] / spin_reference[Z_i]) * s_i`` cleanly."""
        if spins.shape[-1] != 3:
            raise ValueError(
                f"spins must have trailing dimension 3, got {tuple(spins.shape)}"
            )
        table = self._combined_scale
        if table.dtype != spins.dtype:
            if spins.dtype == torch.float64:
                table = _cached_f64_1d(self._combined_f64).to(device=spins.device)
            else:
                table = table.to(dtype=spins.dtype)
        idx = (
            atom_types if atom_types.dtype == torch.long else atom_types.to(torch.long)
        )
        scale = table.index_select(0, idx.reshape(-1)).reshape(atom_types.shape)
        return spins * scale.unsqueeze(-1)


def paramagnetic_coordination(
    edge_weights: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    atom_types: torch.Tensor,
    mask_mag: torch.Tensor,
    num_nodes: int,
    normalizer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Compute spin-independent paramagnetic neighbor coordination :math:`M_{w,i}`.

    .. math::

        M_{w,i} = n_i \sum_{j \in \mathcal{N}(i)} w_{ij}\, m_{Z_j}

    where :math:`m_{Z_j} \in \{0, 1\}` is the magnetic species mask of neighbor ``j``.
    Strictly ``0e|0e`` / ``0ee`` invariant and independent of spin vectors :math:`\mathbf{s}`,
    so :math:`\partial M_w / \partial \mathbf{s} \equiv 0` while remaining nonzero in
    demagnetized paramagnetic configurations (:math:`\mathbf{s} \to 0`).
    """
    mask_tbl = (
        mask_mag
        if (
            mask_mag.dtype == edge_weights.dtype
            and mask_mag.device == edge_weights.device
        )
        else mask_mag.to(dtype=edge_weights.dtype, device=edge_weights.device)
    )
    atypes = atom_types if atom_types.dtype == torch.long else atom_types.to(torch.long)
    src = edge_src if edge_src.dtype == torch.long else edge_src.to(torch.long)
    dst = edge_dst if edge_dst.dtype == torch.long else edge_dst.to(torch.long)

    # Gather magnetic mask on N nodes first, then index_select onto E edges
    node_mask = mask_tbl.index_select(0, atypes)
    src_mask = node_mask.index_select(0, src)

    if edge_weights.ndim == 1:
        weighted = edge_weights * src_mask
        out = edge_weights.new_zeros((num_nodes,))
    else:
        mask_view = src_mask.reshape(
            (src_mask.shape[0],) + (1,) * (edge_weights.ndim - 1)
        )
        weighted = edge_weights * mask_view
        out = edge_weights.new_zeros((num_nodes,) + edge_weights.shape[1:])
    out = out.index_add(0, dst, weighted)
    if normalizer is not None:
        if out.ndim > normalizer.ndim:
            normalizer = normalizer.reshape(
                normalizer.shape + (1,) * (out.ndim - normalizer.ndim)
            )
        out = out * normalizer
    return out


class ParamagneticCoordination(nn.Module):
    """Module wrapper around :func:`paramagnetic_coordination` using a ``SpinScale`` mask."""

    def __init__(self, spin_scale: SpinScale):
        super().__init__()
        self.spin_scale = spin_scale

    def forward(
        self,
        edge_weights: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        atom_types: torch.Tensor,
        num_nodes: int,
        normalizer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return paramagnetic_coordination(
            edge_weights=edge_weights,
            edge_src=edge_src,
            edge_dst=edge_dst,
            atom_types=atom_types,
            mask_mag=self.spin_scale.mask_mag,
            num_nodes=num_nodes,
            normalizer=normalizer,
        )
