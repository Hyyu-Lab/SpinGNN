"""Pure-PyTorch ``torch.compile(fullgraph=True, dynamic=True)`` SpinGNN2 v2 models.

Strict 1-hop architecture with zero ``e3nn`` runtime dependency:
- Stage 0: C^3 cutoff envelope, fixed Gaussian RBF, fused 2*H radial SwiGLU MLP,
  ordered pair FiLM, density-floor relative envelope normalization, vacuum-ref
  isolated-atom energy anchor, and polynomial spherical harmonics.
- Stage 1 (Lattice): Single fused lattice scatter + node-level O(3) contractions
  (``0e, 1o, 1e, 2e``) -> ``E_onlyr``, DREAM even-parity Born effective charge
  tensor ``Z_i* in 0ee + 1ee + 2ee`` with soft/hard ASR, and e2SLP lattice vector head.
- Stage 2 (NoSOC): Decoupled ``O(3)_lat x O(3)_spin`` single fused spin scatter
  ``A_i^(l,p|lam,t)``, ``SpinScale``, paramagnetic coordination ``Mw``,
  superexchange bond-angle coupling ``|A_i^(1o|1o)|^2``, zero-init ``spin_gate``,
  and ``E_noSOC``.
- Stage 3 (Thin SOC Perturbation Head): Parameter-free orthogonal ``CoupleBlock``
  and singularity-free bipolar spin harmonics for explicit 2-body DMI
  ``D_ij . (s_hat_i x s_hat_j)``, symmetric anisotropy / Kitaev, and SIA in a
  single fused SOC scatter; strictly zero-initialized so at init ``E_SOC == E_noSOC``
  identically while all SOC parameters receive non-zero gradients.
- Stage 4 (e2SLP + GRACE): Evidential force & spin-torque uncertainty outputs
  (``sigma``, ``Uepi``, ``nu``, ``kappa``) + atomic extrapolation grade ``gamma``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch
import torch._dynamo.config as _dynamo_config
import torch.nn as nn

from spingnn2.symmetry import (
    CoupleBlock,
    DIrrep,
    SparseBispectrumContract,
    SparseCGContract,
    SphericalHarmonics,
    SpinScale,
    bipolar_spin_pair_j1_j2,
    spherical_harmonics,
)

from .readouts import (
    BornEffectiveChargeHead,
    EvidentialHead,
    ExtrapolationGrade,
    build_l2_to_stf_matrix,
)
from .stage0 import (
    C3CutoffEnvelope,
    DensityFloorNorm,
    FixedGaussianRBF,
    OrderedPairFiLM,
    SwiGLUMLP,
    VacuumRefEnergyHead,
    edge_bf16_autocast,
    maybe_edge_checkpoint,
    safe_edge_geometry,
)

# Enable Dynamo tracing of torch.autograd.grad so conservative force/virial/torque
# training compiles into a single fullgraph with zero graph breaks.
_dynamo_config.trace_autograd_ops = True


def _cross_dim1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Explicit 3D cross product along axis 1: ``(N, 3, ...) x (N, 3, ...) -> (N, 3, ...)``."""
    c0 = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    c1 = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    c2 = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return torch.stack((c0, c1, c2), dim=1)


class _HalfGram(nn.Module):
    """Frobenius-isometric upper-triangular Gram contraction over spatial/spin axes."""

    def __init__(self, width: int, drop_first: bool = False) -> None:
        super().__init__()
        self.width = int(width)
        self.drop_first = bool(drop_first)
        rows = []
        cols = []
        scales = []
        for i in range(self.width):
            for j in range(i, self.width):
                rows.append(i)
                cols.append(j)
                scales.append(1.0 if i == j else math.sqrt(2.0))
        if self.drop_first:
            rows = rows[1:]
            cols = cols[1:]
            scales = scales[1:]
        flat_idx = [r * self.width + c for r, c in zip(rows, cols)]
        self._scales_tuple: tuple[float, ...] = tuple(scales)
        self.out_dim = len(flat_idx)
        self.register_buffer("triu_idx", torch.tensor(flat_idx, dtype=torch.long))
        self.register_buffer(
            "triu_scale",
            torch.tensor(scales, dtype=torch.get_default_dtype()),
        )

    def _apply(self, fn):  # type: ignore[override]
        super()._apply(fn)
        if self.triu_scale.dtype == torch.float64:
            exact = torch.tensor(
                self._scales_tuple,
                dtype=torch.float64,
                device=self.triu_scale.device,
            )
            self.triu_scale.copy_(exact)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, ..., C) -> flatten middle axes into M
        n = x.shape[0]
        c = self.width
        x_flat = x.reshape(n, -1, c)
        gram = torch.einsum("nmc,nmd->ncd", x_flat, x_flat).reshape(n, c * c)
        out = gram.index_select(-1, self.triu_idx)
        scale = (
            self.triu_scale
            if self.triu_scale.dtype == x.dtype
            else self.triu_scale.to(dtype=x.dtype)
        )
        return out * scale


@dataclass
class SpinGNNV2Config:
    """Configuration for :class:`SpinGNNV2`."""

    num_types: int = 2
    r_cut: float = 5.5
    lmax: int = 2
    num_rbf: int = 16
    radial_hidden: int = 64
    radial_layers: int = 2
    radial_modes: int = 2
    c0: int = 48
    c1: int = 12
    c2: int = 6
    c3: int = 2
    type_embed_dim: int = 16
    fitting_hidden: int = 64
    fitting_layers: int = 2
    density_floor_factor: float = 8.0
    envelope_mass: float = 1.0
    # Magnetic Stage 2
    has_spin: bool = False
    use_spin: tuple[bool, ...] | None = None
    spin_reference: tuple[float, ...] | None = None
    c_spin: int = 8
    c_spin_01: int = 6
    c_spin_02: int = 3
    c_spin_11: int = 4
    c_spin_21: int = 3
    c_spin_22: int = 2
    # Thin SOC Stage 3
    has_soc: bool = False
    c_soc_dmi: int = 4
    c_soc_kit: int = 4
    # DREAM BEC head
    predict_bec: bool = False
    bec_asr: str = "soft"
    bec_scale: float | tuple[float, ...] = 1.0
    bec_scale_trainable: bool = True
    # e2SLP Stage 4 + GRACE gamma
    predict_e2slp: bool = False
    e2slp_include_f: bool = True
    e2slp_include_sf: bool = False
    e2slp_vec_channels: int = 6
    e2slp_hidden: int = 32
    spin_force_scale: float = 40.0
    predict_gamma: bool = False
    gamma_proj_dim: int = 32
    # Training memory/precision controls
    edge_amp: bool = False
    edge_checkpoint: bool = False


class SpinGNNV2(nn.Module):
    """Unified strict 1-hop SpinGNN2 v2 trainable model (zero ``e3nn``)."""

    def __init__(self, config: SpinGNNV2Config) -> None:
        super().__init__()
        if config.lmax not in (1, 2, 3):
            raise ValueError(f"SpinGNNV2 requires lmax in (1, 2, 3), got {config.lmax}")
        if config.has_soc and not config.has_spin:
            raise ValueError("has_soc=True requires has_spin=True")
        if config.e2slp_include_sf and not config.has_spin:
            raise ValueError("e2slp_include_sf=True requires has_spin=True")

        self.cfg = config
        self.num_types = int(config.num_types)
        self.lmax = int(config.lmax)
        self.c0 = int(config.c0)
        self.c1 = int(config.c1)
        self.c2 = int(config.c2) if self.lmax >= 2 else 0
        self.c3 = int(config.c3) if self.lmax >= 3 else 0
        if self.num_types < 1:
            raise ValueError(f"num_types must be >= 1, got {self.num_types}")
        if self.c0 < 1 or self.c1 < 1:
            raise ValueError(f"c0 and c1 must be >= 1, got c0={self.c0}, c1={self.c1}")
        if self.lmax >= 2 and self.c2 < 1:
            raise ValueError(f"c2 must be >= 1 when lmax >= 2, got {self.c2}")
        if self.lmax >= 3 and self.c3 < 1:
            raise ValueError(f"c3 must be >= 1 when lmax >= 3, got {self.c3}")
        min_c0 = max(
            self.c1,
            self.c2,
            self.c3,
            int(config.c_spin) if config.has_spin else 0,
        )
        if self.c0 < min_c0:
            raise ValueError(
                f"c0 ({self.c0}) must be >= max(c1={self.c1}, c2={self.c2}, "
                f"c3={self.c3}, c_spin={config.c_spin if config.has_spin else 0})"
            )
        self.c1e = max(4, min(self.c1, 8))
        self.has_spin = bool(config.has_spin)
        self.has_soc = bool(config.has_soc)
        self.predict_bec = bool(config.predict_bec)
        self.predict_e2slp = bool(config.predict_e2slp)
        self.predict_gamma = bool(config.predict_gamma)
        self.edge_amp = bool(config.edge_amp)
        self.edge_checkpoint = bool(config.edge_checkpoint)
        default_dt = torch.get_default_dtype()

        # ---- Stage 0: Shared Geometry & Bounded Frontend ----
        self.envelope = C3CutoffEnvelope(config.r_cut, exponent=5)
        self.rbf = FixedGaussianRBF(config.r_cut, num_basis=config.num_rbf)
        self.radial_mlp = SwiGLUMLP(
            in_dim=config.num_rbf,
            hidden_dim=config.radial_hidden,
            out_dim=self.c0 + config.radial_modes,
            num_layers=config.radial_layers,
        )
        self.pair_film = OrderedPairFiLM(
            num_types=self.num_types,
            channels=self.c0,
            num_modes=config.radial_modes,
        )
        self.density_norm = DensityFloorNorm(
            floor_factor=config.density_floor_factor,
            envelope_mass=config.envelope_mass,
        )
        self.edge_sh = SphericalHarmonics(
            self.lmax,
            normalize=False,
            normalization="component",
            parity=-1,
            time_reversal=1,
        )
        self.type_embedding = nn.Embedding(self.num_types, config.type_embed_dim)

        # ---- Stage 1: Lattice Node-Level Contractions (O(N)) ----
        self.mix_1o = nn.Linear(self.c1, self.c1, bias=False)
        self.mix_2e = (
            nn.Linear(self.c2, self.c2, bias=False) if self.lmax >= 2 else None
        )
        # Fused 2*c1e projections to form axial 1ee via 1o x 1o and (lmax>=2) 2e x 2e
        self.axial_1o_ab = nn.Linear(self.c1, 2 * self.c1e, bias=False)
        self.axial_2e_ab = (
            nn.Linear(self.c2, 2 * self.c1e, bias=False) if self.lmax >= 2 else None
        )

        # Register-fused sparse CG & bispectrum contractions (0 CG buffer loads,
        # eliminates 75-80% zero MACs vs dense einsum)
        self.cg_221_contract = (
            SparseCGContract("2ee", "2ee", "1ee") if self.lmax >= 2 else None
        )
        self.cg_112_contract = (
            SparseCGContract("1oe", "1oe", "2ee") if self.lmax >= 2 else None
        )
        self.bisp_112_contract = (
            SparseBispectrumContract(1, 1, 2, mode="pair") if self.lmax >= 2 else None
        )
        self.bisp_222_contract = (
            SparseBispectrumContract(2, 2, 2, mode="channelwise")
            if self.lmax >= 2
            else None
        )

        self.gram_1o = _HalfGram(self.c1)
        self.gram_2e = _HalfGram(self.c2) if self.lmax >= 2 else None
        self.gram_3o = _HalfGram(self.c3) if self.lmax >= 3 else None

        # Bispectrum low-rank projections (K1, K2) for compact O(N) 4-body terms
        self.bisp_k1 = min(self.c1, 6) if self.lmax >= 2 else 0
        self.bisp_k2 = min(self.c2, 4) if self.lmax >= 2 else 0
        self.bisp_proj_1o = (
            nn.Linear(self.c1, self.bisp_k1, bias=False) if self.lmax >= 2 else None
        )
        self.bisp_proj_2e = (
            nn.Linear(self.c2, self.bisp_k2, bias=False) if self.lmax >= 2 else None
        )

        lat_inv_dim = (
            self.c0
            + self.gram_1o.out_dim
            + (self.gram_2e.out_dim if self.gram_2e is not None else 0)
            + (self.gram_3o.out_dim if self.gram_3o is not None else 0)
            + self.bisp_k1 * self.bisp_k2  # (1, 1, 2) bispectrum (0 when lmax=1)
            + self.bisp_k2  # (2, 2, 2) bispectrum (0 when lmax=1)
            + self.c1e  # |H_1e|^2
            + 2  # 1/n0, 1/np density indicators
            + config.type_embed_dim
        )
        self.lat_inv_dim = lat_inv_dim

        # Equivariant gated refinement for H_1o, H_2e, H_1e (strictly 0ee gated, R3)
        self.gate_1o_mlp = nn.Linear(lat_inv_dim, self.c1)
        self.gate_2e_mlp = nn.Linear(lat_inv_dim, self.c2) if self.lmax >= 2 else None
        self.gate_1e_mlp = nn.Linear(lat_inv_dim, self.c1e)

        # Vacuum-anchored lattice energy head
        self.energy_head_lat = VacuumRefEnergyHead(
            in_dim=lat_inv_dim,
            hidden_dim=config.fitting_hidden,
            num_types=self.num_types,
            num_layers=config.fitting_layers,
        )

        # ---- Stage 2: Decoupled O(3)_lat x O(3)_spin Magnetic Backbone ----
        self.nosoc_inv_dim = 0
        if self.has_spin:
            use_spin = (
                tuple(bool(v) for v in config.use_spin)
                if config.use_spin is not None
                else tuple([True] * self.num_types)
            )
            self.spin_scale = SpinScale(
                num_types=self.num_types,
                use_spin=use_spin,
                spin_reference=config.spin_reference,
            )
            self.c_spin = int(config.c_spin)
            self.c_s01 = int(config.c_spin_01)
            self.c_s02 = int(config.c_spin_02)
            self.c_s11 = int(config.c_spin_11)
            self.c_s21 = int(config.c_spin_21) if self.lmax >= 2 else 0
            self.c_s22 = int(config.c_spin_22) if self.lmax >= 2 else 0
            if min(self.c_spin, self.c_s01, self.c_s02, self.c_s11) < 1:
                raise ValueError(
                    "c_spin, c_spin_01, c_spin_02, c_spin_11 must all be >= 1"
                )
            if self.lmax >= 2 and min(self.c_s21, self.c_s22) < 1:
                raise ValueError("c_spin_21 and c_spin_22 must be >= 1 when lmax >= 2")
            max_spin_c = max(
                self.c_spin, self.c_s01, self.c_s02, self.c_s11, self.c_s21, self.c_s22
            )
            self.c_spin_edge = max_spin_c

            self.spin_radial_mlp = SwiGLUMLP(
                in_dim=config.num_rbf
                + 3,  # RBF + [|s_i|^2, |s_j|^2, s_i . s_j] (0e|0e)
                hidden_dim=max(32, config.radial_hidden // 2),
                out_dim=self.c_spin_edge,
                num_layers=1,
            )
            self.spin_pair_film = OrderedPairFiLM(
                num_types=self.num_types,
                channels=self.c_spin_edge,
                num_modes=0,
            )
            # On-site spin scalar weights per species
            self.adam_onsite_s1 = nn.Parameter(torch.ones(self.num_types))
            self.adam_onsite_s2 = nn.Parameter(torch.ones(self.num_types))

            # Per-block channel mixing (strictly preserves (l, p | lam, t))
            self.mix_s01 = nn.Linear(self.c_s01 + 1, self.c_s01 + 1, bias=False)
            self.mix_s02 = nn.Linear(self.c_s02 + 1, self.c_s02 + 1, bias=False)
            self.mix_s11 = nn.Linear(self.c_s11, self.c_s11, bias=False)
            self.mix_s21 = (
                nn.Linear(self.c_s21, self.c_s21, bias=False)
                if self.lmax >= 2
                else None
            )
            self.mix_s22 = (
                nn.Linear(self.c_s22, self.c_s22, bias=False)
                if self.lmax >= 2
                else None
            )

            self.gram_s01 = _HalfGram(self.c_s01 + 1)
            self.gram_s02 = _HalfGram(self.c_s02 + 1, drop_first=True)
            self.gram_s11 = _HalfGram(self.c_s11)
            self.gram_s21 = _HalfGram(self.c_s21) if self.lmax >= 2 else None
            self.gram_s22 = _HalfGram(self.c_s22) if self.lmax >= 2 else None

            self.cross_11_k = min(self.c_s11, self.c2) if self.lmax >= 2 else 0
            self.lat_summary_proj = nn.Linear(lat_inv_dim, 16)

            self.nosoc_inv_dim = (
                self.gram_s01.out_dim
                + self.gram_s02.out_dim
                + self.gram_s11.out_dim
                + (self.gram_s21.out_dim if self.gram_s21 is not None else 0)
                + (self.gram_s22.out_dim if self.gram_s22 is not None else 0)
                + self.cross_11_k  # <H_2e, A_11 . A_11>_(0e|0e) (0 when lmax=1)
                + self.c_spin  # paramagnetic coordination Mw
                + 16  # 0e|0e lattice summary
            )
            self.nosoc_energy_mlp = SwiGLUMLP(
                in_dim=self.nosoc_inv_dim,
                hidden_dim=max(32, config.fitting_hidden // 2),
                out_dim=1,
                num_layers=2,
            )
            # Zero-initialized branch gate so nonmagnetic pretrains start unchanged
            self.adam_spin_gate = nn.Parameter(torch.zeros(1))

            # Spin equivariant 0e|1o vector projection for e2SLP sf_head
            self.spin_vec_proj = nn.Linear(
                self.c_s01 + 1, config.e2slp_vec_channels, bias=False
            )
            self.spin_vec_gate = nn.Linear(
                self.nosoc_inv_dim, config.e2slp_vec_channels
            )

        # ---- Stage 3: Thin Zero-Initialized SOC Perturbation Head ----
        self.soc_inv_dim = 0
        if self.has_soc:
            # Parameter-free orthogonal L-S Couple blocks preserving (p, t)
            self.couple_11 = CoupleBlock(DIrrep("1o|1o"), jmax=2)
            self.couple_21 = (
                CoupleBlock(DIrrep("2e|1o"), jmax=2) if self.lmax >= 2 else None
            )
            self.couple_22 = (
                CoupleBlock(DIrrep("2e|2e"), jmax=2) if self.lmax >= 2 else None
            )

            # Exact l=2 -> STF 3x3 basis for DMI (Q(H_2e) . u_ij) x u_ij
            if self.lmax >= 2:
                self.register_buffer(
                    "l2_to_stf",
                    build_l2_to_stf_matrix(
                        lambda v: spherical_harmonics(2, v, normalize=False)
                    ).to(dtype=default_dt),
                )

            self.c_soc_dmi = int(config.c_soc_dmi)
            self.c_soc_kit = int(config.c_soc_kit) if self.lmax >= 2 else 0
            if self.c_soc_dmi < 1:
                raise ValueError(f"c_soc_dmi must be >= 1, got {self.c_soc_dmi}")
            if self.lmax >= 2 and self.c_soc_kit < 1:
                raise ValueError(
                    f"c_soc_kit must be >= 1 when lmax >= 2, got {self.c_soc_kit}"
                )

            dmi_feat_dim = 3 * self.c_soc_dmi if self.lmax >= 2 else 2 * self.c_soc_dmi
            if self.lmax >= 2:
                self._soc_scatter_splits: tuple[int, ...] = (
                    (
                        dmi_feat_dim,
                        2 * self.c_soc_kit,
                        3 * self.c_soc_dmi,
                        5 * self.c_soc_kit,
                    )
                    if self.predict_bec
                    else (dmi_feat_dim, 2 * self.c_soc_kit)
                )
            else:
                self._soc_scatter_splits = (
                    (dmi_feat_dim, 3 * self.c_soc_dmi)
                    if self.predict_bec
                    else (dmi_feat_dim,)
                )

            soc_edge_out = self.c_soc_dmi + self.c_soc_kit
            # Edge radial weights for DMI and Kitaev (non-zero init so zero-init
            # output weights immediately get non-zero gradients at step 0!)
            self.soc_edge_mlp = SwiGLUMLP(
                in_dim=config.num_rbf,
                hidden_dim=max(24, config.radial_hidden // 2),
                out_dim=soc_edge_out,
                num_layers=1,
            )
            self.soc_pair_film = OrderedPairFiLM(
                num_types=self.num_types,
                channels=soc_edge_out,
                num_modes=0,
            )

            # DMI 1ee lattice axial field projections
            self.dmi_proj_1o = nn.Linear(self.c1, self.c_soc_dmi, bias=False)
            self.dmi_proj_1e = nn.Linear(self.c1e, self.c_soc_dmi, bias=False)
            self.dmi_proj_2e = (
                nn.Linear(self.c2, self.c_soc_dmi, bias=False)
                if self.lmax >= 2
                else None
            )
            # Zero-initialized DMI combination weights -> E_DMI == 0.0 at init, nonzero grad!
            self.soc_dmi_out = nn.Linear(dmi_feat_dim, 1, bias=False)
            nn.init.zeros_(self.soc_dmi_out.weight)

            # Kitaev / symmetric anisotropic exchange & SIA (require l=2 lattice basis)
            if self.lmax >= 2:
                self.kit_proj_2e = nn.Linear(self.c2, self.c_soc_kit, bias=False)
                self.soc_kit_out = nn.Linear(2 * self.c_soc_kit, 1, bias=False)
                nn.init.zeros_(self.soc_kit_out.weight)
                self.soc_sia_out = nn.Linear(self.c2 * (self.c_s02 + 1), 1, bias=False)
                nn.init.zeros_(self.soc_sia_out.weight)
            else:
                self.kit_proj_2e = None
                self.soc_kit_out = None
                self.soc_sia_out = None

            # Coupled node-level L-S invariants (all strictly 0ee!)
            self.gram_c11_j0 = _HalfGram(self.c_s11)
            self.gram_c11_j1 = _HalfGram(self.c_s11)
            self.gram_c11_j2 = _HalfGram(self.c_s11)
            if self.lmax >= 2:
                self.gram_c21_j1 = _HalfGram(self.c_s21)
                self.gram_c21_j2 = _HalfGram(self.c_s21)
                c21_s01_cross = self.c_s21 * (self.c_s01 + 1)
                c22_j0 = self.c_s22  # directly 0ee!
                c22_j1_cross = self.c_s22 * self.c1e  # 1ee . 1ee -> 0ee
                c22_j2_cross = self.c_s22 * self.c2  # 2ee . 2ee -> 0ee
            else:
                self.gram_c21_j1 = None
                self.gram_c21_j2 = None
                c21_s01_cross = 0
                c22_j0 = 0
                c22_j1_cross = 0
                c22_j2_cross = 0

            self.soc_inv_dim = (
                self.gram_c11_j0.out_dim
                + self.gram_c11_j1.out_dim
                + self.gram_c11_j2.out_dim
                + (self.gram_c21_j1.out_dim if self.gram_c21_j1 is not None else 0)
                + (self.gram_c21_j2.out_dim if self.gram_c21_j2 is not None else 0)
                + c21_s01_cross
                + c22_j0
                + c22_j1_cross
                + c22_j2_cross
            )
            self.soc_couple_out = nn.Linear(self.soc_inv_dim, 1, bias=False)
            nn.init.zeros_(self.soc_couple_out.weight)

        # ---- Optional DREAM Born Effective Charge Head ----
        if self.predict_bec:
            mag_c0 = 0
            mag_c1e = 0
            mag_c2e = 0
            if self.has_spin:
                mag_c0 = self.nosoc_inv_dim + (
                    self.c_s22 if (self.has_soc and self.lmax >= 2) else 0
                )
            if self.has_soc:
                mag_c1e = (self.c_s22 if self.lmax >= 2 else 0) + self.c_soc_dmi
                mag_c2e = (self.c_s22 + self.c_soc_kit) if self.lmax >= 2 else 0
            self.bec_head = BornEffectiveChargeHead(
                c0=lat_inv_dim,
                c1e=self.c1e,
                c2e=self.c2,
                num_types=self.num_types,
                y2_fn=lambda v: spherical_harmonics(2, v, normalize=False),
                asr=config.bec_asr,
                per_species_scale=(
                    list(config.bec_scale)
                    if isinstance(config.bec_scale, (tuple, list))
                    else float(config.bec_scale)
                ),
                per_species_scale_trainable=config.bec_scale_trainable,
                mag_c0=mag_c0,
                mag_c1e=mag_c1e,
                mag_c2e=mag_c2e,
            )
        else:
            self.bec_head = None

        # ---- Stage 4: e2SLP Evidential Uncertainty & GRACE Gamma ----
        final_scalar_dim = lat_inv_dim + (self.nosoc_inv_dim if self.has_spin else 0)
        if self.predict_e2slp:
            self.force_vec_proj = nn.Linear(
                self.c1, config.e2slp_vec_channels, bias=False
            )
            self.e2slp_head = EvidentialHead(
                scalar_dim=final_scalar_dim,
                force_vec_channels=config.e2slp_vec_channels,
                spin_vec_channels=(
                    config.e2slp_vec_channels if config.e2slp_include_sf else 0
                ),
                hidden_dim=config.e2slp_hidden,
                include_f=config.e2slp_include_f,
                include_sf=config.e2slp_include_sf,
                spin_force_scale=config.spin_force_scale,
            )
        else:
            self.force_vec_proj = None
            self.e2slp_head = None

        if self.predict_gamma:
            self.gamma_head = ExtrapolationGrade(
                in_dim=final_scalar_dim,
                proj_dim=config.gamma_proj_dim,
            )
        else:
            self.gamma_head = None

        self._compiled_forward: Any | None = None
        self._overflow_forward: Any | None = None
        self._bucketed_config: dict[str, Any] | None = None
        self._bucket_overflow_count: int = 0

    def _apply(self, fn):  # type: ignore[override]
        super()._apply(fn)
        # Re-materialize exact 53-bit FP64 constant tables on .double() / .to(float64)
        if (
            self.has_soc
            and hasattr(self, "l2_to_stf")
            and self.l2_to_stf.dtype == torch.float64
        ):
            dev = self.l2_to_stf.device
            exact_stf = build_l2_to_stf_matrix(
                lambda v: spherical_harmonics(2, v, normalize=False)
            ).to(device=dev)
            self.l2_to_stf.copy_(exact_stf)
        return self

    def compile_fullgraph(
        self,
        dynamic: bool = True,
        **compile_kwargs: Any,
    ) -> "SpinGNNV2":
        """Compile the full forward + conservative autograd pass with zero graph breaks."""
        _dynamo_config.trace_autograd_ops = True
        self._bucketed_config = None
        self._overflow_forward = None
        self._bucket_overflow_count = 0
        self._compiled_forward = torch.compile(
            self._forward_autograd,
            fullgraph=True,
            dynamic=dynamic,
            **compile_kwargs,
        )
        return self

    def compile_bucketed(
        self,
        atom_buckets: Sequence[int] = (64, 128, 256, 512),
        edge_buckets: Sequence[int] | None = None,
        mode: str = "reduce-overhead",
        **compile_kwargs: Any,
    ) -> "SpinGNNV2":
        """Opt-in capped-bucket (<= 4 buckets) CUDA Graph compilation for training.

        Caps the bucket grid to at most 4 shapes to bound CUDAGraphTrees memory
        pool residency. Inputs smaller than the selected bucket are padded with
        disconnected far-away dummy atoms (distance ``2 * r_cut``, ``chi == 0``)
        and sliced back before returning, guaranteeing zero gradient leakage.
        Inputs larger than the largest bucket gracefully fall back to eager
        ``_forward_autograd`` instead of capturing unbounded CUDA Graph pools.
        """
        a_buckets = tuple(sorted(int(x) for x in atom_buckets))
        if not (1 <= len(a_buckets) <= 4):
            raise ValueError(
                f"atom_buckets must contain 1..4 sizes to bound CUDA Graph VRAM, got {a_buckets}"
            )
        e_buckets: tuple[int, ...] | None = None
        if edge_buckets is not None:
            e_buckets = tuple(sorted(int(x) for x in edge_buckets))
            if len(e_buckets) != len(a_buckets):
                raise ValueError(
                    "edge_buckets must have the same length as atom_buckets"
                )
        _dynamo_config.trace_autograd_ops = True
        self._compiled_forward = torch.compile(
            self._forward_autograd,
            fullgraph=True,
            dynamic=False,
            mode=mode,
            **compile_kwargs,
        )
        self._overflow_forward = self._forward_autograd
        self._bucket_overflow_count = 0
        self._bucketed_config = {
            "atom_buckets": a_buckets,
            "edge_buckets": e_buckets,
            "mode": mode,
        }
        return self

    def set_spin_gate(self, value: float) -> None:
        """Set the NoSOC branch gate value explicitly (useful for tests/warm-start)."""
        if not self.has_spin:
            raise RuntimeError("Model has has_spin=False")
        with torch.no_grad():
            self.adam_spin_gate.fill_(float(value))

    def _compute_lat_edge_amp(
        self, rbf: torch.Tensor, pair_idx: torch.Tensor, chi: torch.Tensor
    ) -> torch.Tensor:
        with edge_bf16_autocast(self.edge_amp, rbf):
            raw = self.radial_mlp(rbf)
            g = raw[:, : self.c0]
            modes = raw[:, self.c0 :] if self.cfg.radial_modes > 0 else None
            film = self.pair_film(g, pair_idx, modes)
        if film.dtype != chi.dtype:
            film = film.to(dtype=chi.dtype)
        return chi.unsqueeze(-1) * film

    def _compute_spin_edge_amp(
        self,
        rbf_and_spin: torch.Tensor,
        pair_idx: torch.Tensor,
        chi2: torch.Tensor,
    ) -> torch.Tensor:
        with edge_bf16_autocast(self.edge_amp, rbf_and_spin):
            g_s = self.spin_radial_mlp(rbf_and_spin)
            film_s = self.spin_pair_film(g_s, pair_idx, None)
        if film_s.dtype != chi2.dtype:
            film_s = film_s.to(dtype=chi2.dtype)
        return chi2.unsqueeze(-1) * film_s

    def _compute_soc_edge_amp(
        self,
        rbf: torch.Tensor,
        pair_idx: torch.Tensor,
        chi2: torch.Tensor,
    ) -> torch.Tensor:
        with edge_bf16_autocast(self.edge_amp, rbf):
            g_soc = self.soc_edge_mlp(rbf)
            film_soc = self.soc_pair_film(g_soc, pair_idx, None)
        if film_soc.dtype != chi2.dtype:
            film_soc = film_soc.to(dtype=chi2.dtype)
        return chi2.unsqueeze(-1) * film_soc

    def _build_vacuum_lat_desc(
        self, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        t = self.num_types
        pad_dim = self.lat_inv_dim - 2 - self.cfg.type_embed_dim
        zeros = torch.zeros((t, pad_dim), dtype=dtype, device=device)
        floor = self.density_norm.floor_value
        if floor.dtype != dtype:
            floor = floor.to(dtype=dtype)
        inv_n_vac = torch.sqrt(floor).expand(t, 2)
        type_ids = torch.arange(t, dtype=torch.long, device=device)
        t_emb = self.type_embedding(type_ids)
        if t_emb.dtype != dtype:
            t_emb = t_emb.to(dtype=dtype)
        return torch.cat((zeros, inv_n_vac, t_emb), dim=-1)

    def _build_vacuum_nosoc_desc(self, vac_lat_desc: torch.Tensor) -> torch.Tensor:
        t = self.num_types
        pad_dim = self.nosoc_inv_dim - 16
        zeros = torch.zeros(
            (t, pad_dim), dtype=vac_lat_desc.dtype, device=vac_lat_desc.device
        )
        lat_sum = self.lat_summary_proj(vac_lat_desc)
        return torch.cat((zeros, lat_sum), dim=-1)

    def forward_core(
        self,
        pos: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cell_shift: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        spin: torch.Tensor | None = None,
        num_graphs: int = 1,
        displacement: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Pure-tensor forward pass returning energies and head outputs (zero graph breaks)."""
        n_atoms = pos.shape[0]
        n_edges = edge_index.shape[1]
        dtype = pos.dtype
        device = pos.device

        if batch is None:
            batch = torch.zeros(n_atoms, dtype=torch.long, device=device)

        if displacement is not None:
            sym_disp = 0.5 * (displacement + displacement.transpose(-1, -2))
            atom_disp = sym_disp.index_select(0, batch)
            pos = pos + torch.einsum("ni,nij->nj", pos, atom_disp)
            if cell is not None:
                cell = cell + torch.bmm(cell, sym_disp)

        dst = edge_index[0]
        src = edge_index[1]

        edge_vec = pos.index_select(0, src) - pos.index_select(0, dst)
        if edge_cell_shift is not None and cell is not None:
            edge_batch = batch.index_select(0, dst)
            edge_cell = cell.index_select(0, edge_batch)
            shift_cart = torch.einsum("ei,eij->ej", edge_cell_shift, edge_cell)
            edge_vec = edge_vec + shift_cart

        # ---- Stage 0 ----
        r_ij, u_ij = safe_edge_geometry(edge_vec)
        chi_ij = self.envelope(r_ij)
        chi2_ij = chi_ij * chi_ij
        rbf_ij = self.rbf(r_ij)

        pair_idx = atom_types.index_select(
            0, dst
        ) * self.num_types + atom_types.index_select(0, src)
        phi_lat = maybe_edge_checkpoint(
            self._compute_lat_edge_amp,
            self.edge_checkpoint,
            rbf_ij,
            pair_idx,
            chi_ij,
        )
        n0, np_ = self.density_norm(chi_ij, dst, n_atoms)

        y_edge = self.edge_sh(u_ij)  # (E, (lmax+1)^2)
        y1 = y_edge[:, 1:4]
        y2: torch.Tensor | None = y_edge[:, 4:9] if self.lmax >= 2 else None

        # ---- Stage 1: Clean Inductor-Fused Node Density Scatters ----
        a0 = torch.zeros((n_atoms, self.c0), dtype=dtype, device=device)
        a0 = a0.index_add(0, dst, phi_lat) * n0.unsqueeze(-1)

        chi_phi_1 = chi_ij.unsqueeze(-1) * phi_lat[:, : self.c1]
        a1_edge = y1.unsqueeze(-1) * chi_phi_1.unsqueeze(1)  # (E, 3, C1)
        a1 = torch.zeros((n_atoms, 3, self.c1), dtype=dtype, device=device)
        a1 = a1.index_add(0, dst, a1_edge) * np_.view(-1, 1, 1)

        h1o = self.mix_1o(a1)  # (N, 3, C1), 1oe

        # Axial 1ee vector strictly from even-parity contractions (fused 2*c1e GEMMs):
        u_1o, v_1o = self.axial_1o_ab(a1).chunk(2, dim=-1)
        h1e = _cross_dim1(u_1o, v_1o)  # (N, 3, C1e), 1ee

        g1 = self.gram_1o(h1o)
        lat_parts: list[torch.Tensor] = [a0, g1]

        h2e: torch.Tensor | None = None
        if self.lmax >= 2:
            assert y2 is not None
            assert self.mix_2e is not None
            assert self.axial_2e_ab is not None
            assert self.cg_221_contract is not None
            assert self.gram_2e is not None
            assert self.bisp_proj_1o is not None
            assert self.bisp_proj_2e is not None
            assert self.bisp_112_contract is not None
            assert self.bisp_222_contract is not None

            chi_phi_2 = chi_ij.unsqueeze(-1) * phi_lat[:, : self.c2]
            a2_edge = y2.unsqueeze(-1) * chi_phi_2.unsqueeze(1)  # (E, 5, C2)
            a2 = torch.zeros((n_atoms, 5, self.c2), dtype=dtype, device=device)
            a2 = a2.index_add(0, dst, a2_edge) * np_.view(-1, 1, 1)
            h2e = self.mix_2e(a2)  # (N, 5, C2), 2ee

            u_2e, v_2e = self.axial_2e_ab(a2).chunk(2, dim=-1)
            h1e = h1e + self.cg_221_contract(u_2e, v_2e)  # (N, 3, C1e), 1ee

            g2 = self.gram_2e(h2e)
            lat_parts.append(g2)

            if self.lmax >= 3 and self.gram_3o is not None:
                y3 = y_edge[:, 9:16]
                chi_phi_3 = chi_ij.unsqueeze(-1) * phi_lat[:, : self.c3]
                a3_edge = y3.unsqueeze(-1) * chi_phi_3.unsqueeze(1)
                a3 = torch.zeros((n_atoms, 7, self.c3), dtype=dtype, device=device)
                a3 = a3.index_add(0, dst, a3_edge) * np_.view(-1, 1, 1)
                lat_parts.append(self.gram_3o(a3))

            p1 = self.bisp_proj_1o(h1o)  # (N, 3, K1)
            p2 = self.bisp_proj_2e(h2e)  # (N, 5, K2)
            bisp_112 = self.bisp_112_contract(p1, p1, p2).reshape(
                n_atoms, self.bisp_k1 * self.bisp_k2
            )
            bisp_222 = self.bisp_222_contract(p2, p2, p2)
            lat_parts.extend([bisp_112, bisp_222])

        h1e_sq = (h1e * h1e).sum(dim=1)  # (N, C1e)
        inv_n0 = (1.0 / n0).unsqueeze(-1)
        inv_np = (1.0 / np_).unsqueeze(-1)
        t_emb = self.type_embedding(atom_types)
        if t_emb.dtype != dtype:
            t_emb = t_emb.to(dtype=dtype)

        lat_parts.extend([h1e_sq, inv_n0, inv_np, t_emb])
        d_lat_inv = torch.cat(lat_parts, dim=-1)  # (N, lat_inv_dim), 0ee

        # Equivariant gated node features (strictly 0ee-gated per R3)
        h1o_ref = h1o * torch.sigmoid(self.gate_1o_mlp(d_lat_inv)).unsqueeze(1)
        h2e_ref: torch.Tensor | None = (
            h2e * torch.sigmoid(self.gate_2e_mlp(d_lat_inv)).unsqueeze(1)
            if (self.lmax >= 2 and h2e is not None and self.gate_2e_mlp is not None)
            else None
        )
        h1e_ref = h1e * torch.sigmoid(self.gate_1e_mlp(d_lat_inv)).unsqueeze(1)

        vac_lat_desc = self._build_vacuum_lat_desc(dtype, device)
        e_onlyr = self.energy_head_lat(d_lat_inv, vac_lat_desc, atom_types)
        per_atom_e = e_onlyr

        out: dict[str, torch.Tensor] = {"energy_onlyr": e_onlyr}

        # ---- Stage 2: NoSOC Decoupled O(3)_lat x O(3)_spin ----
        d_nosoc_inv: torch.Tensor | None = None
        spin_vecs_1eo: torch.Tensor | None = None
        s_hat: torch.Tensor | None = None
        spin_mask: torch.Tensor | None = None
        mag_bec_0e: torch.Tensor | None = None
        mag_bec_1e: torch.Tensor | None = None
        mag_bec_2e: torch.Tensor | None = None

        if self.has_spin:
            if spin is None:
                raise ValueError("Model has has_spin=True but `spin` was not provided")
            s_hat = self.spin_scale(spin, atom_types)  # (N, 3)
            spin_mask_raw = self.spin_scale.is_magnetic(atom_types)
            spin_mask = (
                spin_mask_raw
                if spin_mask_raw.dtype == dtype
                else spin_mask_raw.to(dtype=dtype)
            )

            y_s1 = spherical_harmonics(1, s_hat, normalize=False)  # (N, 3), 0e|1o
            y_s2 = spherical_harmonics(2, s_hat, normalize=False)  # (N, 5), 0e|2e

            s_i = s_hat.index_select(0, dst)
            s_j = s_hat.index_select(0, src)
            s_pair_inv = torch.stack(
                (
                    (s_i * s_i).sum(dim=-1),
                    (s_j * s_j).sum(dim=-1),
                    (s_i * s_j).sum(dim=-1),
                ),
                dim=-1,
            )  # (E, 3), strictly 0e|0e
            rbf_spin_in = torch.cat((rbf_ij, s_pair_inv), dim=-1)
            phi_spin = maybe_edge_checkpoint(
                self._compute_spin_edge_amp,
                self.edge_checkpoint,
                rbf_spin_in,
                pair_idx,
                chi2_ij,
            )

            # Paramagnetic coordination Mw (strictly spin-value-independent 0e|0e!)
            src_mag_mask = spin_mask.index_select(0, src)
            mw_edge = (chi2_ij * src_mag_mask).unsqueeze(-1) * phi_lat[:, : self.c_spin]
            mw = torch.zeros((n_atoms, self.c_spin), dtype=dtype, device=device)
            mw = mw.index_add(0, dst, mw_edge) * np_.unsqueeze(-1)

            y_s1_j = y_s1.index_select(0, src)  # (E, 3)
            y_s2_j = y_s2.index_select(0, src)  # (E, 5)

            # Two-factor node accumulation A_i^(l,p|lam,t) using direct broadcasts
            # (zero 3-operand einsum -> zero opt_einsum edge-count specialization!)
            a01_edge = y_s1_j.unsqueeze(-1) * phi_spin[:, : self.c_s01].unsqueeze(1)
            a01_nbr = torch.zeros((n_atoms, 3, self.c_s01), dtype=dtype, device=device)
            a01_nbr = a01_nbr.index_add(0, dst, a01_edge) * np_.view(-1, 1, 1)
            onsite_w1 = self.adam_onsite_s1.index_select(0, atom_types)
            if onsite_w1.dtype != dtype:
                onsite_w1 = onsite_w1.to(dtype=dtype)
            a01_onsite = (y_s1 * onsite_w1.unsqueeze(-1)).unsqueeze(-1)
            a01 = self.mix_s01(torch.cat((a01_onsite, a01_nbr), dim=-1)).unsqueeze(1)

            a02_edge = y_s2_j.unsqueeze(-1) * phi_spin[:, : self.c_s02].unsqueeze(1)
            a02_nbr = torch.zeros((n_atoms, 5, self.c_s02), dtype=dtype, device=device)
            a02_nbr = a02_nbr.index_add(0, dst, a02_edge) * np_.view(-1, 1, 1)
            onsite_w2 = self.adam_onsite_s2.index_select(0, atom_types)
            if onsite_w2.dtype != dtype:
                onsite_w2 = onsite_w2.to(dtype=dtype)
            a02_onsite = (y_s2 * onsite_w2.unsqueeze(-1)).unsqueeze(-1)
            a02 = self.mix_s02(torch.cat((a02_onsite, a02_nbr), dim=-1)).unsqueeze(1)

            a11_edge = (y1.unsqueeze(-1) * y_s1_j.unsqueeze(-2)).unsqueeze(
                -1
            ) * phi_spin[:, None, None, : self.c_s11]
            a11 = torch.zeros((n_atoms, 3, 3, self.c_s11), dtype=dtype, device=device)
            a11 = self.mix_s11(a11.index_add(0, dst, a11_edge) * np_.view(-1, 1, 1, 1))

            # Decoupled 0e|0e invariants
            g_s01 = self.gram_s01(a01)
            g_s02 = self.gram_s02(a02)
            g_s11 = self.gram_s11(a11)
            nosoc_parts: list[torch.Tensor] = [g_s01, g_s02, g_s11]

            a21: torch.Tensor | None = None
            a22: torch.Tensor | None = None
            if self.lmax >= 2:
                assert y2 is not None
                assert h2e_ref is not None
                assert self.mix_s21 is not None
                assert self.mix_s22 is not None
                assert self.gram_s21 is not None
                assert self.gram_s22 is not None
                assert self.cg_112_contract is not None

                a21_edge = (y2.unsqueeze(-1) * y_s1_j.unsqueeze(-2)).unsqueeze(
                    -1
                ) * phi_spin[:, None, None, : self.c_s21]
                a21 = torch.zeros(
                    (n_atoms, 5, 3, self.c_s21), dtype=dtype, device=device
                )
                a21 = self.mix_s21(
                    a21.index_add(0, dst, a21_edge) * np_.view(-1, 1, 1, 1)
                )

                a22_edge = (y2.unsqueeze(-1) * y_s2_j.unsqueeze(-2)).unsqueeze(
                    -1
                ) * phi_spin[:, None, None, : self.c_s22]
                a22 = torch.zeros(
                    (n_atoms, 5, 5, self.c_s22), dtype=dtype, device=device
                )
                a22 = self.mix_s22(
                    a22.index_add(0, dst, a22_edge) * np_.view(-1, 1, 1, 1)
                )

                g_s21 = self.gram_s21(a21)
                g_s22 = self.gram_s22(a22)

                # 3-body decoupled cross invariant <H_2e, A_11 . A_11>_(0e|0e)
                k_c = self.cross_11_k
                a11_t = a11[..., :k_c].permute(0, 2, 1, 3)  # (N, 3_spin, 3_lat, k_c)
                a11_q2 = self.cg_112_contract(a11_t, a11_t).sum(dim=1)  # (N, 5, k_c)
                cross_11_2 = (a11_q2 * h2e_ref[..., :k_c]).sum(dim=1)
                nosoc_parts.extend([g_s21, g_s22, cross_11_2])

            lat_sum = self.lat_summary_proj(d_lat_inv)
            nosoc_parts.extend([mw, lat_sum])
            d_nosoc_inv = torch.cat(nosoc_parts, dim=-1)

            # Vacuum-anchored NoSOC energy delta gated by zero-init spin_gate
            vac_nosoc_desc = self._build_vacuum_nosoc_desc(vac_lat_desc)
            raw_nosoc_e = self.nosoc_energy_mlp(d_nosoc_inv).squeeze(-1)
            vac_nosoc_e = (
                self.nosoc_energy_mlp(vac_nosoc_desc)
                .squeeze(-1)
                .index_select(0, atom_types)
            )
            spin_gate = (
                self.adam_spin_gate
                if self.adam_spin_gate.dtype == dtype
                else self.adam_spin_gate.to(dtype=dtype)
            )
            e_nosoc_delta = spin_gate * (raw_nosoc_e - vac_nosoc_e)
            e_nosoc = e_onlyr + e_nosoc_delta
            per_atom_e = e_nosoc
            out["energy_nosoc"] = e_nosoc
            out["paramagnetic_mw"] = mw

            # 0e|1o equivariant spin vectors for e2SLP sf_head
            spin_vecs_raw = self.spin_vec_proj(a01.squeeze(1))  # (N, 3, C_vec)
            spin_vecs_1eo = spin_vecs_raw * torch.sigmoid(
                self.spin_vec_gate(d_nosoc_inv)
            ).unsqueeze(1)
            mag_bec_0e = d_nosoc_inv

            # ---- Stage 3: Thin Zero-Initialized SOC Perturbation Head ----
            if self.has_soc:
                phi_soc = maybe_edge_checkpoint(
                    self._compute_soc_edge_amp,
                    self.edge_checkpoint,
                    rbf_ij,
                    pair_idx,
                    chi2_ij,
                )
                phi_dmi = phi_soc[:, : self.c_soc_dmi]

                # Closed-form fused singularity-free bipolar spin harmonics B^{1,2}_{1,1}(s_i, s_j)
                b_cross, b_quad = bipolar_spin_pair_j1_j2(s_i, s_j)

                # 1) Explicit 2-body DMI: D_ij (1ee) . (s_i x s_j) (1ee) -> 0ee
                d_1o = self.dmi_proj_1o(h1o_ref).index_select(0, dst)  # (E, 3, C_dmi)
                d_1e = self.dmi_proj_1e(h1e_ref).index_select(0, dst)  # (E, 3, C_dmi)
                u_exp = u_ij.unsqueeze(-1).expand(-1, -1, self.c_soc_dmi)
                dmi_field_1 = _cross_dim1(d_1o, u_exp)  # 1o x 1o -> 1ee
                dmi_field_2 = d_1e  # 1ee

                b_cross_u = b_cross.unsqueeze(-1)
                dmi_e1 = (dmi_field_1 * b_cross_u).sum(dim=1) * phi_dmi
                dmi_e2 = (dmi_field_2 * b_cross_u).sum(dim=1) * phi_dmi

                if self.lmax >= 2:
                    assert y2 is not None
                    assert h2e_ref is not None
                    assert self.dmi_proj_2e is not None
                    assert self.kit_proj_2e is not None
                    assert self.soc_kit_out is not None
                    assert self.soc_sia_out is not None
                    assert self.couple_21 is not None
                    assert self.couple_22 is not None
                    assert self.gram_c21_j1 is not None
                    assert self.gram_c21_j2 is not None
                    assert a21 is not None and a22 is not None

                    phi_kit = phi_soc[:, self.c_soc_dmi :]
                    d_2e_node = self.dmi_proj_2e(h2e_ref)  # (N, 5, C_dmi)
                    stf_raw = (
                        self.l2_to_stf
                        if self.l2_to_stf.dtype == dtype
                        else self.l2_to_stf.to(dtype=dtype)
                    )
                    stf_basis = stf_raw.reshape(5, 3, 3)
                    q_2e_node = torch.einsum("nmc,mab->nabc", d_2e_node, stf_basis)
                    q_2e_edge = q_2e_node.index_select(0, dst)
                    qu_edge = torch.einsum("eabc,eb->eac", q_2e_edge, u_ij)
                    dmi_field_3 = _cross_dim1(qu_edge, u_exp)  # (2e.1o) x 1o -> 1ee
                    dmi_e3 = (dmi_field_3 * b_cross_u).sum(dim=1) * phi_dmi
                    dmi_edge_feat = torch.cat((dmi_e1, dmi_e2, dmi_e3), dim=-1)

                    # 2) Explicit 2-body Kitaev / Symmetric Anisotropic Exchange (2ee . 2ee -> 0ee)
                    k_node_2e = self.kit_proj_2e(h2e_ref).index_select(0, dst)
                    kit_bond = (y2 * b_quad).sum(dim=-1, keepdim=True) * phi_kit
                    kit_env = (k_node_2e * b_quad.unsqueeze(-1)).sum(dim=1) * phi_kit
                    kit_edge_feat = torch.cat((kit_bond, kit_env), dim=-1)

                    soc_edge_parts = [dmi_edge_feat, kit_edge_feat]
                    if self.predict_bec:
                        b_cross_edge = (b_cross_u * phi_dmi.unsqueeze(1)).reshape(
                            n_edges, 3 * self.c_soc_dmi
                        )
                        b_quad_edge = (
                            b_quad.unsqueeze(-1) * phi_kit.unsqueeze(1)
                        ).reshape(n_edges, 5 * self.c_soc_kit)
                        soc_edge_parts.extend([b_cross_edge, b_quad_edge])

                    soc_edge_pack = torch.cat(soc_edge_parts, dim=-1)
                    soc_node_pack = torch.zeros(
                        (n_atoms, soc_edge_pack.shape[-1]), dtype=dtype, device=device
                    )
                    soc_node_pack = soc_node_pack.index_add(
                        0, dst, soc_edge_pack
                    ) * np_.unsqueeze(-1)
                    soc_unpacked = torch.split(
                        soc_node_pack, self._soc_scatter_splits, dim=-1
                    )

                    e_dmi = self.soc_dmi_out(soc_unpacked[0]).squeeze(-1)
                    e_kit = self.soc_kit_out(soc_unpacked[1]).squeeze(-1)

                    # 3) Single-Ion Anisotropy (SIA): <H_2e, A_02> (2ee . 2ee -> 0ee)
                    a02_mat = a02.squeeze(1)  # (N, 5, c_s02 + 1)
                    sia_feat = torch.einsum("nmc,nmd->ncd", h2e_ref, a02_mat).reshape(
                        n_atoms, self.c2 * (self.c_s02 + 1)
                    )
                    e_sia = self.soc_sia_out(sia_feat).squeeze(-1)

                    # 4) Parameter-free orthogonal CoupleBlock L-S contractions (strictly 0ee)
                    c11 = self.couple_11(a11)  # {0: 0oo, 1: 1oo, 2: 2oo}
                    c21 = self.couple_21(a21)  # {1: 1eo, 2: 2eo}
                    c22 = self.couple_22(a22)  # {0: 0ee, 1: 1ee, 2: 2ee}

                    c21_s01 = torch.einsum(
                        "nmc,nmd->ncd", c21[1], a01.squeeze(1)
                    ).reshape(n_atoms, self.c_s21 * (self.c_s01 + 1))
                    c22_j0 = c22[0].squeeze(1)  # (N, c_s22), strictly 0ee!
                    c22_j1_h1e = torch.einsum("nmc,nmd->ncd", c22[1], h1e_ref).reshape(
                        n_atoms, self.c_s22 * self.c1e
                    )
                    c22_j2_h2e = torch.einsum("nmc,nmd->ncd", c22[2], h2e_ref).reshape(
                        n_atoms, self.c_s22 * self.c2
                    )
                    d_couple_0ee = torch.cat(
                        (
                            self.gram_c11_j0(c11[0]),
                            self.gram_c11_j1(c11[1]),
                            self.gram_c11_j2(c11[2]),
                            self.gram_c21_j1(c21[1]),
                            self.gram_c21_j2(c21[2]),
                            c21_s01,
                            c22_j0,
                            c22_j1_h1e,
                            c22_j2_h2e,
                        ),
                        dim=-1,
                    )
                    e_couple = self.soc_couple_out(d_couple_0ee).squeeze(-1)
                    e_soc_delta = e_dmi + e_kit + e_sia + e_couple

                    if self.predict_bec:
                        mag_bec_0e = torch.cat((d_nosoc_inv, c22_j0), dim=-1)
                        b_cross_node = soc_unpacked[2].reshape(
                            n_atoms, 3, self.c_soc_dmi
                        )
                        mag_bec_1e = torch.cat((c22[1], b_cross_node), dim=-1)
                        b_quad_node = soc_unpacked[3].reshape(
                            n_atoms, 5, self.c_soc_kit
                        )
                        mag_bec_2e = torch.cat((c22[2], b_quad_node), dim=-1)
                else:
                    dmi_edge_feat = torch.cat((dmi_e1, dmi_e2), dim=-1)
                    soc_edge_parts = [dmi_edge_feat]
                    if self.predict_bec:
                        b_cross_edge = (b_cross_u * phi_dmi.unsqueeze(1)).reshape(
                            n_edges, 3 * self.c_soc_dmi
                        )
                        soc_edge_parts.append(b_cross_edge)
                    soc_edge_pack = torch.cat(soc_edge_parts, dim=-1)
                    soc_node_pack = torch.zeros(
                        (n_atoms, soc_edge_pack.shape[-1]), dtype=dtype, device=device
                    )
                    soc_node_pack = soc_node_pack.index_add(
                        0, dst, soc_edge_pack
                    ) * np_.unsqueeze(-1)
                    soc_unpacked = torch.split(
                        soc_node_pack, self._soc_scatter_splits, dim=-1
                    )
                    e_dmi = self.soc_dmi_out(soc_unpacked[0]).squeeze(-1)

                    c11 = self.couple_11(a11)
                    d_couple_0ee = torch.cat(
                        (
                            self.gram_c11_j0(c11[0]),
                            self.gram_c11_j1(c11[1]),
                            self.gram_c11_j2(c11[2]),
                        ),
                        dim=-1,
                    )
                    e_couple = self.soc_couple_out(d_couple_0ee).squeeze(-1)
                    e_soc_delta = e_dmi + e_couple

                    if self.predict_bec:
                        mag_bec_1e = soc_unpacked[1].reshape(n_atoms, 3, self.c_soc_dmi)

                e_soc = e_nosoc + e_soc_delta
                per_atom_e = e_soc
                out["energy_soc_delta"] = e_soc_delta

        # ---- Graph Total Energy ----
        if num_graphs == 1:
            total_energy = per_atom_e.sum(dim=0, keepdim=True)
        else:
            total_energy = torch.zeros((num_graphs,), dtype=dtype, device=device)
            total_energy = total_energy.index_add(0, batch, per_atom_e)
        out["per_atom_energy"] = per_atom_e
        out["total_energy"] = total_energy

        # ---- DREAM BEC Head ----
        if self.bec_head is not None:
            bec, total_bec = self.bec_head(
                h_0e=d_lat_inv,
                h_1e=h1e_ref,
                h_2e=h2e_ref,
                atom_types=atom_types,
                batch=batch,
                num_graphs=num_graphs,
                mag_0e=mag_bec_0e,
                mag_1e=mag_bec_1e,
                mag_2e=mag_bec_2e,
            )
            out["born_effective_charges"] = bec
            out["total_born_effective_charges"] = total_bec

        # ---- Stage 4: e2SLP & GRACE Gamma ----
        final_scalars = (
            torch.cat((d_lat_inv, d_nosoc_inv), dim=-1)
            if (self.has_spin and d_nosoc_inv is not None)
            else d_lat_inv
        )
        if self.e2slp_head is not None and self.force_vec_proj is not None:
            f_vecs = self.force_vec_proj(h1o_ref)  # (N, 3, C_vec), 1oe
            e2_out = self.e2slp_head(
                node_scalars=final_scalars,
                force_vectors=f_vecs,
                batch=batch,
                num_graphs=num_graphs,
                spin_vectors=spin_vecs_1eo,
                spins=s_hat,
                spin_mask=spin_mask,
            )
            out.update(e2_out)

        if self.gamma_head is not None:
            out["gamma"] = self.gamma_head(final_scalars)

        return out

    def _forward_autograd(
        self,
        pos: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cell_shift: torch.Tensor | None,
        cell: torch.Tensor | None,
        batch: torch.Tensor,
        spin: torch.Tensor | None,
        displacement: torch.Tensor | None,
        num_graphs: int,
        compute_forces: bool,
        compute_virial: bool,
        compute_spin_forces: bool,
        create_graph: bool,
    ) -> dict[str, torch.Tensor]:
        """Compile-safe inner forward + ``torch.autograd.grad`` pass (zero graph breaks)."""
        out = self.forward_core(
            pos=pos,
            atom_types=atom_types,
            edge_index=edge_index,
            edge_cell_shift=edge_cell_shift,
            cell=cell,
            batch=batch,
            spin=spin,
            num_graphs=num_graphs,
            displacement=displacement,
        )

        grad_inputs: list[torch.Tensor] = []
        grad_keys: list[str] = []
        if compute_forces:
            grad_inputs.append(pos)
            grad_keys.append("forces")
        if compute_virial and displacement is not None:
            grad_inputs.append(displacement)
            grad_keys.append("virial")
        if compute_spin_forces and spin is not None:
            grad_inputs.append(spin)
            grad_keys.append("spin_forces")

        if grad_inputs:
            energy_sum = out["total_energy"].sum()
            grads = torch.autograd.grad(
                outputs=[energy_sum],
                inputs=grad_inputs,
                create_graph=bool(create_graph),
                retain_graph=bool(create_graph),
                allow_unused=True,
            )
            for key, g, inp in zip(grad_keys, grads, grad_inputs):
                if g is None:
                    out[key] = torch.zeros_like(inp)
                else:
                    out[key] = -g

            if "spin_forces" in out and spin is not None:
                fm = out["spin_forces"]
                s_sq = (spin * spin).sum(dim=-1, keepdim=True)
                s_thresh = (
                    1e-24
                    if spin.dtype == torch.float64
                    else (1e-4 if spin.dtype == torch.float16 else 1e-12)
                )
                nonzero = s_sq > s_thresh
                safe_s_sq = torch.where(nonzero, s_sq, torch.ones_like(s_sq))
                s_hat_unit = torch.where(
                    nonzero, spin * torch.rsqrt(safe_s_sq), torch.zeros_like(spin)
                )
                radial_comp = (fm * s_hat_unit).sum(dim=-1, keepdim=True)
                out["spin_forces_vert"] = fm - radial_comp * s_hat_unit

        return out

    def _select_bucket(self, n_atoms: int, n_edges: int) -> tuple[int, int] | None:
        if self._bucketed_config is None:
            return None
        a_buckets: tuple[int, ...] = self._bucketed_config["atom_buckets"]
        e_buckets: tuple[int, ...] | None = self._bucketed_config["edge_buckets"]
        for idx, ab in enumerate(a_buckets):
            if n_atoms <= ab:
                if e_buckets is None:
                    return (ab, n_edges)
                eb = e_buckets[idx]
                if n_edges <= eb:
                    # Need 2 dummy atoms if padding edges
                    if n_edges < eb and n_atoms > ab - 2:
                        continue
                    return (ab, eb)
        return None

    def _run_bucketed(
        self,
        bucket: tuple[int, int],
        pos_in: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cell_shift: torch.Tensor | None,
        cell: torch.Tensor | None,
        batch: torch.Tensor,
        spin_in: torch.Tensor | None,
        disp_in: torch.Tensor | None,
        num_graphs: int,
        compute_forces: bool,
        compute_virial: bool,
        compute_spin_forces: bool,
        create_graph: bool,
    ) -> dict[str, torch.Tensor]:
        n_atoms = pos_in.shape[0]
        n_edges = edge_index.shape[1]
        n_pad, e_pad = bucket
        pad_a = n_pad - n_atoms
        pad_e = e_pad - n_edges

        if pad_a == 0 and pad_e == 0:
            assert self._compiled_forward is not None
            return self._compiled_forward(
                pos_in,
                atom_types,
                edge_index,
                edge_cell_shift,
                cell,
                batch,
                spin_in,
                disp_in,
                num_graphs,
                compute_forces,
                compute_virial,
                compute_spin_forces,
                create_graph,
            )

        dev = pos_in.device
        dt = pos_in.dtype
        # Place dummy atoms in a separate dummy graph index so real-graph
        # reductions (total_energy, ASR, e2SLP) are 100% isolated.
        num_graphs_pad = num_graphs + 1
        dummy_pos = torch.zeros((pad_a, 3), dtype=dt, device=dev)
        if pad_a >= 1:
            dummy_pos[-1, 0] = 2.0 * self.cfg.r_cut
        pos_pad = torch.cat((pos_in.detach(), dummy_pos), dim=0)
        if compute_forces:
            pos_pad = pos_pad.requires_grad_(True)
        at_pad = torch.cat(
            (atom_types, torch.zeros((pad_a,), dtype=atom_types.dtype, device=dev)),
            dim=0,
        )
        batch_pad = torch.cat(
            (batch, torch.full((pad_a,), num_graphs, dtype=batch.dtype, device=dev)),
            dim=0,
        )

        if pad_e > 0:
            # Dummy edges connect dummy atom (n_pad - 2) -> (n_pad - 1) at
            # distance 2 * r_cut (> r_cut), so chi == 0 identically.
            d0 = n_pad - 2
            d1 = n_pad - 1
            dummy_dst = torch.full((pad_e,), d0, dtype=edge_index.dtype, device=dev)
            dummy_src = torch.full((pad_e,), d1, dtype=edge_index.dtype, device=dev)
            ei_pad = torch.cat(
                (edge_index, torch.stack((dummy_dst, dummy_src), dim=0)), dim=1
            )
        else:
            ei_pad = edge_index

        ecs_pad = edge_cell_shift
        if ecs_pad is not None and pad_e > 0:
            ecs_pad = torch.cat(
                (ecs_pad, torch.zeros((pad_e, 3), dtype=ecs_pad.dtype, device=dev)),
                dim=0,
            )

        cell_pad = cell
        if cell_pad is not None:
            dummy_cell = torch.eye(3, dtype=cell_pad.dtype, device=dev).unsqueeze(0) * (
                4.0 * self.cfg.r_cut
            )
            cell_pad = torch.cat((cell_pad, dummy_cell), dim=0)

        spin_pad = spin_in
        if spin_pad is not None:
            dummy_spin = torch.zeros((pad_a, 3), dtype=spin_pad.dtype, device=dev)
            dummy_spin[:, 0] = 1.0
            spin_pad = torch.cat((spin_pad.detach(), dummy_spin), dim=0)
            if compute_spin_forces:
                spin_pad = spin_pad.requires_grad_(True)

        disp_pad = disp_in
        if disp_pad is not None:
            dummy_disp = torch.zeros((1, 3, 3), dtype=disp_pad.dtype, device=dev)
            disp_pad = torch.cat((disp_pad.detach(), dummy_disp), dim=0)
            if compute_virial:
                disp_pad = disp_pad.requires_grad_(True)

        assert self._compiled_forward is not None
        raw_out = self._compiled_forward(
            pos_pad,
            at_pad,
            ei_pad,
            ecs_pad,
            cell_pad,
            batch_pad,
            spin_pad,
            disp_pad,
            num_graphs_pad,
            compute_forces,
            compute_virial,
            compute_spin_forces,
            create_graph,
        )

        # Slice back to exact real atom / real graph shapes
        node_keys = {
            "per_atom_energy",
            "energy_onlyr",
            "energy_nosoc",
            "energy_soc_delta",
            "paramagnetic_mw",
            "forces",
            "spin_forces",
            "spin_forces_vert",
            "born_effective_charges",
            "gamma",
            "f_sigma",
            "f_Uepi",
            "f_nu",
            "f_kappa",
            "sf_sigma",
            "sf_Uepi",
            "sf_nu",
            "sf_kappa",
        }
        graph_keys = {
            "total_energy",
            "virial",
            "total_born_effective_charges",
            "f_total_Uepi",
            "f_uncertainty",
            "sf_total_Uepi",
            "sf_uncertainty",
            "uncertainty",
        }
        sliced: dict[str, torch.Tensor] = {}
        for k, v in raw_out.items():
            if k in node_keys:
                sliced[k] = v[:n_atoms]
            elif k in graph_keys:
                sliced[k] = v[:num_graphs]
            else:
                sliced[k] = v
        return sliced

    def forward(
        self,
        pos: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cell_shift: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        spin: torch.Tensor | None = None,
        num_graphs: int = 1,
        compute_forces: bool = True,
        compute_virial: bool = False,
        compute_spin_forces: bool | None = None,
        create_graph: bool | None = None,
        displacement: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with exact autograd forces, virial, and magnetic torques.

        When ``torch.compile(model, fullgraph=True, dynamic=True)`` is applied
        directly, pass ``pos`` (with ``requires_grad=True``), ``spin`` (with
        ``requires_grad=True`` when magnetic), and ``displacement`` (with
        ``requires_grad=True`` when ``compute_virial=True``), or call
        :meth:`compile_fullgraph` / :meth:`compile_bucketed` to manage input leaf
        setup automatically.
        """
        if compute_spin_forces is None:
            compute_spin_forces = self.has_spin
        if create_graph is None:
            create_graph = self.training

        n_atoms = pos.shape[0]
        if batch is None:
            batch = torch.zeros(n_atoms, dtype=torch.long, device=pos.device)

        pos_in = pos
        if compute_forces and not pos_in.requires_grad:
            pos_in = pos_in.detach().requires_grad_(True)

        spin_in = spin
        if compute_spin_forces and spin_in is not None and not spin_in.requires_grad:
            spin_in = spin_in.detach().requires_grad_(True)

        disp_in = displacement
        if compute_virial and disp_in is None:
            disp_in = torch.zeros(
                (num_graphs, 3, 3),
                dtype=pos_in.dtype,
                device=pos_in.device,
                requires_grad=True,
            )

        if self._bucketed_config is not None and self._compiled_forward is not None:
            bucket = self._select_bucket(n_atoms, edge_index.shape[1])
            if bucket is not None:
                return self._run_bucketed(
                    bucket,
                    pos_in,
                    atom_types,
                    edge_index,
                    edge_cell_shift,
                    cell,
                    batch,
                    spin_in,
                    disp_in,
                    num_graphs,
                    bool(compute_forces),
                    bool(compute_virial),
                    bool(compute_spin_forces),
                    bool(create_graph),
                )
            # Out-of-bucket overflow: gracefully fall back to eager _forward_autograd
            # instead of invoking the static reduce-overhead CUDA Graph on arbitrary sizes.
            self._bucket_overflow_count += 1
            overflow_fn = (
                self._overflow_forward
                if self._overflow_forward is not None
                else self._forward_autograd
            )
            return overflow_fn(
                pos_in,
                atom_types,
                edge_index,
                edge_cell_shift,
                cell,
                batch,
                spin_in,
                disp_in,
                num_graphs,
                bool(compute_forces),
                bool(compute_virial),
                bool(compute_spin_forces),
                bool(create_graph),
            )

        fn = (
            self._compiled_forward
            if self._compiled_forward is not None
            else self._forward_autograd
        )
        return fn(
            pos_in,
            atom_types,
            edge_index,
            edge_cell_shift,
            cell,
            batch,
            spin_in,
            disp_in,
            num_graphs,
            bool(compute_forces),
            bool(compute_virial),
            bool(compute_spin_forces),
            bool(create_graph),
        )


def build_v2_nosoc(
    num_types: int = 2,
    r_cut: float = 5.5,
    use_spin: Sequence[bool] | None = None,
    **kwargs: Any,
) -> SpinGNNV2:
    """Build a ``v2_nosoc`` model (Stage 0 + 1 + 2, strict ``O(3)_lat x O(3)_spin``)."""
    kwargs.setdefault("predict_bec", False)
    kwargs.setdefault("predict_e2slp", False)
    cfg = SpinGNNV2Config(
        num_types=num_types,
        r_cut=r_cut,
        has_spin=True,
        has_soc=False,
        use_spin=tuple(use_spin) if use_spin is not None else None,
        **kwargs,
    )
    return SpinGNNV2(cfg)


def build_v2_soc(
    num_types: int = 2,
    r_cut: float = 5.5,
    use_spin: Sequence[bool] | None = None,
    **kwargs: Any,
) -> SpinGNNV2:
    """Build a ``v2_soc`` model (Stage 0 + 1 + 2 + zero-init thin SOC Stage 3)."""
    kwargs.setdefault("predict_bec", False)
    kwargs.setdefault("predict_e2slp", False)
    cfg = SpinGNNV2Config(
        num_types=num_types,
        r_cut=r_cut,
        has_spin=True,
        has_soc=True,
        use_spin=tuple(use_spin) if use_spin is not None else None,
        **kwargs,
    )
    return SpinGNNV2(cfg)


def build_v2_born(
    num_types: int = 2,
    r_cut: float = 5.5,
    asr: str = "soft",
    has_spin: bool = False,
    has_soc: bool = False,
    use_spin: Sequence[bool] | None = None,
    **kwargs: Any,
) -> SpinGNNV2:
    """Build a ``v2_born`` (DREAM BEC) model with optional magnetoelectric coupling."""
    kwargs.setdefault("predict_e2slp", False)
    cfg = SpinGNNV2Config(
        num_types=num_types,
        r_cut=r_cut,
        has_spin=has_spin,
        has_soc=has_soc,
        use_spin=tuple(use_spin) if use_spin is not None else None,
        predict_bec=True,
        bec_asr=asr,
        **kwargs,
    )
    return SpinGNNV2(cfg)


def build_v2_e2slp(
    num_types: int = 2,
    r_cut: float = 5.5,
    has_spin: bool = True,
    has_soc: bool = True,
    use_spin: Sequence[bool] | None = None,
    include_f: bool = True,
    include_sf: bool | None = None,
    predict_gamma: bool = True,
    **kwargs: Any,
) -> SpinGNNV2:
    """Build a ``v2_e2slp`` evidential uncertainty model (+ optional GRACE gamma)."""
    if include_sf is None:
        include_sf = has_spin
    kwargs.setdefault("predict_bec", False)
    cfg = SpinGNNV2Config(
        num_types=num_types,
        r_cut=r_cut,
        has_spin=has_spin,
        has_soc=has_soc,
        use_spin=tuple(use_spin) if use_spin is not None else None,
        predict_e2slp=True,
        e2slp_include_f=include_f,
        e2slp_include_sf=include_sf,
        predict_gamma=predict_gamma,
        **kwargs,
    )
    return SpinGNNV2(cfg)
