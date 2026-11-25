"""Irreducible representation algebras for coupled O(3)xZ2^T and decoupled O(3)_lat x O(3)_spin.

Implements:
- Coupled SOC group :math:`G_{\\mathrm{SOC}} = O(3) \\times \\mathbb{Z}_2^T`:
  ``Irrep(l, p, t)`` and ``Irreps`` with canonical spec/hash preserving ``(p, t)`` (R4).
- Decoupled no-SOC group :math:`G_{\\mathrm{noSOC}} = O(3)_{\\mathrm{lat}} \\times O(3)_{\\mathrm{spin}}`:
  ``DIrrep(l, p, lam, t)`` and ``DIrreps`` with Kronecker Clebsch-Gordan bodies
  and canonical spec/hash preserving ``(p, t)`` (R4).
"""

import collections
import functools
import hashlib
import numbers
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from .wigner import _ordinary_tensors, proper_wigner_Ds, wigner_3j

__all__ = [
    "Irrep",
    "MulIr",
    "Irreps",
    "DIrrep",
    "DMulIr",
    "DIrreps",
    "decoupled_clebsch_gordan",
    "decoupled_body_key",
    "o3_body_key",
    "symmetry_spec_hash",
]

_LETTER_TO_SIGN = {"e": 1, "o": -1}
_SIGN_TO_LETTER = {1: "e", -1: "o"}


def _is_nonneg_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool) and val >= 0


def _parse_single_degree_parity(text: str) -> Tuple[int, int]:
    text = text.strip()
    if len(text) < 2:
        raise ValueError(f"Invalid degree/parity token: {text!r}")
    letter = text[-1]
    deg = int(text[:-1])
    if deg < 0:
        raise ValueError(f"Negative degree in {text!r}")
    if letter == "y":
        return deg, (-1) ** deg
    if letter not in _LETTER_TO_SIGN:
        raise ValueError(f"Unknown parity letter {letter!r} in {text!r}")
    return deg, _LETTER_TO_SIGN[letter]


def _invert_perm(perm: Sequence[int]) -> Tuple[int, ...]:
    inv = [0] * len(perm)
    for idx, pos in enumerate(perm):
        inv[pos] = idx
    return tuple(inv)


def _direct_sum(*matrices: torch.Tensor) -> torch.Tensor:
    if not matrices:
        raise ValueError("_direct_sum requires at least one matrix")
    batch = matrices[0].shape[:-2]
    m_total = sum(mat.shape[-2] for mat in matrices)
    n_total = sum(mat.shape[-1] for mat in matrices)
    out = matrices[0].new_zeros(batch + (m_total, n_total))
    row = 0
    col = 0
    for mat in matrices:
        dm, dn = mat.shape[-2], mat.shape[-1]
        out[..., row : row + dm, col : col + dn] = mat
        row += dm
        col += dn
    return out


def _parity_sign(
    sign: int,
    k: Union[int, torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Compute ``sign ** k`` safely for ``sign in {-1, +1}`` without float-power NaNs."""
    k_t = torch.as_tensor(k, device=device)
    if sign == 1:
        return torch.ones(k_t.shape, dtype=dtype, device=device)
    odd = (torch.round(k_t.to(dtype=torch.float64)).to(dtype=torch.long) & 1) == 1
    return torch.where(
        odd,
        torch.full(k_t.shape, -1.0, dtype=dtype, device=device),
        torch.ones(k_t.shape, dtype=dtype, device=device),
    )


def _canonical_encode(obj: Any) -> str:
    """Deterministic serialization preserving grading metadata (p, t) for R4 hashing."""
    if obj is None:
        return "N"
    if isinstance(obj, bool):
        return "T" if obj else "F"
    if isinstance(obj, (Irrep, Irreps, DIrrep, DIrreps)):
        return obj.canonical_spec()
    if isinstance(obj, str):
        return f"s:{obj!r}"
    if isinstance(obj, numbers.Integral):
        return f"i:{int(obj)}"
    if isinstance(obj, numbers.Real):
        return f"f:{float(obj).hex()}"
    if isinstance(obj, dict):
        items = sorted((str(k), _canonical_encode(v)) for k, v in obj.items())
        return "{" + ",".join(f"{k}={v}" for k, v in items) + "}"
    if isinstance(obj, (tuple, list)):
        return "[" + ",".join(_canonical_encode(v) for v in obj) + "]"
    return f"r:{obj!r}"


def symmetry_spec_hash(*items: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Compute a SHA-256 canonical hash preserving all ``(p, t)`` grading labels (R4)."""
    payload = _canonical_encode(items)
    if metadata is not None:
        payload = payload + "|meta:" + _canonical_encode(metadata)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ============================================================================
# Coupled SOC group: O(3) x Z2^T
# ============================================================================


class Irrep(tuple):
    r"""Irreducible representation ``(l, p, t)`` of :math:`O(3) \times \mathbb{Z}_2^T`.

    - ``l >= 0``: angular momentum degree (dimension ``2l + 1``)
    - ``p in {+1, -1}``: spatial inversion parity (``e`` = +1, ``o`` = -1)
    - ``t in {+1, -1}``: time-reversal parity (``e`` = +1, ``o`` = -1)
    """

    def __new__(
        cls,
        l: Union[int, "Irrep", str, Sequence[int]],
        p: Optional[int] = None,
        t: Optional[int] = None,
    ):
        if p is None and t is None:
            if isinstance(l, Irrep):
                return l
            if isinstance(l, str):
                name = l.strip()
                try:
                    if len(name) >= 3 and name[-2] in ("e", "o", "y"):
                        deg = int(name[:-2])
                        p_char, t_char = name[-2], name[-1]
                        p = (-1) ** deg if p_char == "y" else _LETTER_TO_SIGN[p_char]
                        t = (-1) ** deg if t_char == "y" else _LETTER_TO_SIGN[t_char]
                        l = deg
                    else:
                        l, p = _parse_single_degree_parity(name)
                        t = 1
                except Exception as exc:
                    raise ValueError(
                        f'Unable to parse string "{l}" as Irrep (expected e.g. "1oe", "1eo", "2ee")'
                    ) from exc
            elif isinstance(l, (tuple, list)):
                if len(l) == 2:
                    l, p = l
                    t = 1
                elif len(l) == 3:
                    l, p, t = l
                else:
                    raise ValueError(f"Irrep tuple must have length 2 or 3, got {l!r}")
            else:
                raise ValueError(f"Unable to interpret {l!r} as Irrep")
        elif t is None:
            t = 1

        if not _is_nonneg_int(l):
            raise ValueError(f"l must be a non-negative integer, got {l!r}")
        if p not in (-1, 1):
            raise ValueError(f"Spatial parity p must be -1 or +1, got {p!r}")
        if t not in (-1, 1):
            raise ValueError(f"Time-reversal parity t must be -1 or +1, got {t!r}")
        return super().__new__(cls, (int(l), int(p), int(t)))

    @property
    def l(self) -> int:  # noqa: E743
        return self[0]

    @property
    def p(self) -> int:
        return self[1]

    @property
    def t(self) -> int:
        return self[2]

    @property
    def dim(self) -> int:
        return 2 * self.l + 1

    @property
    def grading(self) -> Tuple[int, int]:
        return (self.p, self.t)

    def is_scalar(self) -> bool:
        """True only for the full group invariant ``0ee`` (l=0, p=+1, t=+1)."""
        return self.l == 0 and self.p == 1 and self.t == 1

    def canonical_spec(self) -> str:
        return f"O3xZ2T:Irrep({self.l},{self.p},{self.t})"

    def canonical_hash(self) -> str:
        return hashlib.sha256(self.canonical_spec().encode("utf-8")).hexdigest()

    def _sort_key(self) -> Tuple[int, int, int]:
        return (self.l, int(self.p != (-1) ** self.l), int(self.t != 1))

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Irrep):
            return NotImplemented
        return self._sort_key() < other._sort_key()

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, Irrep):
            return NotImplemented
        return self._sort_key() <= other._sort_key()

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, Irrep):
            return NotImplemented
        return self._sort_key() > other._sort_key()

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, Irrep):
            return NotImplemented
        return self._sort_key() >= other._sort_key()

    def __repr__(self) -> str:
        return f"{self.l}{_SIGN_TO_LETTER[self.p]}{_SIGN_TO_LETTER[self.t]}"

    def __mul__(self, other: Any) -> Iterable["Irrep"]:
        other = Irrep(other)
        p_out = self.p * other.p
        t_out = self.t * other.t
        for l_out in range(abs(self.l - other.l), self.l + other.l + 1):
            yield Irrep(l_out, p_out, t_out)

    def __rmul__(self, other: int) -> "Irreps":
        if not _is_nonneg_int(other):
            return NotImplemented
        return Irreps([(other, self)])

    def __add__(self, other: Any) -> "Irreps":
        return Irreps(self) + Irreps(other)

    def D_from_matrix(
        self,
        R: torch.Tensor,
        k_inv: Optional[Union[int, torch.Tensor]] = None,
        k_t: Union[int, torch.Tensor] = 0,
    ) -> torch.Tensor:
        r"""Exact representation matrix :math:`p^{k_{\mathrm{inv}}} t^{k_t} D^l(R_{\mathrm{proper}})`.

        If ``k_inv is None`` and ``R`` is improper (:math:`\det R = -1`), spatial
        inversion is extracted automatically via :math:`R = -R_{\mathrm{proper}}`.
        """
        R = torch.as_tensor(R)
        R_det = R.float() if R.dtype in (torch.float16, torch.bfloat16) else R
        det = torch.det(R_det).to(dtype=R.dtype)
        if k_inv is None:
            sign_det = torch.where(
                det >= 0, torch.ones_like(det), -torch.ones_like(det)
            )
            R_proper = R * sign_det[..., None, None]
            k_inv_t = (1.0 - sign_det) * 0.5
        else:
            R_proper = R
            k_inv_t = torch.as_tensor(k_inv, dtype=R.dtype, device=R.device)
        k_t_t = torch.as_tensor(k_t, dtype=R.dtype, device=R.device)
        D = proper_wigner_Ds(self.l, R_proper)[self.l]
        factor = _parity_sign(self.p, k_inv_t, D.dtype, D.device) * _parity_sign(
            self.t, k_t_t, D.dtype, D.device
        )
        return D * factor[..., None, None]


class MulIr(tuple):
    """Multiplicity-Irrep pair ``(mul, Irrep)``."""

    def __new__(cls, mul: Any, ir: Optional[Irrep] = None):
        if ir is None:
            mul, ir = mul
        ir = Irrep(ir)
        if not _is_nonneg_int(mul):
            raise ValueError(f"Multiplicity must be non-negative int, got {mul!r}")
        return super().__new__(cls, (int(mul), ir))

    @property
    def mul(self) -> int:
        return self[0]

    @property
    def ir(self) -> Irrep:
        return self[1]

    @property
    def dim(self) -> int:
        return self.mul * self.ir.dim

    def __repr__(self) -> str:
        return f"{self.mul}x{self.ir}"


class Irreps(tuple):
    """Direct sum of coupled ``Irrep`` blocks for :math:`O(3) \\times \\mathbb{Z}_2^T`."""

    def __new__(cls, irreps: Any = None):
        if isinstance(irreps, Irreps):
            return super().__new__(cls, irreps)
        out: List[MulIr] = []
        if isinstance(irreps, Irrep):
            out.append(MulIr(1, irreps))
        elif isinstance(irreps, str):
            text = irreps.strip()
            if text != "":
                try:
                    for token in text.split("+"):
                        token = token.strip()
                        if "x" in token:
                            mul_s, ir_s = token.split("x", 1)
                            out.append(MulIr(int(mul_s.strip()), Irrep(ir_s)))
                        else:
                            out.append(MulIr(1, Irrep(token)))
                except Exception as exc:
                    raise ValueError(
                        f'Unable to convert string "{irreps}" into Irreps'
                    ) from exc
        elif irreps is not None:
            for item in irreps:
                if isinstance(item, MulIr):
                    out.append(item)
                elif isinstance(item, Irrep):
                    out.append(MulIr(1, item))
                elif isinstance(item, str):
                    out.append(MulIr(1, Irrep(item)))
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    out.append(MulIr(item[0], Irrep(item[1])))
                else:
                    raise ValueError(
                        f"Unable to interpret {item!r} as a (mul, Irrep) pair"
                    )
        return super().__new__(cls, out)

    @staticmethod
    def spherical_harmonics(lmax: int, p: int = -1, t: int = 1) -> "Irreps":
        if not _is_nonneg_int(lmax):
            raise ValueError(f"lmax must be non-negative int, got {lmax!r}")
        return Irreps([(1, Irrep(l, p**l, t**l)) for l in range(lmax + 1)])

    def slices(self) -> List[slice]:
        out = []
        offset = 0
        for mul_ir in self:
            out.append(slice(offset, offset + mul_ir.dim))
            offset += mul_ir.dim
        return out

    def randn(
        self,
        *size: int,
        normalization: str = "component",
        requires_grad: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        di = size.index(-1)
        lsize = size[:di]
        rsize = size[di + 1 :]
        if normalization == "component":
            return torch.randn(
                *lsize,
                self.dim,
                *rsize,
                requires_grad=requires_grad,
                dtype=dtype,
                device=device,
            )
        if normalization == "norm":
            x = torch.zeros(*lsize, self.dim, *rsize, dtype=dtype, device=device)
            for sl, (mul, ir) in zip(self.slices(), self):
                if mul == 0:
                    continue
                r = torch.randn(*lsize, mul, ir.dim, *rsize, dtype=dtype, device=device)
                r = r / r.norm(2, dim=di + 1, keepdim=True)
                x.narrow(di, sl.start, mul * ir.dim).copy_(
                    r.reshape(*lsize, mul * ir.dim, *rsize)
                )
            return x.requires_grad_(requires_grad)
        raise ValueError("normalization must be 'component' or 'norm'")

    def __getitem__(self, item: Any):
        res = super().__getitem__(item)
        if isinstance(item, slice):
            return Irreps(res)
        return res

    def __contains__(self, ir: Any) -> bool:
        target = Irrep(ir)
        return any(item_ir == target for _, item_ir in self)

    def count(self, ir: Any) -> int:
        target = Irrep(ir)
        return sum(mul for mul, item_ir in self if item_ir == target)

    def __add__(self, other: Any) -> "Irreps":
        return Irreps(super().__add__(Irreps(other)))

    def __mul__(self, other: Any) -> "Irreps":
        if isinstance(other, Irreps):
            raise NotImplementedError("Use StaticTensorProduct for tensor products")
        return Irreps(super().__mul__(other))

    def __rmul__(self, other: int) -> "Irreps":
        return Irreps(super().__rmul__(other))

    def simplify(self) -> "Irreps":
        """Merge consecutive identical ``(l, p, t)`` blocks (R2: never merges distinct p or t)."""
        out: List[Tuple[int, Irrep]] = []
        for mul, ir in self:
            if mul == 0:
                continue
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            else:
                out.append((mul, ir))
        return Irreps(out)

    def remove_zero_multiplicities(self) -> "Irreps":
        return Irreps([(mul, ir) for mul, ir in self if mul > 0])

    def sort(self):
        Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
        ordered = sorted((ir, i, mul) for i, (mul, ir) in enumerate(self))
        inv = tuple(i for _, i, _ in ordered)
        p = _invert_perm(inv)
        return Ret(Irreps([(mul, ir) for ir, _, mul in ordered]), p, inv)

    @property
    def dim(self) -> int:
        return sum(mul_ir.dim for mul_ir in self)

    @property
    def num_irreps(self) -> int:
        return sum(mul for mul, _ in self)

    @property
    def ls(self) -> List[int]:
        return [ir.l for mul, ir in self for _ in range(mul)]

    @property
    def lmax(self) -> int:
        if not self.ls:
            raise ValueError("Cannot get lmax of empty Irreps")
        return max(self.ls)

    def canonical_spec(self) -> str:
        body = ",".join(f"({mul},{ir.l},{ir.p},{ir.t})" for mul, ir in self)
        return f"O3xZ2T:Irreps[{body}]"

    def canonical_hash(self) -> str:
        return hashlib.sha256(self.canonical_spec().encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        return "+".join(repr(mul_ir) for mul_ir in self)

    def D_from_matrix(
        self,
        R: torch.Tensor,
        k_inv: Optional[Union[int, torch.Tensor]] = None,
        k_t: Union[int, torch.Tensor] = 0,
    ) -> torch.Tensor:
        R = torch.as_tensor(R)
        R_det = R.float() if R.dtype in (torch.float16, torch.bfloat16) else R
        det = torch.det(R_det).to(dtype=R.dtype)
        if k_inv is None:
            sign_det = torch.where(
                det >= 0, torch.ones_like(det), -torch.ones_like(det)
            )
            R_proper = R * sign_det[..., None, None]
            k_inv_t = (1.0 - sign_det) * 0.5
        else:
            R_proper = R
            k_inv_t = torch.as_tensor(k_inv, dtype=R.dtype, device=R.device)
        k_t_t = torch.as_tensor(k_t, dtype=R.dtype, device=R.device)
        lmax = max([ir.l for mul, ir in self if mul > 0], default=0)
        Ds = proper_wigner_Ds(lmax, R_proper)
        cache: Dict[Irrep, torch.Tensor] = {}
        blocks: List[torch.Tensor] = []
        for mul, ir in self:
            if mul == 0:
                continue
            if ir not in cache:
                factor = _parity_sign(ir.p, k_inv_t, R.dtype, R.device) * _parity_sign(
                    ir.t, k_t_t, R.dtype, R.device
                )
                cache[ir] = Ds[ir.l] * factor[..., None, None]
            blocks.extend([cache[ir]] * mul)
        if not blocks:
            batch = torch.broadcast_shapes(R.shape[:-2], k_inv_t.shape, k_t_t.shape)
            return R.new_zeros(batch + (0, 0))
        return _direct_sum(*blocks)


# ============================================================================
# Decoupled no-SOC group: O(3)_lat x O(3)_spin
# ============================================================================


class DIrrep(tuple):
    r"""Irreducible representation ``(l, p | lam, t)`` of :math:`O(3)_{\mathrm{lat}} \times O(3)_{\mathrm{spin}}`.

    - ``(l, p)``: lattice angular degree and spatial inversion parity
    - ``(lam, t)``: spin angular degree and spin inversion (= time reversal) parity
    - Dimension: ``(2l + 1) * (2lam + 1)``, stored lattice-index major ``[m_lat, m_spin]``.
    """

    def __new__(
        cls,
        l: Any,
        p: Optional[int] = None,
        lam: Optional[int] = None,
        t: Optional[int] = None,
    ):
        if p is None and lam is None and t is None:
            if isinstance(l, DIrrep):
                return l
            if isinstance(l, str):
                name = l.strip()
                try:
                    lat_part, spin_part = name.split("|")
                    l, p = _parse_single_degree_parity(lat_part)
                    lam, t = _parse_single_degree_parity(spin_part)
                except Exception as exc:
                    raise ValueError(
                        f'Unable to parse string "{l}" as DIrrep (expected e.g. "1o|1o")'
                    ) from exc
            elif isinstance(l, (tuple, list)) and len(l) == 4:
                l, p, lam, t = l
            else:
                raise ValueError(
                    f"Unable to interpret {l!r} as DIrrep; pass (l, p, lam, t) or '1o|1o'"
                )
        elif p is None or lam is None or t is None:
            raise ValueError("DIrrep(l, p, lam, t) requires all four arguments")

        if not _is_nonneg_int(l):
            raise ValueError(f"l must be a non-negative integer, got {l!r}")
        if not _is_nonneg_int(lam):
            raise ValueError(f"lam must be a non-negative integer, got {lam!r}")
        if p not in (-1, 1):
            raise ValueError(f"Lattice parity p must be -1 or +1, got {p!r}")
        if t not in (-1, 1):
            raise ValueError(f"Spin/time-reversal parity t must be -1 or +1, got {t!r}")
        return super().__new__(cls, (int(l), int(p), int(lam), int(t)))

    @property
    def l(self) -> int:  # noqa: E743
        return self[0]

    @property
    def p(self) -> int:
        return self[1]

    @property
    def lam(self) -> int:
        return self[2]

    @property
    def t(self) -> int:
        return self[3]

    @property
    def dim(self) -> int:
        return (2 * self.l + 1) * (2 * self.lam + 1)

    @property
    def grading(self) -> Tuple[int, int]:
        return (self.p, self.t)

    def is_scalar(self) -> bool:
        """True only for the full group invariant ``0e|0e``."""
        return self.l == 0 and self.lam == 0 and self.p == 1 and self.t == 1

    def canonical_spec(self) -> str:
        return f"O3_latxO3_spin:DIrrep({self.l},{self.p},{self.lam},{self.t})"

    def canonical_hash(self) -> str:
        return hashlib.sha256(self.canonical_spec().encode("utf-8")).hexdigest()

    def couple_irreps(self, jmax: Optional[int] = None) -> List[Irrep]:
        """Decompose ``(l, p | lam, t)`` under the SOC diagonal subgroup ``O(3) x Z2^T``."""
        if jmax is not None and not _is_nonneg_int(jmax):
            raise ValueError(f"jmax must be None or non-negative int, got {jmax!r}")
        return [
            Irrep(J, self.p, self.t)
            for J in range(abs(self.l - self.lam), self.l + self.lam + 1)
            if jmax is None or J <= jmax
        ]

    def _sort_key(self) -> Tuple[int, int, int, int]:
        return (
            self.l,
            self.lam,
            int(self.p != (-1) ** self.l),
            int(self.t != (-1) ** self.lam),
        )

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, DIrrep):
            return NotImplemented
        return self._sort_key() < other._sort_key()

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, DIrrep):
            return NotImplemented
        return self._sort_key() <= other._sort_key()

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, DIrrep):
            return NotImplemented
        return self._sort_key() > other._sort_key()

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, DIrrep):
            return NotImplemented
        return self._sort_key() >= other._sort_key()

    def __repr__(self) -> str:
        return f"{self.l}{_SIGN_TO_LETTER[self.p]}|{self.lam}{_SIGN_TO_LETTER[self.t]}"

    def __mul__(self, other: Any) -> Iterable["DIrrep"]:
        other = DIrrep(other)
        p_out = self.p * other.p
        t_out = self.t * other.t
        for l_out in range(abs(self.l - other.l), self.l + other.l + 1):
            for lam_out in range(abs(self.lam - other.lam), self.lam + other.lam + 1):
                yield DIrrep(l_out, p_out, lam_out, t_out)

    def __rmul__(self, other: int) -> "DIrreps":
        if not _is_nonneg_int(other):
            return NotImplemented
        return DIrreps([(other, self)])

    def __add__(self, other: Any) -> "DIrreps":
        return DIrreps(self) + DIrreps(other)

    def D_from_matrices(
        self,
        R_lat: torch.Tensor,
        R_spin: torch.Tensor,
        k_inv: Union[int, torch.Tensor] = 0,
        k_t: Union[int, torch.Tensor] = 0,
    ) -> torch.Tensor:
        r"""Exact representation matrix :math:`p^{k_{\mathrm{inv}}} t^{k_t} D^l(R_{\mathrm{lat}}) \otimes D^\lambda(R_{\mathrm{spin}})`."""
        D_lat = proper_wigner_Ds(self.l, R_lat)[self.l]
        D_spin = proper_wigner_Ds(self.lam, R_spin)[self.lam]
        return _assemble_decoupled_D(self, D_lat, D_spin, k_inv, k_t)


def _assemble_decoupled_D(
    ir: DIrrep,
    D_lat: torch.Tensor,
    D_spin: torch.Tensor,
    k_inv: Union[int, torch.Tensor],
    k_t: Union[int, torch.Tensor],
) -> torch.Tensor:
    D = torch.einsum("...ij,...kl->...ikjl", D_lat, D_spin)
    D = D.reshape(D.shape[:-4] + (ir.dim, ir.dim))
    sign = _parity_sign(ir.p, k_inv, D.dtype, D.device) * _parity_sign(
        ir.t, k_t, D.dtype, D.device
    )
    return D * sign[..., None, None]


class DMulIr(tuple):
    """Multiplicity-DIrrep pair ``(mul, DIrrep)``."""

    def __new__(cls, mul: Any, ir: Optional[DIrrep] = None):
        if ir is None:
            mul, ir = mul
        ir = DIrrep(ir)
        if not _is_nonneg_int(mul):
            raise ValueError(f"Multiplicity must be non-negative int, got {mul!r}")
        return super().__new__(cls, (int(mul), ir))

    @property
    def mul(self) -> int:
        return self[0]

    @property
    def ir(self) -> DIrrep:
        return self[1]

    @property
    def dim(self) -> int:
        return self.mul * self.ir.dim

    def __repr__(self) -> str:
        return f"{self.mul}x{self.ir}"


class DIrreps(tuple):
    """Direct sum of decoupled ``DIrrep`` blocks for :math:`O(3)_{\\mathrm{lat}} \\times O(3)_{\\mathrm{spin}}`."""

    def __new__(cls, irreps: Any = None):
        if isinstance(irreps, DIrreps):
            return super().__new__(cls, irreps)
        out: List[DMulIr] = []
        if isinstance(irreps, DIrrep):
            out.append(DMulIr(1, irreps))
        elif isinstance(irreps, str):
            text = irreps.strip()
            if text != "":
                try:
                    for token in text.split("+"):
                        token = token.strip()
                        if "x" in token:
                            mul_s, ir_s = token.split("x", 1)
                            out.append(DMulIr(int(mul_s.strip()), DIrrep(ir_s)))
                        else:
                            out.append(DMulIr(1, DIrrep(token)))
                except Exception as exc:
                    raise ValueError(
                        f'Unable to convert string "{irreps}" into DIrreps'
                    ) from exc
        elif irreps is not None:
            for item in irreps:
                if isinstance(item, DMulIr):
                    out.append(item)
                elif isinstance(item, DIrrep):
                    out.append(DMulIr(1, item))
                elif isinstance(item, str):
                    out.append(DMulIr(1, DIrrep(item)))
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    out.append(DMulIr(item[0], DIrrep(item[1])))
                else:
                    raise ValueError(
                        f"Unable to interpret {item!r} as a (mul, DIrrep) pair"
                    )
        return super().__new__(cls, out)

    @staticmethod
    def spherical_harmonics_products(
        lmax: int, lammax: int, p: int = -1, t: int = -1
    ) -> "DIrreps":
        if not (_is_nonneg_int(lmax) and _is_nonneg_int(lammax)):
            raise ValueError("lmax and lammax must be non-negative integers")
        return DIrreps(
            [
                (1, DIrrep(l, p**l, lam, t**lam))
                for l in range(lmax + 1)
                for lam in range(lammax + 1)
            ]
        )

    def slices(self) -> List[slice]:
        out = []
        offset = 0
        for mul_ir in self:
            out.append(slice(offset, offset + mul_ir.dim))
            offset += mul_ir.dim
        return out

    def randn(
        self,
        *size: int,
        normalization: str = "component",
        requires_grad: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        di = size.index(-1)
        lsize = size[:di]
        rsize = size[di + 1 :]
        if normalization == "component":
            return torch.randn(
                *lsize,
                self.dim,
                *rsize,
                requires_grad=requires_grad,
                dtype=dtype,
                device=device,
            )
        if normalization == "norm":
            x = torch.zeros(*lsize, self.dim, *rsize, dtype=dtype, device=device)
            for sl, (mul, ir) in zip(self.slices(), self):
                if mul == 0:
                    continue
                r = torch.randn(*lsize, mul, ir.dim, *rsize, dtype=dtype, device=device)
                r = r / r.norm(2, dim=di + 1, keepdim=True)
                x.narrow(di, sl.start, mul * ir.dim).copy_(
                    r.reshape(*lsize, mul * ir.dim, *rsize)
                )
            return x.requires_grad_(requires_grad)
        raise ValueError("normalization must be 'component' or 'norm'")

    def __getitem__(self, item: Any):
        res = super().__getitem__(item)
        if isinstance(item, slice):
            return DIrreps(res)
        return res

    def __contains__(self, ir: Any) -> bool:
        target = DIrrep(ir)
        return any(item_ir == target for _, item_ir in self)

    def count(self, ir: Any) -> int:
        target = DIrrep(ir)
        return sum(mul for mul, item_ir in self if item_ir == target)

    def __add__(self, other: Any) -> "DIrreps":
        return DIrreps(super().__add__(DIrreps(other)))

    def __mul__(self, other: Any) -> "DIrreps":
        if isinstance(other, DIrreps):
            raise NotImplementedError("Use StaticTensorProduct for tensor products")
        return DIrreps(super().__mul__(other))

    def __rmul__(self, other: int) -> "DIrreps":
        return DIrreps(super().__rmul__(other))

    def simplify(self) -> "DIrreps":
        """Merge consecutive identical ``(l, p, lam, t)`` blocks (R2)."""
        out: List[Tuple[int, DIrrep]] = []
        for mul, ir in self:
            if mul == 0:
                continue
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            else:
                out.append((mul, ir))
        return DIrreps(out)

    def remove_zero_multiplicities(self) -> "DIrreps":
        return DIrreps([(mul, ir) for mul, ir in self if mul > 0])

    def sort(self):
        Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
        ordered = sorted((ir, i, mul) for i, (mul, ir) in enumerate(self))
        inv = tuple(i for _, i, _ in ordered)
        p = _invert_perm(inv)
        return Ret(DIrreps([(mul, ir) for ir, _, mul in ordered]), p, inv)

    @property
    def dim(self) -> int:
        return sum(mul_ir.dim for mul_ir in self)

    @property
    def num_irreps(self) -> int:
        return sum(mul for mul, _ in self)

    @property
    def ls(self) -> List[int]:
        return [ir.l for mul, ir in self for _ in range(mul)]

    @property
    def lams(self) -> List[int]:
        return [ir.lam for mul, ir in self for _ in range(mul)]

    @property
    def lmax(self) -> int:
        if not self.ls:
            raise ValueError("Cannot get lmax of empty DIrreps")
        return max(self.ls)

    @property
    def lammax(self) -> int:
        if not self.lams:
            raise ValueError("Cannot get lammax of empty DIrreps")
        return max(self.lams)

    def canonical_spec(self) -> str:
        body = ",".join(f"({mul},{ir.l},{ir.p},{ir.lam},{ir.t})" for mul, ir in self)
        return f"O3_latxO3_spin:DIrreps[{body}]"

    def canonical_hash(self) -> str:
        return hashlib.sha256(self.canonical_spec().encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        return "+".join(repr(mul_ir) for mul_ir in self)

    def D_from_matrices(
        self,
        R_lat: torch.Tensor,
        R_spin: torch.Tensor,
        k_inv: Union[int, torch.Tensor] = 0,
        k_t: Union[int, torch.Tensor] = 0,
    ) -> torch.Tensor:
        lmax = max([ir.l for mul, ir in self if mul > 0], default=0)
        lammax = max([ir.lam for mul, ir in self if mul > 0], default=0)
        D_lat = proper_wigner_Ds(lmax, R_lat)
        D_spin = proper_wigner_Ds(lammax, R_spin)
        cache: Dict[DIrrep, torch.Tensor] = {}
        blocks: List[torch.Tensor] = []
        for mul, ir in self:
            if mul == 0:
                continue
            if ir not in cache:
                cache[ir] = _assemble_decoupled_D(
                    ir, D_lat[ir.l], D_spin[ir.lam], k_inv, k_t
                )
            blocks.extend([cache[ir]] * mul)
        if not blocks:
            D0 = _assemble_decoupled_D(
                DIrrep(0, 1, 0, 1), D_lat[0], D_spin[0], k_inv, k_t
            )
            return D0.new_zeros(D0.shape[:-2] + (0, 0))
        return _direct_sum(*blocks)


@functools.lru_cache(maxsize=1024)
def _decoupled_cg_f64(
    l1: int, l2: int, l3: int, lam1: int, lam2: int, lam3: int
) -> torch.Tensor:
    with _ordinary_tensors():
        c_lat = wigner_3j(l1, l2, l3, dtype=torch.float64, device="cpu")
        c_spin = wigner_3j(lam1, lam2, lam3, dtype=torch.float64, device="cpu")
        body = torch.einsum("ikm,jln->ijklmn", c_lat, c_spin)
        return body.reshape(
            (2 * l1 + 1) * (2 * lam1 + 1),
            (2 * l2 + 1) * (2 * lam2 + 1),
            (2 * l3 + 1) * (2 * lam3 + 1),
        ).contiguous()


def decoupled_clebsch_gordan(
    ir1: Any,
    ir2: Any,
    ir3: Any,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Kronecker Clebsch-Gordan tensor for ``O(3)_lat x O(3)_spin`` with Frobenius norm 1.

    Validates both triangle rules and ``(p, t)`` grading selection rules (R1).
    """
    d1, d2, d3 = DIrrep(ir1), DIrrep(ir2), DIrrep(ir3)
    if d1.p * d2.p != d3.p:
        raise ValueError(f"{d1} x {d2} -> {d3} violates lattice parity p3 = p1 * p2")
    if d1.t * d2.t != d3.t:
        raise ValueError(
            f"{d1} x {d2} -> {d3} violates time-reversal parity t3 = t1 * t2"
        )
    if not (abs(d1.l - d2.l) <= d3.l <= d1.l + d2.l):
        raise ValueError(f"{d1} x {d2} -> {d3} violates lattice degree triangle rule")
    if not (abs(d1.lam - d2.lam) <= d3.lam <= d1.lam + d2.lam):
        raise ValueError(f"{d1} x {d2} -> {d3} violates spin degree triangle rule")
    body = _decoupled_cg_f64(d1.l, d2.l, d3.l, d1.lam, d2.lam, d3.lam)
    return body.to(dtype=dtype, device=device, copy=True)


def decoupled_body_key(ir1: Any, ir2: Any, ir3: Any) -> Tuple:
    d1, d2, d3 = DIrrep(ir1), DIrrep(ir2), DIrrep(ir3)
    return ("o3xo3", (d1.l, d2.l, d3.l), (d1.lam, d2.lam, d3.lam))


def o3_body_key(ir1: Any, ir2: Any, ir3: Any) -> Tuple:
    r1, r2, r3 = Irrep(ir1), Irrep(ir2), Irrep(ir3)
    return ("o3", r1.l, r2.l, r3.l)
