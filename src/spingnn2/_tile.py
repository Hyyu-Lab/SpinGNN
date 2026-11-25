"""Forward and analytic vector-Jacobian products for one complete set of centers."""

from dataclasses import dataclass

import numpy as np

from .backbone import backbone_forward, mlp_forward
from .harmonics import cartesian_norm, evaluate_harmonics
from .splines import evaluate_basis, polynomial_cutoff, spin_embedding


def scatter(values, indices, size):
    out = np.zeros((size, *values.shape[1:]), dtype=values.dtype)
    np.add.at(out, indices, values)
    return out


def project(adjoint, jacobian):
    return np.einsum("ek,ekd->ed", adjoint, jacobian)


@dataclass
class Tile:
    start: int
    stop: int
    center: np.ndarray
    neighbor: np.ndarray
    species: np.ndarray
    types_i: np.ndarray
    types_j: np.ndarray
    mode: str

    @property
    def size(self):
        return self.stop - self.start


class Sweep:
    """Tape of explicit layer adjoints; each assignment versions its output fields."""

    def __init__(self, tile, fields, tape=()):
        self.tile = tile
        self.f = dict(fields)
        self.tape = list(tape)

    def record(self, inputs, outputs, values, pullback):
        self.f.update(zip(outputs, values))
        self.tape.append((inputs, outputs, pullback))

    def reverse(self, atomic_seed):
        grads = {"atomic_energy": atomic_seed}
        for inputs, outputs, pullback in reversed(self.tape):
            dy = [grads.pop(key, None) for key in outputs]
            if all(g is None for g in dy):
                continue
            for key, delta in zip(inputs, pullback(*dy)):
                if delta is not None:
                    grads[key] = grads[key] + delta if key in grads else delta
        return grads

    def one_hot(self, layer, spin):
        t = self.tile
        eye = np.eye(layer["num_types"], dtype=self.f["r"].dtype)
        for suffix, types in (("i", t.types_i), ("j", t.types_j)):
            key = "node_" + suffix
            values = eye[types]
            if not spin:
                self.f[key] = values
                continue
            s = self.f[suffix]
            norm = cartesian_norm(s, keepdims=True)
            unit = s / np.maximum(norm, 1e-12)
            values = np.concatenate(
                (values, (unit * unit).sum(-1, keepdims=True)), axis=1
            )
            # The source uses squared norm after F.normalize. Its derivative
            # vanishes on the unit sphere and is linear below the epsilon floor.
            ds = np.where(norm < 1e-12, 2 * s / 1e-24, 0)
            self.record(
                (suffix,), (key,), (values,), lambda dy, ds=ds: (dy[:, -1:] * ds,)
            )

    def run(self, layer):
        op, t, f = layer["op"], self.tile, self.f
        field, out = layer.get("field"), layer.get("out_field")
        if op in ("OneHotAtomEncoding", "OneHotAtomEncoding_spin"):
            self.one_hot(layer, spin=op.endswith("_spin"))
        elif op.startswith("SphericalHarmonicEdgeAttrs"):
            value, jac = evaluate_harmonics(
                layer["harmonics"], f["r"], f.get("j"), f.get("i")
            )
            self.record(
                tuple(jac),
                (out,),
                (value,),
                lambda dy: tuple(project(dy, j) for j in jac.values()),
            )
        elif op == "RadialBasisEdgeEncoding":
            val, deriv = evaluate_basis(layer["basis"], f["length"], t.mode)
            self.record(("length",), (out,), (val,), lambda dy: ((dy * deriv).sum(-1),))
        elif op in ("EdgeScalarAddSpin", "RadialBasisEdgeEncoding_spin"):
            basis = layer["basis"]
            val, dr, dj, di = spin_embedding(basis, f["length"], f["j"], f["i"], t.mode)
            if field is None:
                field = "_radial_part"
                attr, ds = evaluate_basis(basis["radial"], f["length"], t.mode)
                self.record(
                    ("length",), (field,), (attr,), lambda dy: ((dy * ds).sum(-1),)
                )
            width = f[field].shape[-1]

            def spin_vjp(dy):
                tail = dy[:, width:]
                return (
                    dy[:, :width],
                    (tail * dr).sum(-1),
                    project(tail, dj),
                    project(tail, di),
                )

            self.record(
                (field, "length", "j", "i"),
                (out,),
                (np.concatenate((f[field], val), axis=1),),
                spin_vjp,
            )
        elif op in ("Allegro_og_Module", "Allegro_born_Module", "Allegro_spin_Module"):
            inputs = (
                field,
                layer["edge_invariant_field"],
                "node_i",
                "node_j",
                "length",
            )
            values, reverse = backbone_forward(
                layer["backbone"], *(f[k] for k in inputs), t.center, t.size, t.types_i
            )
            keys = tuple(values)
            self.record(
                inputs,
                keys,
                tuple(values.values()),
                lambda *dy: reverse({k: g for k, g in zip(keys, dy) if g is not None}),
            )
        elif op == "ScalarMLP":
            self.scalar_mlp(layer)
        elif op in (
            "Spin_JMatrix_Energy",
            "Spin_ani_Energy",
            "HB_Energy",
            "Biquad_Energy",
        ):
            self.spin_energy(layer)
        elif op == "EdgewiseEnergySum":
            self.edge_energy_sum(layer)
        elif op == "EdgewiseReduce":
            self.reduce_edges(field, out, layer["reduce"], layer["factor"])
        elif op == "PerSpeciesScaleShift":
            scale = (
                1
                if layer["scales"] is None
                else layer["scales"][t.species].reshape(-1, 1)
            )
            shift = (
                0
                if layer["shifts"] is None
                else layer["shifts"][t.species].reshape(-1, 1)
            )
            key = "atomic_energy"
            self.record(
                (key,), (key,), (f[key] * scale + shift,), lambda dy: (dy * scale,)
            )
        elif op == "CartesianBECReduce":
            # Global acoustic sum rule is applied after all centers are assembled.
            edge = f[field].reshape((len(t.center), 9))
            bec = scatter(edge, t.center, t.size)
            if layer["reduce"] == "mean":
                bec /= np.maximum(1, np.bincount(t.center, minlength=t.size))[:, None]
            if layer["factor"] is not None:
                bec *= layer["factor"]
            f[out] = (
                bec * layer["scales"][t.species, None]
                + layer["shifts"][t.species, None]
            )
        else:
            raise ValueError(f"Unsupported tile operation {op}")

    def scalar_mlp(self, layer):
        field, out = layer["field"], layer["out_field"]
        val, reverse = mlp_forward(layer["mlp"], self.f[field])
        coeff = layer["coeff"]
        val = val * coeff
        normalize = layer["normalize"] and not layer["add_collect"]
        if normalize:
            norm = np.linalg.norm(val, axis=-1, keepdims=True)
            denominator = np.maximum(norm, 1e-12)
            val = val / denominator
        width = val.shape[1]
        inputs = [field]
        value = val
        if layer["add_collect"]:
            inputs.append(out)
            value = val + self.f[out]
        elif layer["cat_collect"]:
            inputs.append(out)
            value = np.concatenate((self.f[out], val), axis=-1)

        def pullback(dy):
            old = []
            if layer["add_collect"]:
                old = [dy]
            elif layer["cat_collect"]:
                old = [dy[:, :-width]]
                dy = dy[:, -width:]
            if normalize:
                dy = (
                    dy
                    - np.where(
                        norm >= 1e-12, val * (val * dy).sum(-1, keepdims=True), 0
                    )
                ) / denominator
            return [reverse(dy * coeff), *old]

        self.record(inputs, (out,), (value,), pullback)

    def spin_energy(self, layer):
        field, out, op = layer["field"], layer["out_field"], layer["op"]
        a = self.f["i"]
        end = "i" if op == "Spin_ani_Energy" else "j"
        b = self.f[end]
        coupling = self.f[field] * layer["coeff"]
        if op in ("HB_Energy", "Biquad_Energy"):
            dot = (a * b).sum(-1, keepdims=True)
            basis = dot if op == "HB_Energy" else dot * dot
            slope = coupling if op == "HB_Energy" else 2 * coupling * dot
            energy = coupling * basis

            def pullback(dy):
                return dy * basis, dy * slope * b, dy * slope * a

        else:
            matrix = coupling.reshape(len(a), 3, 3)
            mb = np.einsum("eij,ej->ei", matrix, b)
            ma = np.einsum("ei,eij->ej", a, matrix)
            energy = (a * mb).sum(-1, keepdims=True)

            def pullback(dy):
                dm = dy[:, :, None] * a[:, :, None] * b[:, None, :]
                return dm.reshape(coupling.shape), dy * mb, dy * ma

        # Both source endpoints may name the same spin field for anisotropy.
        inputs = [field, "i", end]
        if layer["add_collect"]:
            inputs.append(out)
            energy = self.f[out] + energy

        def reverse(dy):
            dc, da, db = pullback(dy)
            return (
                (dc * layer["coeff"], da, db, dy)
                if layer["add_collect"]
                else (dc * layer["coeff"], da, db)
            )

        self.record(inputs, (out,), (energy,), reverse)

    def reduce_edges(self, field, out, reduction, factor):
        t = self.tile
        scale = 1.0 if factor is None else factor
        if reduction == "mean":
            counts = np.bincount(t.center, minlength=t.size).astype(self.f[field].dtype)
            scale = scale / np.maximum(counts, 1)[:, None]
        value = scatter(self.f[field], t.center, t.size) * scale
        self.record((field,), (out,), (value,), lambda dy: ((dy * scale)[t.center],))

    def edge_energy_sum(self, layer):
        t = self.tile
        key = "edge_energy"
        scale = (
            1
            if layer["scales"] is None
            else layer["scales"][t.types_i, t.types_j, None]
        )
        edge = self.f[key]
        if layer["cutoff"] is None:
            self.record((key,), (key,), (edge * scale,), lambda dy: (dy * scale,))
        else:
            val, dr = polynomial_cutoff(self.f["length"], *layer["cutoff"])
            self.record(
                (key, "length"),
                (key,),
                (edge * scale * val[:, None],),
                lambda dy: (
                    dy * scale * val[:, None],
                    (dy * scale * edge).sum(-1) * dr,
                ),
            )
        self.reduce_edges(key, "atomic_energy", "sum", layer["factor"])
