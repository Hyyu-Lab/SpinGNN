"""Center-atom-tiled NumPy reference evaluator for a SpinGNN2 v1 contract."""

from dataclasses import dataclass
from operator import index

import numpy as np

from ._evidence import evidence
from ._tile import Sweep, Tile, scatter
from .contract import load_contract
from .harmonics import cartesian_norm


class _Graph:
    def __init__(self, model, data, tile_atoms, radial_mode):
        if radial_mode not in ("analytic", "spline"):
            raise ValueError("radial_mode must be analytic or spline")
        self.mode = radial_mode
        pos = np.asarray(data["pos"], dtype=model.dtype)
        self.n = len(pos)
        if pos.shape != (self.n, 3) or not self.n:
            raise ValueError("pos must have shape [N,3] with at least one center")
        self.width = self.n if tile_atoms is None else index(tile_atoms)
        if self.width < 1:
            raise ValueError("tile_atoms must be positive")
        self.types = self.indices(data["atom_types"], "atom_types").reshape(-1)
        if len(self.types) != self.n or np.any(self.types >= model.num_types):
            raise ValueError("atom_types must contain one valid species index per atom")
        self.batch = self.indices(
            data.get("batch", np.zeros(self.n, dtype=int)), "batch"
        )
        if self.batch.shape != (self.n,) or self.batch.max() >= self.n:
            raise ValueError("Invalid batch indices")
        self.counts = np.bincount(self.batch).astype(model.dtype)
        self.frames = len(self.counts)
        if (
            not self.counts.all()
            or int(data.get("num_batch", self.frames)) != self.frames
        ):
            raise ValueError("Invalid batch frame count")
        edge = self.indices(data["edge_index"], "edge_index")
        if edge.ndim != 2 or edge.shape[0] != 2 or np.any(edge >= self.n):
            raise ValueError("edge_index must index atom pairs with shape [2,E]")
        center, neighbor = edge
        if np.any(self.batch[center] != self.batch[neighbor]):
            raise ValueError("An edge crosses batch frames")
        # Group once, stably, so each tile streams a contiguous edge interval.
        order = np.argsort(center, kind="stable")
        self.center, self.neighbor = center[order], neighbor[order]
        self.offsets = np.r_[0, np.cumsum(np.bincount(center, minlength=self.n))]
        self.pos = pos
        self.cell = self.shifts = None
        self.volume = None
        if "cell" in data:
            cell = np.broadcast_to(
                np.asarray(data["cell"], dtype=model.dtype).reshape(-1, 3, 3),
                (self.frames, 3, 3),
            )
            pbc = np.asarray(data.get("pbc", False))
            if not np.all(np.broadcast_to(pbc, (self.frames, 3))):
                raise ValueError(
                    "Stress requires explicitly periodic boundary conditions"
                )
            self.volume = np.abs(np.linalg.det(cell))
            if not np.isfinite(self.volume).all() or np.any(
                self.volume <= np.finfo(model.dtype).eps
            ):
                raise ValueError("Stress requires a finite non-zero cell volume")
            shifts = np.asarray(data["edge_cell_shift"], dtype=model.dtype)
            if shifts.shape != (len(center), 3):
                raise ValueError("Invalid periodic edge shifts")
            self.cell, self.shifts = cell, shifts[order]

    @staticmethod
    def indices(value, name):
        array = np.asarray(value)
        if array.dtype.kind not in "iu" or np.any(array < 0):
            raise ValueError(f"{name} must contain nonnegative integers")
        return np.array(array, dtype=np.intp, copy=True)

    def tiles(self):
        for start in range(0, self.n, self.width):
            stop = min(start + self.width, self.n)
            edges = slice(self.offsets[start], self.offsets[stop])
            i, j = self.center[edges], self.neighbor[edges]
            tile = Tile(
                start,
                stop,
                i - start,
                j,
                self.types[start:stop],
                self.types[i],
                self.types[j],
                self.mode,
            )
            r = self.pos[j] - self.pos[i]
            if self.cell is not None:
                r += np.einsum(
                    "ei,eij->ej", self.shifts[edges], self.cell[self.batch[i]]
                )
            yield Sweep(tile, {"r": r, "length": cartesian_norm(r)})


@dataclass(frozen=True)
class LatticeCache:
    """Fixed geometry plus lattice layer activations and reusable reverse maps."""

    model: object
    graph: _Graph
    tiles: tuple


class ReferenceModel:
    """Evaluate NumPy AtomicDataDict inputs, using the original directed edge list."""

    def __init__(self, contract):
        self.contract = contract
        self.dtype = np.dtype(contract["dtype"])
        self.lattice, self.spin = [], []
        self.bec = self.evidential = self.total = None
        in_spin = False
        self.magnetic = contract["family"] in (
            "Allegro_soc",
            "Allegro_nosoc",
            "Allegro_soc_e2IP",
        )
        for layer in contract["layers"]:
            op = layer["op"]
            if op.startswith("OneHotAtomEncoding"):
                self.num_types = layer["num_types"]
            if op == "AtomwiseReduce":
                self.total = layer
                continue
            if op == "e2IP_nodewise":
                self.evidential = layer
                continue
            if op == "CartesianBECReduce":
                self.bec = layer
            if op in (
                "OneHotAtomEncoding_spin",
                "HB_Energy",
                "Biquad_Energy",
                "Spin_JMatrix_Energy",
                "Spin_ani_Energy",
            ):
                in_spin = True
            # The builders interleave spin physics with independent lattice MLP
            # readouts. Cache these coefficients alongside J, A and the latents.
            lattice_coeff = layer["name"] in ("edge_hb_coeff", "edge_biquad_coeff")
            (self.spin if in_spin and not lattice_coeff else self.lattice).append(layer)
        if self.total is None:
            raise ValueError("Contract contains no total energy reduction")

    @classmethod
    def load(cls, manifest):
        return cls(load_contract(manifest))

    def _stage1(self, graph):
        for sweep in graph.tiles():
            for layer in self.lattice:
                sweep.run(layer)
            yield sweep

    def evaluate(self, data, *, tile_atoms=256, radial_mode="analytic"):
        """Stream one tile through both stages and its energy adjoint before reuse."""
        graph = _Graph(self, data, tile_atoms, radial_mode)
        return self._evaluate(
            graph, self._stage1(graph), data.get("spin"), cached=False
        )

    def evaluate_stage1_lattice(self, data, *, tile_atoms=256, radial_mode="analytic"):
        graph = _Graph(self, data, tile_atoms, radial_mode)
        return LatticeCache(self, graph, tuple(self._stage1(graph)))

    def evaluate_stage2_spin_cached(self, cache, spin):
        """Reuse the lattice forward pass, including its force pullback."""
        if not isinstance(cache, LatticeCache) or cache.model is not self:
            raise ValueError("Lattice cache belongs to a different model")
        return self._evaluate(cache.graph, cache.tiles, spin, cached=True)

    def _evaluate(self, graph, tiles, spin, cached):
        if self.magnetic or spin is not None:
            spin = np.asarray(spin, dtype=self.dtype)
            if spin.shape != (graph.n, 3):
                raise ValueError("spin must have shape [N,3]")
        force = np.zeros((graph.n, 3), dtype=self.dtype)
        spin_force = np.zeros_like(force)
        atomic = {}
        for lattice in tiles:
            sweep = Sweep(lattice.tile, lattice.f, lattice.tape) if cached else lattice
            t, f = sweep.tile, sweep.f
            atoms = slice(t.start, t.stop)
            center = t.center + t.start
            if spin is not None:
                f["i"], f["j"] = spin[center], spin[t.neighbor]
            for layer in self.spin:
                sweep.run(layer)
            energy = f["atomic_energy"]
            seed = np.full_like(energy, self.total["constant"])
            if self.total["reduce"] == "mean":
                seed /= graph.counts[graph.batch[atoms], None]
            grads = sweep.reverse(seed)
            radius = f["length"][:, None]
            unit = np.divide(
                f["r"], radius, out=np.zeros_like(f["r"]), where=radius != 0
            )
            dr = (
                grads.get("r", 0)
                + np.asarray(grads.get("length", 0)).reshape(-1, 1) * unit
            )
            np.add.at(force, center, dr)
            np.add.at(force, t.neighbor, -dr)
            if "i" in grads:
                np.add.at(spin_force, center, -grads["i"])
            if "j" in grads:
                np.add.at(spin_force, t.neighbor, -grads["j"])
            local = {
                "atomic_energy": energy,
                "atom_virial": scatter(
                    -dr[:, :, None] * f["r"][:, None, :], t.center, t.size
                ),
            }
            if self.bec:
                local[self.bec["out_field"]] = f[self.bec["out_field"]]
            if self.evidential:
                local.update(
                    evidence(
                        self.evidential, f, spin[atoms] if spin is not None else None
                    )
                )
            for key, value in local.items():
                if key not in atomic:
                    atomic[key] = np.empty(
                        (graph.n, *value.shape[1:]), dtype=value.dtype
                    )
                atomic[key][atoms] = value

        def frame_sum(values):
            return scatter(values, graph.batch, graph.frames)

        result = atomic
        energy = frame_sum(atomic["atomic_energy"]) * self.total["constant"]
        if self.total["reduce"] == "mean":
            energy /= graph.counts[:, None]
        virial = frame_sum(atomic["atom_virial"])
        virial = (virial + virial.swapaxes(1, 2)) * 0.5
        result.update(
            total_energy=energy,
            forces=force,
            virial=virial,
            stress=(
                np.empty(0, dtype=self.dtype)
                if graph.volume is None
                else -virial / graph.volume[:, None, None]
            ),
        )
        if self.magnetic:
            sf = spin_force * np.sum(spin * spin, axis=1, keepdims=True)
            result["spin_forces"] = sf
            result["spin_forces_vert"] = (
                sf - (sf * spin).sum(-1, keepdims=True) * spin
            ) * self.contract["spin_fv_coeff"]
        if self.bec:
            key = self.bec["out_field"]
            if self.bec["pre_asr"]:
                result[self.bec["pre_asr"]] = result[key]
            if self.bec["asr"] == "hard":
                mean = frame_sum(result[key]) / graph.counts[:, None]
                result[key] = result[key] - mean[graph.batch]
            if self.bec["total"]:
                result[self.bec["total"]] = (
                    frame_sum(result[key]) / graph.counts[:, None]
                )
        if self.evidential:
            for prefix in self.evidential["prefixes"]:
                uepi = result[prefix + "_Uepi"]
                result["total_" + prefix + "_Uepi"] = frame_sum(uepi)
                count = graph.counts
                if prefix == "sf":
                    count = frame_sum((cartesian_norm(spin) > 1e-6).astype(self.dtype))
                result[prefix + "_uncertainty"] = frame_sum(
                    np.trace(uepi, axis1=1, axis2=2)
                )[:, None] / np.maximum(1, count[:, None])
        for layer in self.contract["rescales"]:
            for key in result.keys() & set(layer["scale_keys"]):
                result[key] = result[key] * layer["scale"]
            for key in result.keys() & set(layer["square_keys"]):
                result[key] = result[key] * layer["scale"] ** 2
            for key in result.keys() & set(layer["shift_keys"]):
                result[key] = result[key] + layer["shift"]
        return result
