"""ASE calculator for the native SpinGNN2 engine."""

import numpy as np
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)
from ase.neighborlist import neighbor_list
from ase.stress import full_3x3_to_voigt_6_stress

from .native import NativeModel


class SpinGNN2Calculator(Calculator):
    """Use model units (eV, Angstrom). Cartesian spin vectors retain their magnitude."""

    def __init__(self, model, *, type_names=None, **native_options):
        super().__init__()
        if isinstance(model, NativeModel):
            if native_options:
                raise ValueError("Pass native options when constructing NativeModel")
            self.model = model
        else:
            self.model = NativeModel.load(model, **native_options)
        info = self.model.info
        names = type_names if type_names is not None else info.get("type_names")
        if (
            not names
            or len(names) != info["num_types"]
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "Provide the chemical symbols in training type order via type_names"
            )
        self.types = {symbol: i for i, symbol in enumerate(names)}
        self.implemented_properties = [
            "energy",
            "energies",
            "forces",
            "stress",
            "virial",
            "atom_virial",
        ]
        if info["magnetic"]:
            self.implemented_properties += ["spin_forces", "spin_forces_vert"]
        if "bec" in info:
            self.implemented_properties += [info["bec"], "total_born_effective_charges"]
        for prefix in info["prefixes"]:
            self.implemented_properties += [
                prefix + "_" + k
                for k in ("nu", "kappa", "sigma", "Uepi", "uncertainty")
            ]
            self.implemented_properties += ["total_" + prefix + "_Uepi"]

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol)
        if self.atoms is not None and not np.array_equal(
            self.atoms.arrays.get("spin"), atoms.arrays.get("spin")
        ):
            changes.append("spin")
        return changes

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms
        i, j, s = neighbor_list(
            "ijS", a, self.model.info["r_max"], self_interaction=False
        )
        try:
            ids = [self.types[symbol] for symbol in a.get_chemical_symbols()]
        except KeyError as exc:
            raise ValueError(f"Unknown model species {exc.args[0]}") from exc
        data = {
            "pos": a.positions,
            "atom_types": np.asarray(ids),
            "edge_index": np.stack((i, j)),
        }
        if np.any(a.pbc):
            # ASE selects images along the periodic axes. Completing the
            # unused cell rows also supports rank-one chains and rank-two slabs.
            # Only a three-periodic system has the native 3D stress observable.
            data.update(
                cell=a.cell.complete().array, pbc=np.ones(3, bool), edge_cell_shift=s
            )
        if self.model.info["magnetic"]:
            spin = a.arrays.get("spin", a.arrays.get("initial_magmoms"))
            if spin is None:
                raise ValueError("Magnetic models require spin or initial_magmoms")
            if spin.ndim == 1:
                spin = np.column_stack((np.zeros_like(spin), np.zeros_like(spin), spin))
            data["spin"] = spin
        out = self.model.evaluate(data)
        self.results = dict(out)
        self.results["energy"] = out["total_energy"].item()
        self.results["energies"] = out["atomic_energy"].ravel()
        if np.all(a.pbc) and out["stress"].size:
            self.results["stress"] = full_3x3_to_voigt_6_stress(out["stress"][0])
        else:
            self.results.pop("stress")
            if "stress" in properties:
                raise PropertyNotImplementedError(
                    "Stress requires a fully periodic cell"
                )
        if "born_effective_charges" in out:
            self.results["born_effective_charges"] = out[
                "born_effective_charges"
            ].reshape(-1, 3, 3)
