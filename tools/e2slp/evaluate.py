#!/usr/bin/env python3
"""e2SLP evaluation: spin-tangent uncertainty per structure and error correlation.

Run deployed inference through tools/ci/source_python.py. Saved-array scoring
needs only NumPy/SciPy (and matplotlib for labelled correlation plots).
"""
import argparse
import csv
import hashlib
import json
import logging
from contextlib import closing, nullcontext
from itertools import islice
from pathlib import Path
import tempfile
import zipfile

import numpy as np


SPIN_EPS = 1e-6
ARRAY_KEYS = {
    "spin": ("spin",),
    "forces": ("forces", "force_hat"),
    "force_label": ("force_label",),
    "spin_forces": ("spin_forces", "spin_fv_hat"),
    "spin_force_label": ("spin_force_label", "spin_fv_label"),
}


def finite_array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(
            f"{name} must be finite with shape {shape}; got {result.shape}"
        )
    return result


def atom_traces(covariance, spin=None, target="spin-forces"):
    """Return physical per-atom traces and magnetic mask, in float64.

    For spin-forces, return Tr(B^T U B) on the tangent plane. Nonmagnetic
    atoms contribute zero. For forces, use the full Cartesian 3D trace.
    """
    if target not in ("spin-forces", "forces"):
        raise ValueError("target must be spin-forces or forces")
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 3 or cov.shape[1:] != (3, 3) or len(cov) == 0:
        raise ValueError("covariance must have shape (N, 3, 3), with N > 0")
    cov = finite_array(cov, cov.shape, "covariance")
    # Allow float32 arithmetic noise relative to the covariance's own scale.
    tolerance = 1e-5 * max(float(np.abs(cov).max()), np.finfo(float).tiny)
    if np.abs(cov - cov.swapaxes(-1, -2)).max() > tolerance:
        raise ValueError("covariance is not symmetric")
    cov = 0.5 * (cov + cov.swapaxes(-1, -2))
    if spin is None:
        if target == "spin-forces":
            raise ValueError("spin-forces scoring requires spin directions")
        mask = np.zeros(len(cov), dtype=bool)
    else:
        spin = finite_array(spin, (len(cov), 3), "spin")
        norm = np.linalg.norm(spin, axis=-1)
        mask = norm > SPIN_EPS
    if target == "forces":
        projected = cov
    else:
        unit = np.divide(
            spin, norm[:, None], out=np.zeros_like(spin), where=mask[:, None]
        )
        ref = np.zeros_like(unit)
        ref[:, 2] = 1
        ref[np.abs(unit[:, 2]) > 0.9] = [1, 0, 0]
        first = np.cross(unit, ref)
        first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), SPIN_EPS)
        basis = np.stack((first, np.cross(unit, first)), axis=-1)
        projected = np.einsum("nia,nij,njb->nab", basis, cov, basis)
    if np.linalg.eigvalsh(projected).min() < -tolerance:
        raise ValueError("covariance has a negative eigenvalue in the scored subspace")
    return np.trace(projected, axis1=-2, axis2=-1).clip(min=0), mask


def score_frame(record, target="spin-forces", reduction="mean", atom_scope="all"):
    """Score a frame; the same atom scope is used for component RMSE."""
    traces, magnetic = atom_traces(record["covariance"], record.get("spin"), target)
    if atom_scope not in ("all", "magnetic") or reduction not in ("mean", "max"):
        raise ValueError("invalid atom_scope or reduction")
    selected = (
        magnetic if atom_scope == "magnetic" else np.ones(len(traces), dtype=bool)
    )
    if not selected.any():
        raise ValueError("magnetic atom scope requires at least one nonzero spin")
    score = traces[selected].mean() if reduction == "mean" else traces[selected].max()
    row = {key: record[key] for key in ("frame", "source", "timestep")}
    row.update(
        n_atoms=len(traces), n_magnetic=int(magnetic.sum()), uncertainty=float(score)
    )
    for prediction, reference, name in (
        ("forces", "force_label", "force_rmse"),
        ("spin_forces", "spin_force_label", "spin_force_rmse"),
    ):
        if reference in record and prediction in record:
            pred = finite_array(record[prediction], (len(traces), 3), prediction)
            ref = finite_array(record[reference], pred.shape, reference)
            row[name] = float(np.sqrt(np.mean((pred[selected] - ref[selected]) ** 2)))
    return row


def archive_frames(path, target, covariance_key=None):
    """Read legacy fixed-shape or this script's flattened, variable-size NPZ."""
    with np.load(path, allow_pickle=False) as archive:
        candidates = (
            (covariance_key,)
            if covariance_key
            else (
                ("sf_Uepi", "Uepi", "covariance")
                if target == "spin-forces"
                else ("f_Uepi", "Uepi", "covariance")
            )
        )
        key = next((k for k in candidates if k in archive), None)
        if key is None:
            raise ValueError(f"No covariance array found; expected one of {candidates}")
        cov = archive[key]
        flat = "frame_ptr" in archive
        if flat:
            ptr = archive["frame_ptr"]
            if (
                ptr.ndim != 1
                or ptr.dtype.kind not in "iu"
                or len(ptr) < 2
                or ptr[0] != 0
                or ptr[-1] != len(cov)
                or (np.diff(ptr) <= 0).any()
                or cov.ndim != 3
            ):
                raise ValueError("invalid frame_ptr / flattened covariance")
            count = len(ptr) - 1
        else:
            if cov.ndim != 4 or cov.shape[0] == 0:
                raise ValueError(
                    "legacy covariance must have shape (frames, atoms, 3, 3)"
                )
            count = len(cov)
        arrays = {}
        for name, aliases in ARRAY_KEYS.items():
            alias = next((a for a in aliases if a in archive), None)
            if alias is not None:
                value = archive[alias]
                if value.shape != cov.shape[:-2] + (3,):
                    raise ValueError(
                        f"{alias} does not match covariance frame/atom dimensions"
                    )
                arrays[name] = value
        metadata, presence_masks = {}, {}
        for name in ("source", "timestep"):
            if name in archive:
                metadata[name] = archive[name]
                if metadata[name].shape != (count,):
                    raise ValueError(f"invalid {name} array length")
        for name in arrays:
            presence = f"{name}_present"
            if presence in archive:
                presence_masks[name] = archive[presence]
                if presence_masks[name].shape != (count,):
                    raise ValueError(f"invalid {presence} array length")
        for i in range(count):
            index = slice(int(ptr[i]), int(ptr[i + 1])) if flat else i
            record = dict(
                frame=i,
                source=str(path.resolve()),
                timestep="",
                covariance=cov[index],
                input_covariance_key=key,
            )
            for name, value in metadata.items():
                record[name] = str(value[i])
            for name, value in arrays.items():
                if name not in presence_masks or bool(presence_masks[name][i]):
                    record[name] = value[index]
            yield record


def deployed_frames(args):
    """Infer from stripped geometry inputs with gradients enabled for forces."""
    from ase.io import iread
    from nequip.data.transforms import TypeMapper
    from nequip.scripts.deploy import (
        load_deployed_model,
        TYPE_NAMES_KEY,
        DEFAULT_DTYPE_KEY,
    )
    from nequip.utils._global_options import (
        global_options_context,
        _get_latest_global_options,
    )
    from nequip.utils.misc import dtype_from_name

    with global_options_context():
        model, metadata = load_deployed_model(
            str(args.model), device=args.device, set_global_options=True
        )
        inference_options = dict(_get_latest_global_options())
    dtype_name = metadata[DEFAULT_DTYPE_KEY]
    try:
        dtype_from_name(dtype_name)
    except KeyError as exc:
        raise ValueError(f"unsupported deployed default_dtype: {dtype_name!r}") from exc
    inference_options["default_dtype"] = dtype_name
    # Type names are normally chemical symbols. Explicit JSON mapping supports
    # custom names, without relying on calculator-specific species assumptions.
    names = metadata[TYPE_NAMES_KEY].split()
    mapping = (
        json.loads(args.species_map)
        if args.species_map
        else {name: name for name in names}
    )
    mapper = TypeMapper(
        chemical_symbol_to_type={s: names.index(t) for s, t in mapping.items()}
    )
    cov_key = args.covariance_key or (
        "sf_Uepi" if args.target == "spin-forces" else "f_Uepi"
    )
    trajectory = iread(args.dataset, index=":")
    with closing(trajectory):
        yield from _infer_frames(
            islice(trajectory, args.limit),
            args,
            model,
            metadata,
            mapper,
            cov_key,
            inference_options,
        )


def _infer_frames(trajectory, args, model, metadata, mapper, cov_key, options):
    """Restore process options before yielding, even on partial consumption."""
    import torch
    from nequip.data import AtomicData, AtomicDataDict
    from nequip.scripts.deploy import R_MAX_KEY
    from nequip.utils._global_options import global_options_context

    for i, atoms in enumerate(trajectory):
        record = dict(
            frame=i,
            source=str(args.dataset.resolve()),
            timestep=str(atoms.info.get("timestep", "")),
        )
        record["input_covariance_key"] = cov_key
        if atoms.has("initial_magmoms"):
            spin = np.asarray(atoms.get_initial_magnetic_moments())
            # Scalar magnetic moments represent signed z-directed spins.
            if spin.ndim == 1:
                spin = np.column_stack((np.zeros_like(spin), np.zeros_like(spin), spin))
        if args.labels != "none":
            if atoms.calc is not None and "forces" in atoms.calc.results:
                record["force_label"] = np.array(
                    atoms.calc.results["forces"], copy=True
                )
            if "spin_forces_vert" in atoms.arrays:
                record["spin_force_label"] = atoms.arrays["spin_forces_vert"].copy()
        clean = atoms.copy()
        clean.calc = None
        clean.info.clear()
        for key in list(clean.arrays):
            if key not in ("numbers", "positions", "initial_magmoms", "spin_length"):
                del clean.arrays[key]
        if atoms.has("initial_magmoms"):
            # ASE cannot change an existing array from scalar to vector shape.
            del clean.arrays["initial_magmoms"]
            clean.set_initial_magnetic_moments(spin)
        # Construct geometry at the deployed precision: casting an already
        # rounded float32 graph cannot recover float64 positions or neighbors.
        with global_options_context(options), torch.enable_grad():
            data = mapper(
                AtomicData.from_ase(clean, r_max=float(metadata[R_MAX_KEY]))
            ).to(args.device)
            inputs = AtomicData.to_AtomicDataDict(data)
            if any(
                k in inputs
                for k in ("total_energy", "forces", "stress", "spin_forces_vert")
            ):
                raise ValueError("reference labels leaked into model inputs")
            if AtomicDataDict.SPIN_KEY in inputs:
                record["spin"] = (
                    inputs[AtomicDataDict.SPIN_KEY].detach().cpu().numpy().copy()
                )
            out = model(inputs)
        if cov_key not in out:
            raise ValueError(
                f"model has no {cov_key}; check --target / --covariance-key"
            )
        record["covariance"] = out[cov_key].detach().cpu().numpy()
        for dst, src in (("forces", "forces"), ("spin_forces", "spin_forces_vert")):
            if src in out:
                record[dst] = finite_array(
                    out[src].detach().cpu().numpy(), (len(atoms), 3), src
                )
        del out, inputs, data
        yield record
        del record
        if (i + 1) % 25 == 0:
            logging.info("Inferred %d frames", i + 1)


class PredictionArchive:
    """Spool atom arrays to disk; only frame metadata stays resident.

    Per-field segments retain their original dtype until final promotion, just
    like concatenate. Missing optional values contribute float64 zeros to that
    promotion. ZIP members are written in bounded blocks, without touching a
    trajectory-sized memory map or making a second concatenated array.
    """

    BLOCK_BYTES = 1024 * 1024

    def __init__(self, path, target):
        self.path = Path(path)
        self.covariance_key = "sf_Uepi" if target == "spin-forces" else "f_Uepi"
        self.temporary = None
        self.counts, self.sources, self.timesteps = [], [], []
        self.fields = {}

    def __enter__(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".predictions-", dir=self.path.parent
        )
        return self

    def __exit__(self, *exc):
        self.temporary.cleanup()

    def append(self, record):
        n = len(record["covariance"])
        if n == 0:
            raise ValueError("prediction frames must contain atoms")
        for name in ("covariance", *ARRAY_KEYS):
            if name not in record and name not in self.fields:
                continue
            tail = (3, 3) if name == "covariance" else (3,)
            if name not in self.fields:
                self.fields[name] = [(count, None) for count in self.counts]
            if name in record:
                array = np.asarray(record[name])
                if array.shape != (n, *tail) or array.dtype.kind not in "biufc":
                    raise ValueError(f"invalid prediction array {name}")
                with (Path(self.temporary.name) / name).open("ab") as stream:
                    array.tofile(stream)
                dtype = array.dtype
            else:
                dtype = None
            self.fields[name].append((n, dtype))
        self.counts.append(n)
        self.sources.append(record["source"])
        self.timesteps.append(record["timestep"])

    def finish(self):
        if not self.counts:
            raise ValueError("input contains no structures")
        staging = Path(self.temporary.name) / "predictions.npz"
        total = sum(self.counts)
        with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            arrays = {
                "frame_ptr": np.r_[0, np.cumsum(self.counts)],
                "source": np.asarray(self.sources),
                "timestep": np.asarray(self.timesteps),
            }
            for name, segments in self.fields.items():
                if name != "covariance":
                    arrays[f"{name}_present"] = np.asarray(
                        [dtype is not None for _, dtype in segments]
                    )
                tail = (3, 3) if name == "covariance" else (3,)
                width = 9 if name == "covariance" else 3
                dtype = np.result_type(
                    *{d if d is not None else np.dtype("float64") for _, d in segments}
                )
                key = self.covariance_key if name == "covariance" else name
                with archive.open(key + ".npy", "w", force_zip64=True) as output:
                    np.lib.format.write_array_header_2_0(
                        output,
                        dict(
                            descr=np.lib.format.dtype_to_descr(dtype),
                            fortran_order=False,
                            shape=(total, *tail),
                        ),
                    )
                    with (Path(self.temporary.name) / name).open("rb") as source:
                        for count, original_dtype in segments:
                            remaining = count * width
                            chunk_size = self.BLOCK_BYTES // max(
                                dtype.itemsize,
                                original_dtype.itemsize if original_dtype else 1,
                            )
                            while remaining:
                                size = min(remaining, chunk_size)
                                if original_dtype is None:
                                    block = np.zeros(size, dtype=dtype)
                                else:
                                    block = np.fromfile(
                                        source, dtype=original_dtype, count=size
                                    )
                                    if block.size != size:
                                        raise ValueError("truncated prediction spool")
                                    block = block.astype(dtype, copy=False)
                                output.write(block.tobytes())
                                remaining -= size
            for name, array in arrays.items():
                with archive.open(name + ".npy", "w", force_zip64=True) as output:
                    np.lib.format.write_array(output, array, allow_pickle=False)
        staging.replace(self.path)


def save_predictions(path, records, target):
    """Consume predictions once and publish a compatible pickle-free archive."""
    with PredictionArchive(path, target) as archive:
        for record in records:
            archive.append(record)
            del record
        archive.finish()


def correlations(rows):
    from scipy.stats import spearmanr, pearsonr

    result = {}
    for column in ("force_rmse", "spin_force_rmse"):
        pairs = [(r["uncertainty"], r[column]) for r in rows if column in r]
        if not pairs:
            continue
        x, y = np.asarray(pairs).T
        if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            result[column] = dict(
                frames=len(x),
                spearman_rho=None,
                pearson_r=None,
                reason="need >=3 frames and nonconstant scores/errors",
            )
        else:
            result[column] = dict(
                frames=len(x),
                spearman_rho=float(spearmanr(x, y).statistic),
                pearson_r=float(pearsonr(x, y).statistic),
            )
    return result


def write_plot(output, rows, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = list(report["correlations"])
    if not columns:
        return
    fig, axes = plt.subplots(
        1,
        len(columns),
        figsize=(5.4 * len(columns), 4.3),
        squeeze=False,
        layout="constrained",
    )
    for ax, column in zip(axes[0], columns):
        x, y = np.asarray(
            [(r["uncertainty"], r[column]) for r in rows if column in r]
        ).T
        ax.scatter(x, y, s=9, alpha=0.5)
        ax.set_xlabel(
            f"{report['reduction']} {report['covariance_key']} trace ({report['atom_scope']} atoms)"
        )
        ax.set_ylabel(f"{column} (model units)")
        rho = report["correlations"][column]["spearman_rho"]
        ax.set_title(
            "Correlation undefined" if rho is None else f"Spearman rho = {rho:.4f}"
        )
    fig.savefig(output / "correlation.png", dpi=240)
    fig.savefig(output / "correlation.pdf")
    plt.close(fig)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="TorchScript deployed model")
    source.add_argument(
        "--predictions", type=Path, help="Saved NPZ; see skill for array schema"
    )
    parser.add_argument(
        "--dataset", type=Path, help="ASE-readable trajectory, required with --model"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New or empty output directory"
    )
    parser.add_argument(
        "--target", choices=("spin-forces", "forces"), default="spin-forces"
    )
    parser.add_argument("--reduction", choices=("mean", "max"), default="mean")
    parser.add_argument("--atom-scope", choices=("all", "magnetic"), default="all")
    parser.add_argument(
        "--covariance-key",
        help="Override covariance key; defaults to sf_Uepi or f_Uepi",
    )
    parser.add_argument(
        "--labels",
        choices=("auto", "required", "none"),
        default="auto",
        help="required needs target reference labels in every frame",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--species-map", help="JSON chemical-symbol to model-type-name mapping"
    )
    parser.add_argument("--limit", type=int, help="Process only the first N frames")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.model) != bool(args.dataset):
        parser.error(
            "--dataset is required with --model and cannot accompany --predictions"
        )
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    frames = (
        deployed_frames(args)
        if args.model
        else archive_frames(args.predictions, args.target, args.covariance_key)
    )
    rows, input_covariance_keys = [], set()
    writer = (
        PredictionArchive(args.output / "predictions.npz", args.target)
        if args.model
        else nullcontext()
    )
    with closing(frames), writer as archive:
        for record in islice(frames, args.limit):
            i = len(rows)
            if args.labels == "none":
                record = {
                    k: v
                    for k, v in record.items()
                    if k not in ("force_label", "spin_force_label")
                }
            required = (
                "spin_force_label" if args.target == "spin-forces" else "force_label"
            )
            if args.labels == "required" and required not in record:
                raise ValueError(f"frame {i} missing required {required}")
            row = score_frame(record, args.target, args.reduction, args.atom_scope)
            if (
                args.labels == "required"
                and (
                    "spin_force_rmse" if args.target == "spin-forces" else "force_rmse"
                )
                not in row
            ):
                raise ValueError(f"frame {i} missing target prediction for RMSE")
            rows.append(row)
            input_covariance_keys.add(record["input_covariance_key"])
            if archive is not None:
                archive.append(record)
            del record
        if not rows:
            raise ValueError("input contains no structures")
        if archive is not None:
            archive.finish()
    fields = ["frame", "source", "timestep", "n_atoms", "n_magnetic", "uncertainty"]
    fields += [
        key for key in ("force_rmse", "spin_force_rmse") if any(key in r for r in rows)
    ]
    with (args.output / "per_structure.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = dict(
        schema_version=1,
        frames=len(rows),
        target=args.target,
        reduction=args.reduction,
        atom_scope=args.atom_scope,
        magnetic_spin_norm_threshold=SPIN_EPS,
        covariance_key=args.covariance_key
        or ("sf_Uepi" if args.target == "spin-forces" else "f_Uepi"),
        input_covariance_keys=sorted(input_covariance_keys),
        projection="B^T Uepi B, tangent to spin"
        if args.target == "spin-forces"
        else "3D Cartesian",
        units="covariance in squared model target units; no display scaling",
        display_scale=1.0,
        rmse_definition="sqrt(mean(error**2 over selected atoms and 3 Cartesian components))",
        correlations=correlations(rows),
        script_sha256=sha256(__file__),
        input_sha256={
            str(p.resolve()): sha256(p)
            for p in (args.model, args.dataset, args.predictions)
            if p
        },
        command_options={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        notes=[
            "Correlations are structure-level, not pooled atom-level.",
            "Correlation measures ranking, not absolute uncertainty calibration.",
            "Magnetic/all means differ if magnetic fractions vary between frames.",
        ],
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    if not args.no_plot:
        write_plot(args.output, rows, report)
    print(
        json.dumps(
            dict(
                output=str(args.output.resolve()),
                frames=len(rows),
                correlations=report["correlations"],
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
