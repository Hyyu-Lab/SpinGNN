"""Export and benchmark full example models on a periodic synthetic lattice."""

import argparse
from contextlib import contextmanager
import ctypes as ct
from hashlib import sha256
import itertools
import json
from pathlib import Path
from statistics import median
import sys
from threading import Event, Thread
from time import perf_counter
from tempfile import TemporaryDirectory


import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "nosoc": ("02_spin_nosoc.yaml", 10.4),
    "soc": ("03_spin_soc.yaml", 21.5),
    "born": ("04_born_effective_charge.yaml", 7.4),
}
SHAPES = {
    512: (8, 8, 8),
    4096: (16, 16, 16),
    32768: (32, 32, 32),
    262144: (64, 64, 64),
    500000: (100, 100, 50),
    1000000: (100, 100, 100),
    1048576: (128, 128, 64),
}


def configuration(name, neighbors):
    import yaml

    values = yaml.safe_load((ROOT / "example" / MODELS[name][0]).read_text())
    values.update(
        avg_num_neighbors=float(neighbors),
        global_rescale_scale=1.0,
        per_species_rescale_shifts=0.0,
        per_species_rescale_scales=1.0,
        default_dtype="float32",
        allow_tf32=False,
        float32_matmul_precision="highest",
        spingnn_compile={"enabled": False},
        train_autocast={"enabled": False},
    )
    return values


def lattice_offsets(neighbor_radius):
    extent = int(np.ceil(neighbor_radius))
    return np.array(
        [
            t
            for t in itertools.product(range(-extent, extent + 1), repeat=3)
            if 0 < sum(s * s for s in t) < neighbor_radius**2
        ],
        dtype=np.int32,
    )


def graph(n, radius, species, spin, neighbor_radius):
    """Generate an exact-size periodic graph without allocating an [N,E,3] grid."""
    shape = np.asarray(SHAPES[n], dtype=np.int32)
    lattice = (np.indices(shape, dtype=np.int32).reshape(3, n)).T.copy()
    offsets = lattice_offsets(neighbor_radius)
    degree = len(offsets)
    if not degree or 2 * int(np.max(np.abs(offsets))) >= min(shape):
        raise ValueError("Neighbor radius must fit the smallest periodic lattice")
    centers = np.arange(n, dtype=np.int32)
    edge_index = np.empty((2, n, degree), dtype=np.int32)
    edge_index[0] = centers[:, None]
    shifts = np.empty((n, degree, 3), dtype=np.float32)
    for j, offset in enumerate(offsets):
        images = lattice + offset
        shifts[:, j, :] = np.floor_divide(images, shape)
        images %= shape
        edge_index[1, :, j] = (images[:, 0] * shape[1] + images[:, 1]) * shape[
            2
        ] + images[:, 2]
    pitch = np.float32(radius / neighbor_radius)
    output = {
        "pos": lattice.astype(np.float32) * pitch,
        "edge_index": edge_index.reshape(2, -1),
        "edge_cell_shift": shifts.reshape(-1, 3),
        "cell": np.diag(shape.astype(np.float32) * pitch),
        "atom_types": centers % species,
        "pbc": np.ones(3, dtype=bool),
    }
    if spin:
        angles = np.float32(0.163) * (centers % 19)
        moments = np.stack((np.cos(angles), np.sin(angles), np.zeros(n)), axis=1)
        moments[centers % 7 == 0] = 0
        output["spin"] = moments.astype(np.float32)
    return output


class GPU:
    """Read sampled device residency and hardware identity through libcudart."""

    def __init__(self, library):
        self.lib = ct.CDLL(library)
        for name, args in {
            "cudaMemGetInfo": [ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)],
            "cudaDeviceGetPCIBusId": [ct.c_char_p, ct.c_int, ct.c_int],
            "cudaSetDevice": [ct.c_int],
        }.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, ct.c_int
        self.lib.cudaGetErrorString.argtypes = [ct.c_int]
        self.lib.cudaGetErrorString.restype = ct.c_char_p
        self.check(self.lib.cudaSetDevice(0))
        pci = ct.create_string_buffer(32)
        self.check(self.lib.cudaDeviceGetPCIBusId(pci, len(pci), 0))
        self.pci = pci.value.decode().lower()
        slot = Path("/sys/bus/pci/devices") / self.pci
        self.device_id = (slot / "device").read_text().strip()

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.cudaGetErrorString(code).decode())

    def used(self):
        free, total = ct.c_size_t(), ct.c_size_t()
        self.check(self.lib.cudaMemGetInfo(ct.byref(free), ct.byref(total)))
        return total.value - free.value, total.value

    @contextmanager
    def sample_peak(self, report):
        baseline, total = self.used()
        stats = {
            "device_baseline_bytes": baseline,
            "device_total_bytes": total,
            "peak_device_bytes": baseline,
        }
        done, errors = Event(), []

        def poll():
            try:
                while not done.wait(0.002):
                    used, _ = self.used()
                    stats["peak_device_bytes"] = max(stats["peak_device_bytes"], used)
            except RuntimeError as exc:
                errors.append(exc)

        sampler = Thread(target=poll, daemon=True)
        sampler.start()
        try:
            yield
        finally:
            done.set()
            sampler.join()
            try:
                used, _ = self.used()
                stats["peak_device_bytes"] = max(stats["peak_device_bytes"], used)
            except RuntimeError as exc:
                errors.append(exc)
            stats["peak_increase_bytes"] = stats["peak_device_bytes"] - baseline
            if errors:
                stats["sampling_error"] = str(errors[0])
            report.update(stats)


def _weight_digest(model):
    digest = sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _metadata_path(manifest):
    return Path(manifest).with_suffix(".benchmark.json")


def calibrated_model(config, checkpoint, output, eager):
    import torch

    from nequip.model import model_from_config
    from nequip.utils import Config
    from nequip.utils._global_options import global_options_context
    from spingnn2.export import export_model

    with global_options_context(config):
        torch.manual_seed(1793)
        model = model_from_config(Config.from_dict(config), initialize=True)
        if checkpoint:
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
        model.eval()
        digest = _weight_digest(model)
        if eager:
            metadata = _metadata_path(output)
            if metadata.is_file():
                recorded = json.loads(metadata.read_text())
                if recorded["state_sha256"] != digest:
                    raise ValueError(
                        "Eager weights differ from the exported benchmark state"
                    )
            return model.to(device="cuda"), digest
        manifest, weights = export_model(model, output)
        sidecar = _metadata_path(manifest)
        with TemporaryDirectory(dir=sidecar.parent) as directory:
            temp = Path(directory) / sidecar.name
            temp.write_text(
                json.dumps(
                    {
                        "state_sha256": digest,
                        "binary_sha256": sha256(weights.read_bytes()).hexdigest(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            temp.replace(sidecar)
    return None, digest


def verify_results(result, atoms, magnetic):
    fields = [
        "total_energy",
        "atomic_energy",
        "forces",
        "virial",
        "atom_virial",
        "stress",
    ]
    fields.extend(
        ["spin_forces", "spin_forces_vert"] if magnetic else ["born_effective_charges"]
    )
    for key in fields:
        if key not in result:
            raise ValueError(f"Missing benchmark field {key}")
        value = result[key]
        if (
            key
            in (
                "atomic_energy",
                "forces",
                "spin_forces",
                "atom_virial",
                "born_effective_charges",
            )
            and value.shape[0] != atoms
        ):
            raise ValueError(f"Incorrect benchmark atom count for {key}")
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite benchmark output {key}")


def measure(call, warmups, repetitions, synchronize=lambda: None):
    for _ in range(warmups):
        call()
        synchronize()
    samples = []
    for _ in range(repetitions):
        start = perf_counter()
        call()
        synchronize()
        samples.append(perf_counter() - start)
    return samples


def benchmark(args, config, report):
    magnetic = args.model != "born"
    data = graph(
        args.atoms,
        config["r_max"],
        len(config["chemical_symbol_to_type"]),
        magnetic,
        args.neighbor_radius,
    )
    gpu = GPU(args.cudart)
    if gpu.device_id != "0x2b85" and not args.allow_other_gpu:
        raise RuntimeError(f"Expected an RTX 5090, got PCI device {gpu.device_id}")
    report.update(
        lattice_shape=SHAPES[args.atoms],
        device_pci=gpu.pci,
        device_id=gpu.device_id,
        pt2_baseline_us_per_atom=MODELS[args.model][1],
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    with gpu.sample_peak(report):
        if args.method == "eager":
            import torch

            eager, report["state_sha256"] = calibrated_model(
                config, args.checkpoint, args.contract, True
            )
            # model construction temporarily overrides global math settings.
            # Select the requested inference mode after restoring that context.
            allow_tf32 = getattr(args, "allow_tf32", False)
            torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            tensors = {}
            for key, value in data.items():
                if key in ("edge_index", "atom_types"):
                    value = value.astype(np.int64)
                tensors[key] = torch.from_numpy(value).cuda()
            # The graph is resident for the eager benchmark, like the PT2 reference.
            torch.cuda.reset_peak_memory_stats()

            def evaluate():
                result = eager(dict(tensors))
                if args.eager_copyback:
                    for name in (
                        "total_energy",
                        "atomic_energy",
                        "forces",
                        "stress",
                        "virial",
                        "spin_forces",
                        "spin_forces_vert",
                        "atom_virial",
                        "born_effective_charges",
                    ):
                        if name in result:
                            result[name].detach().cpu()
                return result

            samples = measure(
                evaluate, args.warmups, args.repetitions, torch.cuda.synchronize
            )
            report["eager_copyback"] = args.eager_copyback
            report["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            check = evaluate()
            verify_results(
                {
                    key: value.detach().cpu().numpy()
                    for key, value in check.items()
                    if key
                    in {
                        "total_energy",
                        "atomic_energy",
                        "forces",
                        "virial",
                        "atom_virial",
                        "stress",
                        "spin_forces",
                        "spin_forces_vert",
                        "born_effective_charges",
                    }
                },
                args.atoms,
                magnetic,
            )
        else:
            from spingnn2.native import NativeModel

            if _metadata_path(args.contract).is_file():
                report.update(json.loads(_metadata_path(args.contract).read_text()))
                binary_sha256 = json.loads(args.contract.read_text())["sha256"]
                if report["binary_sha256"] != binary_sha256:
                    raise ValueError(
                        "Benchmark state metadata does not match the contract"
                    )
            with NativeModel.load(
                args.contract,
                backend="cuda",
                precision="float32",
                tile_atoms=args.tile_atoms,
                radial_mode=args.radial_mode,
                library=args.library,
            ) as model:
                if args.method == "cache":
                    if not magnetic:
                        raise ValueError(
                            "The lattice cache benchmark requires a magnetic model"
                        )
                    start = perf_counter()
                    with model.evaluate_stage1_lattice(data) as cache:
                        report["cache_creation_seconds"] = perf_counter() - start
                        update = dict(data, spin=np.roll(data["spin"], 1, axis=0))
                        # The paired full timing uses the same spins as the cached update.
                        full = measure(
                            lambda: model.evaluate(update),
                            args.warmups,
                            args.repetitions,
                        )
                        samples = measure(
                            lambda: model.evaluate_stage2_spin_cached(
                                cache, update["spin"]
                            ),
                            args.warmups,
                            args.repetitions,
                        )
                        report["full_recompute_seconds"] = full
                        report["cache_speedup"] = median(full) / median(samples)
                        verify_results(
                            model.evaluate_stage2_spin_cached(cache, update["spin"]),
                            args.atoms,
                            magnetic,
                        )
                else:

                    def evaluate():
                        return model.evaluate(data)

                    # Capacity rows time the first (cold) evaluation; validating
                    # the result afterward avoids a second million-center run.
                    if args.warmups == 0 and args.repetitions == 1:
                        start = perf_counter()
                        result = evaluate()
                        samples = [perf_counter() - start]
                    else:
                        samples = measure(evaluate, args.warmups, args.repetitions)
                        result = evaluate()
                    verify_results(result, args.atoms, magnetic)
                report["workspace"] = model.diagnostics.copy()
    report["elapsed_seconds"] = samples
    report["us_per_atom"] = 1e6 * median(samples) / args.atoms
    if args.profile and args.method == "native":
        with NativeModel.load(
            args.contract,
            backend="cuda",
            precision="float32",
            tile_atoms=args.tile_atoms,
            radial_mode=args.radial_mode,
            library=args.library,
        ) as model:
            model.evaluate(data)
            model.profile = True
            start = perf_counter()
            model.evaluate(data)
            report["profile"] = model.diagnostics.copy()
            report["profile_wall_seconds"] = perf_counter() - start


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=("export", "native", "eager", "cache"), required=True
    )
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, help="Optional eager state dictionary"
    )
    parser.add_argument("--library", type=Path)
    parser.add_argument("--cudart", default="libcudart.so.12")
    parser.add_argument("--atoms", type=int, choices=SHAPES, default=4096)
    parser.add_argument("--tile-atoms", type=int, default=512)
    parser.add_argument(
        "--neighbor-radius",
        type=float,
        default=2.42,
        help="Cutoff in lattice-spacing units (default graph has 56 edges/atom)",
    )
    parser.add_argument(
        "--radial-mode", choices=("analytic", "spline"), default="spline"
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--eager-copyback",
        action="store_true",
        help="Time eager transfers of output fields to host",
    )
    parser.add_argument("--allow-other-gpu", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.warmups < 0
        or args.repetitions < 1
        or args.tile_atoms < 1
        or args.neighbor_radius < 1
    ):
        parser.error("Invalid warmup, repetitions, tile size, or lattice cutoff")
    degree = len(lattice_offsets(args.neighbor_radius))
    config = configuration(args.model, degree)
    record = {
        "method": args.method,
        "model": args.model,
        "atoms": args.atoms,
        "edges": args.atoms * degree,
        "r_max": config["r_max"],
        "tile_atoms": args.tile_atoms,
        "radial_mode": args.radial_mode,
        "precision": "float32",
    }
    try:
        if args.method == "export":
            stem = str(args.contract).removesuffix(".sp2.json")
            _, digest = calibrated_model(config, args.checkpoint, stem, False)
            print(f"M2_EXPORT model={args.model} state_sha256={digest}", flush=True)
            return 0
        benchmark(args, config, record)
        record["status"] = "passed"
    except (RuntimeError, MemoryError, ValueError, OSError) as exc:
        record.update(status="failed", error=str(exc))
    print("M2_BENCH " + json.dumps(record, allow_nan=False, sort_keys=True), flush=True)
    return int(record["status"] != "passed")


if __name__ == "__main__":
    sys.exit(main())
