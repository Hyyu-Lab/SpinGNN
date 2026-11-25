"""Prepare full-width example evidence models and exported trained checkpoints."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from nequip.model import model_from_config
from nequip.utils import Config
from nequip.utils._global_options import global_options_context
from spingnn2.export import export_model
from spingnn2.jit import prepare_jit_artifact


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "e2ip": "06_allegro_e2ip.yaml",
    "soc-e2ip": "07_allegro_soc_e2ip.yaml",
}
MD_OUTPUT_SCALE = 0.001


def checksum(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def image_record(manifest):
    image = manifest.with_name(json.loads(manifest.read_text())["artifact"])
    return {
        "manifest": str(manifest),
        "manifest_sha256": checksum(manifest),
        "image_sha256": checksum(image),
    }


def evidence_models(args):
    models = {}
    for family, example in EXAMPLES.items():
        config = yaml.safe_load((ROOT / "example" / example).read_text())
        config.update(
            avg_num_neighbors=56.0,
            # Untrained weights can heat the lattice by orders of magnitude.
            # Apply a fixed output scale so this exercises thermal motion and
            # readouts with the original layer widths and neighbor workload.
            global_rescale_scale=MD_OUTPUT_SCALE,
            per_species_rescale_shifts=0.0,
            per_species_rescale_scales=1.0,
            default_dtype="float32",
            allow_tf32=False,
            float32_matmul_precision="highest",
            spingnn_compile={"enabled": False},
            train_autocast={"enabled": False},
        )
        with global_options_context(config):
            torch.manual_seed(1793)
            model = model_from_config(Config.from_dict(config), initialize=True).eval()
            contract, _ = export_model(model, args.output / family)
        artifact = prepare_jit_artifact(
            contract,
            backend="cuda",
            gpu_arch=args.arch,
            cache_dir=args.output / "cache",
            library=args.library,
        )
        models[family] = dict(
            contract=str(contract),
            contract_sha256=checksum(contract),
            seed=1793,
            output_scale=MD_OUTPUT_SCALE,
            example=str(Path("example") / example),
            **image_record(artifact),
        )
    return models


def trained_models(args):
    index = json.loads(args.trained.read_text())
    for precision, spec in index["models"].items():
        contract = Path(spec["contract"])
        if checksum(contract) != spec["contract_sha256"]:
            raise RuntimeError(f"Changed checkpoint export {contract}")
        if checksum(contract.with_suffix(".bin")) != spec["weights_sha256"]:
            raise RuntimeError(
                f"Changed checkpoint weights {contract.with_suffix('.bin')}"
            )
        # Test the supported CLI on exactly the trained exports.
        log = subprocess.check_output(
            [
                sys.executable,
                str(ROOT / "tools/ci/source_python.py"),
                "spingnn2-prepare-jit-artifact",
                "--model",
                str(contract),
                "--backend",
                "cuda",
                "--arch",
                args.arch,
                "--precision",
                precision,
                "--library",
                str(args.library),
                "--cache-dir",
                str(args.output / "cache"),
            ],
            text=True,
        )
        spec.update(image_record(Path(log.strip().splitlines()[-1])))
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--arch", default="sm120", choices=("sm89", "sm120"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trained", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = {"evidence": evidence_models(args)}
    if args.trained is not None:
        cases["trained"] = trained_models(args)
    for tag, data in cases.items():
        (args.output / f"{tag}.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
