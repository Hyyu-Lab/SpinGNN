"""Train and export compact e2SLP checkpoints from a pinned ab initio MD example."""

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml
from ase.io import read, write


ROOT = Path(__file__).resolve().parents[1]
DATASET_URL = (
    "https://raw.githubusercontent.com/materialsproject/pymatgen/"
    "c7397054015500cc6d287e2c44804346fed1f5e1/test-files/io/vasp/outputs/"
    "vasprun.md.xml.gz"
)
DATASET_SHA256 = "61774b76374744f07bb40b2fad2f2a72556be1009ed8541d27c386dff590c8ff"
HOLDOUT = [4, 8]
TRAIN = [0, 1, 2, 5, 6, 9]
VALIDATION = [3, 7]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def run(command, output):
    with output.open("w") as log:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/ci/source_python.py"), *command],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"CLI failed with exit {result.returncode}, see {output}")


def train(dataset, output, epochs):
    if digest(dataset) != DATASET_SHA256:
        raise ValueError(f"Expected the pinned trajectory from {DATASET_URL}")
    frames = read(dataset, index=":", format="vasp-xml")
    if len(frames) != 10 or not all(
        len(frame) == 64
        and set(frame.get_chemical_symbols()) == {"Si"}
        and np.isfinite(frame.get_potential_energy())
        and np.isfinite(frame.get_forces()).all()
        and np.isfinite(frame.get_stress()).all()
        for frame in frames
    ):
        raise ValueError("Invalid energy, force, or stress labels in the Si trajectory")

    training_data = output / "fit.extxyz"
    holdout = output / "held-out.extxyz"
    write(training_data, [frames[i] for i in TRAIN + VALIDATION], format="extxyz")
    write(holdout, [frames[i] for i in HOLDOUT], format="extxyz")

    report = dict(
        dataset_url=DATASET_URL,
        dataset_sha256=DATASET_SHA256,
        frames=dict(train=TRAIN, validation=VALIDATION, parity=HOLDOUT),
        held_out=str(holdout),
        held_out_sha256=digest(holdout),
        models={},
    )
    template = ROOT / "example/06_allegro_e2ip.yaml"
    for precision in ("float64", "float32"):
        config = yaml.safe_load(template.read_text())
        config.update(
            {
                name: [8]
                for name, width in config.items()
                if name.endswith("_mlp_latent_dimensions") and width
            }
        )
        config.update(
            root=str(output / "runs"),
            run_name=precision,
            dataset_file_name=str(training_data),
            dataset_root=str(output / f"data-{precision}"),
            chemical_symbol_to_type={"Si": 0},
            n_train=len(TRAIN),
            n_val=len(VALIDATION),
            train_val_split="sequential",
            r_max=5.0,
            device="cpu",
            default_dtype=precision,
            max_epochs=epochs,
            batch_size=2,
            validation_batch_size=2,
            learning_rate=0.003,
            use_ema=False,
            num_layers=1,
            l_max=1,
            env_embed_multiplicity=2,
            e2ip_latent_channels=2,
        )
        config_path = output / f"{precision}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        run(["spingnn-train", str(config_path)], output / f"train-{precision}.log")
        directory = output / "runs" / precision
        with (directory / "metrics_epoch.csv").open() as stream:
            metrics = list(csv.DictReader(stream))
        if len(metrics) != epochs or not all(
            math.isfinite(float(value)) for row in metrics for value in row.values()
        ):
            raise RuntimeError(f"Missing or nonfinite epoch metrics for {precision}")
        checkpoint = directory / "trainer.pth"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        adam = state["state_dict"]["optim"]["state"]
        expected_updates = epochs * math.ceil(len(TRAIN) / config["batch_size"])
        if not adam or {float(v["step"]) for v in adam.values()} != {expected_updates}:
            raise RuntimeError(f"Optimizer did not finish {expected_updates} updates")
        if state["progress"]["iepoch"] != epochs:
            raise RuntimeError("Training checkpoint epoch mismatch")

        stem = output / f"trained-{precision}"
        run(
            ["spingnn2-export", str(directory), "--output", str(stem)],
            output / f"export-{precision}.log",
        )
        contract = stem.with_suffix(".sp2.json")
        weights = stem.with_suffix(".sp2.bin")
        report["models"][precision] = dict(
            directory=str(directory),
            checkpoint_sha256=digest(directory / "best_model.pth"),
            trainer_sha256=digest(checkpoint),
            optimizer_updates=expected_updates,
            epochs=epochs,
            contract=str(contract),
            contract_sha256=digest(contract),
            weights_sha256=digest(weights),
            first_epoch_loss=float(metrics[0]["validation_loss"]),
            last_epoch_loss=float(metrics[-1]["validation_loss"]),
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report = train(args.dataset.resolve(strict=True), args.output, args.epochs)
    (args.output / "exported.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
