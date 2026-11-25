#!/usr/bin/env python3
"""Train all e2IP targets on derivative-consistent synthetic labels.

Run via tools/ci/source_python.py. This checks training plumbing, not physical
accuracy, uncertainty calibration, or reproduction of the paper's results.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from nequip.data import AtomicData
from nequip.data.transforms import TypeMapper
from nequip.train import Trainer
from nequip.utils._global_options import global_options_context


REPO = Path(__file__).resolve().parents[2]


def optimizer_checkpoint_evidence(checkpoint, expected_steps):
    """Check actual Adam counters, including parameters that never received a gradient."""
    optimizer = checkpoint.get("state_dict", {}).get("optim", {})
    groups = optimizer.get("param_groups", [])
    states = optimizer.get("state", {})
    parameter_ids = [parameter for group in groups for parameter in group["params"]]
    if not parameter_ids or len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("Missing or duplicated optimizer parameter groups")
    if not states or not set(states).issubset(parameter_ids):
        raise RuntimeError("Missing optimizer state or unknown optimizer parameters")
    counters = {}
    for parameter_id, state in states.items():
        step = state.get("step")
        if isinstance(step, torch.Tensor):
            if step.ndim != 0:
                raise RuntimeError("Optimizer step counter must be scalar")
            step = step.item()
        if (
            isinstance(step, bool)
            or not isinstance(step, (int, float))
            or not math.isfinite(step)
            or step <= 0
            or int(step) != step
        ):
            raise RuntimeError("Invalid optimizer step counter")
        counters[str(parameter_id)] = int(step)
    # A planned epoch count cannot establish that any update occurred. Adam
    # counters establish the work performed by every initialized parameter.
    actual_steps = set(counters.values())
    if actual_steps != {expected_steps}:
        raise RuntimeError(
            f"Optimizer checkpoint counters {sorted(actual_steps)} "
            f"do not match the expected {expected_steps} updates"
        )
    return {
        "steps": next(iter(actual_steps)),
        "expected_steps": expected_steps,
        "initialized_parameter_states": len(counters),
        "parameter_steps": counters,
        "uninitialized_parameter_ids": [
            parameter_id for parameter_id in parameter_ids if parameter_id not in states
        ],
    }


def make_frames(magnetic):
    generator = torch.Generator().manual_seed(1234)
    frames = []
    base = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.6]],
        dtype=torch.float64,
    )
    for _ in range(16):
        positions = (
            base + 0.07 * torch.randn(4, 3, generator=generator, dtype=torch.float64)
        ).requires_grad_()
        spin = torch.randn(4, 3, generator=generator, dtype=torch.float64)
        spin = spin / spin.norm(dim=-1, keepdim=True)
        spin[-1] = 0.0
        spin.requires_grad_()
        energy = positions.sum() * 0.0
        for i in range(4):
            for j in range(i + 1, 4):
                displacement = positions[j] - positions[i]
                distance = displacement.norm()
                energy = energy + 0.15 * (distance - 1.6) ** 2
                if magnetic:
                    direction = displacement / distance
                    energy = energy - 0.02 * spin[i].dot(spin[j]) * torch.exp(-distance)
                    energy = energy + 0.01 * (
                        (spin[i].dot(direction)) ** 2 + (spin[j].dot(direction)) ** 2
                    )
        inputs = (positions, spin) if magnetic else (positions,)
        gradients = torch.autograd.grad(energy, inputs)
        forces = -gradients[0]
        atoms = Atoms(
            "Fe3O", positions=positions.detach().numpy(), cell=np.eye(3) * 8, pbc=True
        )
        stress = -(positions.detach().T @ forces).numpy() / atoms.get_volume()
        atoms.calc = SinglePointCalculator(
            atoms, energy=energy.item(), forces=forces.numpy(), stress=stress
        )
        if magnetic:
            sf = -gradients[1]
            sf = sf - (sf * spin.detach()).sum(-1, keepdim=True) * spin.detach()
            sf[-1] = 0.0
            atoms.set_initial_magnetic_moments(spin.detach().numpy())
            atoms.set_array("spin_forces_vert", sf.numpy())
        frames.append(atoms)
    return frames


def run_case(output, soc, key, epochs, device):
    name = (
        "soc_spin"
        if key == "spin_forces_vert"
        else ("soc_force" if soc else "onlyr_force")
    )
    template = (
        REPO
        / "example"
        / ("07_allegro_soc_e2ip.yaml" if soc else "06_allegro_e2ip.yaml")
    )
    config = yaml.safe_load(template.read_text())
    # Scoped trunk/head settings override generic widths. Keep empty lists,
    # which intentionally select linear or identity embedding paths.
    config.update(
        {
            setting: [8]
            for setting, dimensions in config.items()
            if setting.endswith("_mlp_latent_dimensions") and dimensions
        }
    )
    frames = make_frames(soc)
    dataset_path = output / (name + ".extxyz")
    write(dataset_path, frames, format="extxyz")
    config.update(
        root=str(output / "runs"),
        run_name=name,
        dataset_file_name=str(dataset_path),
        dataset_root=str(output / (name + "_data")),
        chemical_symbol_to_type={"Fe": 0, "O": 1},
        n_train=12,
        n_val=4,
        r_max=4.0,
        num_layers=1,
        env_embed_multiplicity=2,
        two_body_latent_mlp_latent_dimensions=[8],
        latent_mlp_latent_dimensions=[8],
        edge_eng_mlp_latent_dimensions=[8],
        edge_onlyr_eng_mlp_latent_dimensions=[8],
        edge_biquad_coeff_mlp_latent_dimensions=[8],
        edge_eng_soc_on_mlp_latent_dimensions=[8],
        e2ip_latent_channels=2,
        e2ip_mlp_latent_dimensions=[8],
        evidential_key=key,
        batch_size=2,
        validation_batch_size=2,
        max_epochs=epochs,
        use_ema=False,
        learning_rate=0.003,
        max_gradient_norm=10.0,
        device=device,
    )
    config_path = output / (name + ".yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    env = dict(
        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NEQUIP_NUM_TASKS="1"
    )
    log_path = output / (name + ".log")
    with log_path.open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(REPO / "tools/ci/source_python.py"),
                "spingnn-train",
                str(config_path),
            ],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"{name} training failed; see {log_path}")
    run = output / "runs" / name
    with (run / "metrics_epoch.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != epochs:
        raise RuntimeError(f"{name}: expected {epochs} epochs, found {len(rows)}")
    for row in rows:
        if not all(np.isfinite(float(value)) for value in row.values()):
            raise RuntimeError(f"{name}: nonfinite epoch metrics")
    with global_options_context(config):
        model, saved = Trainer.load_model_from_training_session(
            run, model_name="best_model.pth", device=device
        )
        model.eval()
        sample = AtomicData.from_ase(frames[-1], r_max=4.0)
        sample = TypeMapper(chemical_symbol_to_type=config["chemical_symbol_to_type"])(
            sample
        )
        sample = AtomicData.to_AtomicDataDict(sample.to(device))
        out = model(dict(sample))
        prefix = "sf" if key == "spin_forces_vert" else "f"
        for field in (
            key,
            prefix + "_sigma",
            prefix + "_Uepi",
            prefix + "_uncertainty",
        ):
            if not torch.isfinite(out[field]).all():
                raise RuntimeError(f"{name}: reloaded {field} is nonfinite")
        checkpoint_path = run / "trainer.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_steps = epochs * math.ceil(config["n_train"] / config["batch_size"])
        optimizer_evidence = optimizer_checkpoint_evidence(checkpoint, expected_steps)
        if checkpoint.get("progress", {}).get("iepoch") != epochs:
            raise RuntimeError(
                f"{name}: checkpoint does not record all completed epochs"
            )
        if not saved["use_e2IP_loss"] or not saved["validation_use_e2IP_loss"]:
            raise RuntimeError(f"{name}: evidential loss flags were not serialized")
        report = dict(
            name=name,
            epochs=epochs,
            optimizer_steps=optimizer_evidence["steps"],
            optimizer_evidence=optimizer_evidence,
            optimizer_checkpoint={
                "path": str(checkpoint_path),
                "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            },
            device=device,
            first_epoch=rows[0],
            last_epoch=rows[-1],
            checkpoint_reload="PASS",
            checkpoint_reload_model="best_model.pth",
            validation_scope="training_and_best_checkpoint_reload",
            resume_tested=False,
            uncertainty=out[prefix + "_uncertainty"].detach().cpu().tolist(),
        )
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.epochs < 2:
        parser.error("--epochs must be at least 2")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = [
        run_case(output, soc, key, args.epochs, args.device)
        for soc, key in (
            (False, "forces"),
            (True, "spin_forces_vert"),
            (True, "forces"),
        )
    ]
    (output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
