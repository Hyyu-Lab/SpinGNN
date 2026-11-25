"""Command-line entrypoints for offline GPST moment extraction and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .api import analyze
from .core import compute_moments
from .dump_adapter import read_dump


def main_moments(argv: Sequence[str] | None = None) -> None:
    """Convert an accepted-state LAMMPS custom dump to sufficient GPST moments (.npz)."""
    parser = argparse.ArgumentParser(description=main_moments.__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--hbar", required=True, type=float, help="Native hbar in eV ps"
    )
    parser.add_argument(
        "--cutoff", required=True, type=float, help="Probe cutoff in Angstrom"
    )
    parser.add_argument("--accept-every", required=True, type=int)
    parser.add_argument("--force-contract-confirmed", action="store_true")
    args = parser.parse_args(argv)
    frame = read_dump(
        args.dump,
        hbar=args.hbar,
        accept_every=args.accept_every,
        force_contract_confirmed=args.force_contract_confirmed,
    )
    moments = compute_moments(frame, args.cutoff)
    np.savez(args.output, **moments)
    print(
        json.dumps(
            dict(
                frames=len(frame.frame_index),
                atoms=len(frame.atom_ids),
                output=str(args.output),
                force_conversion="f=hbar*fm",
            )
        )
    )


def main_infer(argv: Sequence[str] | None = None) -> None:
    """Run independent-chain GPST CF_GMM & Fieller inference from moment .npz records."""
    parser = argparse.ArgumentParser(description=main_infer.__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-ids", nargs="+", required=True)
    parser.add_argument("--kb", type=float, required=True, help="Native kB in eV/K")
    parser.add_argument(
        "--dt", type=float, required=True, help="Saved-frame interval in ps"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    records = []
    for path in args.records:
        with np.load(path, allow_pickle=False) as data:
            records.append({key: data[key] for key in data.files})
    result = analyze(records, args.seed_ids, kb=args.kb, dt=args.dt)

    def _serial(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    args.output.write_text(
        json.dumps(result, default=_serial, indent=2, allow_nan=False) + "\n"
    )
