"""Read accepted-state LAMMPS custom dumps into validated GPST frames."""

from pathlib import Path

import numpy as np

from .core import from_lammps_fm, validate_frame

FIELDS = ["id", "x", "y", "z"] + [f"c_gp[{i}]" for i in range(1, 8)]


def read_dump(path, *, hbar, accept_every, force_contract_confirmed):
    """Parse an orthogonal accepted-state dump with unit spins and conservative fm."""
    if force_contract_confirmed is not True:
        raise ValueError("FORCE_CONTRACT_MISSING")
    if accept_every <= 0:
        raise ValueError("ACCEPT_INTERVAL_REQUIRED")
    frames, boxes, steps = [], [], []
    ids = mask = pbc = None
    with Path(path).open() as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            if line.strip() != "ITEM: TIMESTEP":
                raise ValueError("DUMP_TIMESTEP_HEADER")
            step = int(stream.readline())
            if step % accept_every:
                raise ValueError("DUMP_NOT_ACCEPTED_STATE")
            if stream.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("DUMP_ATOM_HEADER")
            n = int(stream.readline())
            flags = stream.readline().split()
            if flags[:3] != ["ITEM:", "BOX", "BOUNDS"] or len(flags) != 6:
                raise ValueError("DUMP_REQUIRES_ORTHOGONAL_BOX")
            current_pbc = np.array([x == "pp" for x in flags[3:]], bool)
            bounds = np.array(
                [list(map(float, stream.readline().split())) for _ in range(3)]
            )
            if bounds.shape != (3, 2):
                raise ValueError("DUMP_REQUIRES_ORTHOGONAL_BOX")
            if stream.readline().split() != ["ITEM:", "ATOMS", *FIELDS]:
                raise ValueError("DUMP_FIELD_CONTRACT")
            raw = np.array(
                [list(map(float, stream.readline().split())) for _ in range(n)]
            )
            if raw.shape != (n, len(FIELDS)) or not np.isfinite(raw).all():
                raise ValueError("DUMP_ROW_CONTRACT")
            raw = raw[np.argsort(raw[:, 0])]
            if np.any(raw[:, 0] != raw[:, 0].astype(np.int64)):
                raise ValueError("ID_MISMATCH")
            current_ids = raw[:, 0].astype(np.int64)
            current_mask = raw[:, 7] > 0
            if ids is None:
                ids, mask, pbc = current_ids, current_mask, current_pbc
            if not (
                np.array_equal(ids, current_ids)
                and np.array_equal(mask, current_mask)
                and np.array_equal(pbc, current_pbc)
            ):
                raise ValueError("ID_MISMATCH")
            steps.append(step)
            boxes.append(np.diag(bounds[:, 1] - bounds[:, 0]))
            frames.append(raw)
    if not frames:
        raise ValueError("EMPTY_DUMP")
    data = np.array(frames)
    return validate_frame(
        atom_ids=ids,
        magnetic_mask=mask,
        positions=data[:, :, 1:4],
        cell=np.array(boxes),
        pbc=pbc,
        spins=data[:, mask, 4:7],
        spin_forces=from_lammps_fm(data[:, mask, 8:11], hbar),
        frame_index=np.array(steps, dtype=np.int64),
        force_contract_confirmed=True,
    )
