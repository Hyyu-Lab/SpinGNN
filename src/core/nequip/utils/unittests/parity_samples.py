"""Nondegenerate fixed samples for precision and derivative comparisons."""

import torch

from nequip.data import AtomicData, AtomicDataDict


def deterministic_spin_batch(atomic_batch_spin, device):
    """Use H2 with two reproducible noncollinear unit spins.

    The default H2 frame has only one magnetic atom. Pair-only delta heads then
    have identically zero energy, and anisotropy-only heads can have nearly zero
    forces. Neither is a useful reference for relative precision measurements."""
    # One neighbor per center avoids multi-edge atomic reduction ordering in
    # the unchanged bitwise eager selfcheck, including its CUDA lane.
    sample = atomic_batch_spin.get_example(1).clone()
    positions = sample[AtomicDataDict.POSITIONS_KEY]
    generator = torch.Generator().manual_seed(193)
    spins = torch.randn(positions.shape, generator=generator, dtype=torch.float64)
    spins = spins / torch.linalg.vector_norm(spins, dim=-1, keepdim=True)
    sample[AtomicDataDict.SPIN_KEY] = spins.to(dtype=positions.dtype)
    data = AtomicData.to_AtomicDataDict(sample.to(device))
    return AtomicDataDict.with_batch(data)
