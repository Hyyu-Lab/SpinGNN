"""First-party PhysMuon, Muon, and HybridMuon adapters for the v1 YAML trainer.

Strictly one-way dependency (``nequip -> spingnn2.v2.optim``) so importing
``spingnn2.v2`` never imports ``e3nn`` or ``nequip``.
"""

from spingnn2.v2.optim import (
    PHYSMUON_PRESETS,
    HybridMuon,
    Muon,
    PhysMuon,
    WSDCosineScheduler,
    age,
    beta_for_age,
    fast_weight_for_age,
    interatomic_potential,
    interatomic_potential_gated,
    kernel_age,
    language_model,
    paper_bm,
    switch_step,
)

__all__ = [
    "HybridMuon",
    "Muon",
    "PHYSMUON_PRESETS",
    "PhysMuon",
    "WSDCosineScheduler",
    "age",
    "beta_for_age",
    "fast_weight_for_age",
    "interatomic_potential",
    "interatomic_potential_gated",
    "kernel_age",
    "language_model",
    "paper_bm",
    "switch_step",
]
