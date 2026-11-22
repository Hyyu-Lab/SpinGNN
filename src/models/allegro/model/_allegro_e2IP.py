"""Structure-only Allegro with equivariant force evidence."""
from typing import Optional
from nequip.data import AtomicDataset
from ._allegro import Allegro
from ._e2ip import add_evidence_branch, finish_evidence, validate_e2ip


def Allegro_e2IP(config, initialize: bool, dataset: Optional[AtomicDataset] = None):
    validate_e2ip(config, soc=False)
    config["allegro_e2IP_out_field"] = "f_edge_vectors"
    model = Allegro(config, initialize, dataset)
    add_evidence_branch(model, config, "allegro", "f")
    return finish_evidence(model, config, "f")
