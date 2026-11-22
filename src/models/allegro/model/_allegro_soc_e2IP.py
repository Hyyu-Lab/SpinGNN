"""SOC Allegro with force or tangent spin-force evidence."""
from typing import Optional
from nequip.data import AtomicDataset
from ._allegro_soc import Allegro_soc
from ._e2ip import add_evidence_branch, finish_evidence, validate_e2ip


def Allegro_soc_e2IP(config, initialize: bool, dataset: Optional[AtomicDataset] = None):
    validate_e2ip(config, soc=True)
    force = config.get("evidential_key", "spin_forces_vert") == "forces"
    prefix = "f" if force else "sf"
    if force and not config.get("lonly_soc_layer", False):
        config["allegro_onlyr_e2IP_out_field"] = "f_edge_vectors"
        config["allegro_soc_on_e2IP_out_field"] = "socf_edge_vectors"
    else:
        config["allegro_soc_on_e2IP_out_field"] = prefix + "_edge_vectors"
    model = Allegro_soc(config, initialize, dataset)
    if force and not config.get("lonly_soc_layer", False):
        add_evidence_branch(model, config, "allegro_onlyr", "f")
        add_evidence_branch(model, config, "allegro_soc_on", "socf")
    else:
        add_evidence_branch(model, config, "allegro_soc_on", prefix)
    return finish_evidence(model, config, prefix)
