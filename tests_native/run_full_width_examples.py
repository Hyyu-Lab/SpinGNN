"""Check the example architectures at their original widths against eager inference."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml

from nequip.model import model_from_config
from nequip.utils import Config
from nequip.utils._global_options import global_options_context
from spingnn2.export import export_model
from spingnn2.reference import ReferenceModel
from test_contract_oracle import CASES, EXAMPLES, REPO, _observables, _sample


def compare(device, directory):
    failures = []
    for precision, analytic_limit in (("float64", 1e-11), ("float32", 1e-5)):
        for case in CASES:
            if case.name not in (
                "nosoc-te3nn",
                "soc-u2",
                "born-soft",
                "e2ip-force",
                "soc-e2ip-force",
                "soc-e2ip-spin",
            ):
                continue
            config = yaml.safe_load(
                (REPO / "example" / EXAMPLES[case.family]).read_text()
            )
            # Stand in for dataset statistics while retaining the example's widths.
            config.update(
                avg_num_neighbors=4.25,
                global_rescale_scale=1.125,
                per_species_rescale_shifts=0.07,
                per_species_rescale_scales=0.85,
                default_dtype=precision,
                allow_tf32=False,
                float32_matmul_precision="highest",
                spingnn_compile={"enabled": False},
                train_autocast={"enabled": False},
            )
            if case.family == "soc-e2ip":
                config.update(case.overrides)
            graph = _sample(config)
            with global_options_context(config):
                torch.manual_seed(1793)
                model = (
                    model_from_config(Config.from_dict(config), initialize=True)
                    .eval()
                    .to(device)
                )
                inputs = {
                    key: torch.from_numpy(v.copy()).to(device)
                    for key, v in graph.items()
                }
                expected = model(inputs)
                manifest, _ = export_model(
                    model, directory / (case.name + "-" + precision)
                )
                oracle = ReferenceModel.load(manifest)
                modes = (
                    ("analytic", "spline") if precision == "float64" else ("analytic",)
                )
                for mode in modes:
                    actual = oracle.evaluate(graph, tile_atoms=3, radial_mode=mode)
                    limit = analytic_limit if mode == "analytic" else 1e-6
                    errors = {}
                    for field in _observables(case):
                        target = expected[field].detach().cpu().numpy()
                        assert actual[field].shape == target.shape
                        error = float(np.max(np.abs(actual[field] - target)))
                        errors[field] = error
                        if not error < limit:
                            failures.append((case.name, precision, mode, field, error))
                    print(
                        "EXAMPLE_RESULT",
                        json.dumps(
                            dict(
                                case=case.name, precision=precision, mode=mode, **errors
                            )
                        ),
                        flush=True,
                    )
    print("FAILED_GATES", json.dumps(failures), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    with TemporaryDirectory() as directory:
        raise SystemExit(compare(args.device, Path(directory)))
