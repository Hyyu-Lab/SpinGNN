#!/usr/bin/env python3
"""Parse and build the public example models without requiring user datasets."""

import argparse
from pathlib import Path
import runpy

import torch

runpy.run_path(str(Path(__file__).resolve().parents[1] / "_spingnn_source.py"))[
    "activate_source_tree"
]()

from nequip.data import AtomicDataDict  # noqa: E402
from nequip.model import model_from_config  # noqa: E402
from nequip.scripts.train import default_config  # noqa: E402
from nequip.utils import Config  # noqa: E402
from nequip.utils._global_options import global_options_context  # noqa: E402
from nequip.utils.compile_utils import (  # noqa: E402
    _capture_signature,
    _full_fx_cache_key,
    configure_model_for_compile,
    get_compile_stats,
)
from nequip.utils.fx_compat import (  # noqa: E402
    expand_training_graph,
    validate_training_batch,
)


def _validation_batch(num_types):
    return {
        AtomicDataDict.POSITIONS_KEY: torch.tensor(
            [[0.2, 0.3, 0.4], [1.4, 0.6, 0.5], [0.7, 1.6, 0.9]]
        ),
        AtomicDataDict.SPIN_KEY: torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
        ),
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: torch.zeros((6, 3)),
        AtomicDataDict.ATOM_TYPE_KEY: torch.arange(3, dtype=torch.long).remainder(
            num_types
        ),
        AtomicDataDict.BATCH_KEY: torch.zeros(3, dtype=torch.long),
        AtomicDataDict.NUM_BATCH_KEY: torch.tensor(1, dtype=torch.long),
        AtomicDataDict.CELL_KEY: torch.eye(3).reshape(1, 3, 3) * 8.0,
        AtomicDataDict.PBC_KEY: torch.ones((1, 3), dtype=torch.bool),
    }


def _install_eval_graph(model, sample):
    targets = [
        module
        for module in model.modules()
        if hasattr(module, "_spingnn_compile_state")
    ]
    if len(targets) != 1:
        raise RuntimeError(f"expected one full-FX target, found {len(targets)}")
    target = targets[0]
    sample, (n_frames, _, _) = validate_training_batch(target, sample)
    graph = expand_training_graph(target, sample, create_graph=False)
    state = target._spingnn_compile_state
    cache_key = _full_fx_cache_key(state, sample, "eval", n_frames=n_frames)
    # Exercise the expanded graph through the model without compiling machine
    # code on the CPU runner. Match the signature checked by the runtime cache.
    state.capture_signature = _capture_signature(target, sample)
    state.compiled_by_mode[cache_key] = graph
    state.input_signatures[cache_key] = tuple(graph.input_signature)
    state.output_signatures[cache_key] = tuple(graph.output_signature)
    state.trace_sizes[cache_key] = tuple(graph.trace_sizes)
    state.full_fx_expand_count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", type=Path)
    args = parser.parse_args()

    paths = args.configs or sorted(Path(__file__).parent.glob("*.yaml"))
    if not paths:
        raise RuntimeError("no example YAML files found")

    for path in paths:
        config = Config.from_file(str(path), defaults=default_config)
        # Dataset-derived neighbor statistics are unavailable in a public
        # source checkout. A finite placeholder exercises the complete builder.
        if config.get("avg_num_neighbors") == "auto":
            config["avg_num_neighbors"] = 1.0
        with global_options_context(config):
            model = model_from_config(config=config, initialize=False).eval()
            model = configure_model_for_compile(model, config)
            stats = get_compile_stats(model)
            compile_enabled = bool(config["spingnn_compile"]["enabled"])
            if compile_enabled and stats["targets"] != 1:
                raise RuntimeError(
                    f"{path}: compile enabled but no target was attached"
                )
            if not compile_enabled and stats["targets"] != 0:
                raise RuntimeError(
                    f"{path}: compile disabled but a target was attached"
                )
            if config["train_autocast"]["enabled"]:
                raise RuntimeError(
                    f"{path}: force-training autocast must stay disabled"
                )
            sample = _validation_batch(len(config["chemical_symbol_to_type"]))
            if compile_enabled:
                _install_eval_graph(model, sample)
            output = model(dict(sample))
            if AtomicDataDict.TOTAL_ENERGY_KEY not in output:
                raise RuntimeError(f"{path}: eval forward returned no total energy")
        print(
            f"EXAMPLE_FORWARD_CHECK path={path} result=PASS mode=eval "
            f"compile_targets={stats['targets']} scope={config['spingnn_compile']['scope']}"
        )


if __name__ == "__main__":
    main()
