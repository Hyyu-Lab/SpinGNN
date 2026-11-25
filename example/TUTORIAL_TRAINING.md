# SpinGNN training tutorial

This tutorial trains and validates a magnetic potential from a source
checkout. Set up Python, PyTorch, and T-e3nn with the
[CUDA 12.8 installation guide](../docs/INSTALL_CUDA128.md).

## Quick start

Run from the repository root. First confirm that Python resolves both SpinGNN
packages and the spin-aware e3nn fork from the intended environment:

```bash
python tools/ci/source_python.py -c 'import sys; exec(sys.stdin.read())' <<'PY'
import allegro
import e3nn
import nequip
import torch

assert e3nn.__version__ == "0.5.0+spin"
assert tuple(map(int, torch.__version__.split("+", 1)[0].split(".")[:2])) >= (2, 11)
print("nequip", nequip.__file__)
print("allegro", allegro.__file__)
print("torch", torch.__version__)
PY
```

Choose the model contract before editing data or losses:

| Starting config | Contract | Compile default |
| --- | --- | --- |
| `01_structure_only.yaml` | Energy, force, stress | Eager; current output wrapper is unsupported by full-FX training |
| `02_spin_nosoc.yaml` | Rotation-invariant spin model | `full_fx` enabled |
| `03_spin_soc.yaml` | SOC magnetic model | `full_fx` enabled |
| `04_born_effective_charge.yaml` | Energy, force, stress, flattened `(N,9)` BEC | Eager; current output wrapper is unsupported by full-FX training |
| `05_spin_soc_production.yaml` | Larger SOC model with production training controls | `full_fx` enabled |
| [`06_allegro_e2ip.yaml`](06_allegro_e2ip.yaml) | Structure-only model with evidential force uncertainty | Eager |
| [`07_allegro_soc_e2ip.yaml`](07_allegro_soc_e2ip.yaml) | SOC model with evidential spin-force uncertainty | Eager |

See the [e2SLP guide](E2IP.md) for uncertainty training and trajectory scoring.

An ASE extxyz magnetic frame needs `energy`, `forces`, optional `stress`, and
per-atom `initial_magmoms`. Use `spin_forces_vert` only for the perpendicular
OSZICAR constrained-moment quantity; do not rename it to full `spin_forces`.
Remove any absent target from both `loss_coeffs` and `metrics_components`.

Validate the public YAML syntax and complete model builders without requiring
the placeholder datasets:

```bash
python tools/ci/source_python.py example/validate_configs.py
```

For a new SOC study, copy the production starting point, then replace the data
path, species map, split sizes, cutoff, loss weights, output directory, and run
name. Reserve a held-out test set outside `n_train` and `n_val`.

```bash
cp example/05_spin_soc_production.yaml /tmp/spingnn-soc-production.yaml
```

Replace the dataset placeholder in the copied configuration before training.

```bash
python tools/ci/source_python.py spingnn-train /tmp/spingnn-soc-production.yaml
```

A successful run writes under `root/run_name`. Expect at least `config.yaml`,
`trainer.pth`, `metrics_epoch.csv`, and a nonempty `best_model.pth`. The first
compiled batch is slow because it includes graph expansion and Inductor
compilation; steady-state timing starts only after this warmup.

## Recommended configuration

The modern training defaults and magnetic examples use:

```yaml
e3nn_optimization_defaults:
  jit_script_fx: false
spingnn_compile:
  enabled: true
  scope: full_fx
  dynamic: true
  inductor_options: {}
float32_matmul_precision: high
allow_tf32: true
train_autocast:
  enabled: false
  dtype: bfloat16
```

`full_fx` expands energy and its force, spin-force, stress, and virial
derivatives into one differentiable FX graph. It avoids the double-forward
gradient bridge used by the legacy `energy_core` scope. On one H100, the
19-step steady-state medians were:

| Model | Eager | `full_fx` | Speedup | Maximum loss-curve relative error |
| --- | ---: | ---: | ---: | ---: |
| Allegro | 0.092098 s/step | 0.003936 s/step | 23.4x | 3.65e-7 |

These are reference measurements, not guaranteed throughput for another
architecture, batch distribution, or GPU. `dynamic: true` lets atom/edge/frame
axes vary, but new shapes or guards can still trigger bounded recompilation.

H100 TF32 calibration measured force relative errors of 2.93e-3 to 3.92e-3
and energy errors of 2.227e-4 to 2.264e-4 eV/atom. The test limits were 8e-3
and 5e-4 eV/atom, respectively. For a tighter workload-specific tolerance,
set `float32_matmul_precision: highest` and `allow_tf32: false`, then rerun
validation.

Keep `train_autocast.enabled: false` when supervising forces, stress, or spin
forces. H100 BF16 force relative error reached 7.755e-2 against a tolerance of
1e-2. Use BF16 only for separately validated energy-only experiments.

## Acceleration validation

The no-data validator confirms that a magnetic model has one compile target:

```bash
python tools/ci/source_python.py example/validate_configs.py example/02_spin_nosoc.yaml
```

After an in-process train/eval forward, inspect the live model with:

```python
from nequip.utils.compile_utils import get_compile_stats

stats = get_compile_stats(trainer.model)
print(stats)
assert stats["targets"] == 1
assert stats["scopes"] == ["full_fx"]
assert stats["full_fx_expand_count"] >= 1
assert stats["eager_grad_bridge_calls"] == 0
```

`compile_count` and `cache_modes` show whether train/eval graphs were created.
Compare `metrics_epoch.csv` wall times only after the compile-bearing first
epoch, using the same source, seed, data order, batch size, hardware, and loss
settings. Compare against an otherwise identical config with only the eager
fallback values changed.

## Precision validation

Run the CPU full-FX output and parameter-gradient tests from the checkout
(setting one OpenMP/MKL thread to match CI):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SPINGNN_TEST_COMPILE=1 \
  python tools/ci/source_python.py -m pytest \
    src/models/tests/model/test_allegro_spin_nosoc.py \
    src/models/tests/model/test_allegro_spin_soc.py -q -p no:cacheprovider
```

The tests require physical-output relative error at most 1e-5,
parameter-gradient relative error at most 1e-4, and zero eager bridge calls.
CPU results do not validate TF32; repeat the precision ladder on the target GPU
before accepting a tighter workload-specific tolerance.

## Common problems

### The first step is much slower

This is expected compilation overhead. Measure steady state after the first
step/epoch, and keep eager mode for very short debugging runs where compilation
cannot amortize.

### The model recompiles repeatedly

Keep `dynamic: true`, avoid changing dictionary keys between batches, and use
stable batching policies. Inspect `compile_count`, `cache_modes`, trace sizes,
and `TORCH_LOGS=recompiles` before changing Inductor options. A new model mode
creates a separate train or eval cache by design.

### Compilation uses too much memory

Lower `batch_size` first. Graph expansion and compilation have a temporary
memory peak in addition to model activations. Do not enable aggressive
autotuning or CUDA graphs: the checked-in conservative Inductor options keep
them disabled.

### Compilation is unsupported or fails

Structure-only and BEC output wrappers are intentionally eager. For a magnetic
model, confirm PyTorch >=2.11 and `jit_script_fx: false`. To recover immediately,
use the exact eager fp32 block:

```yaml
spingnn_compile:
  enabled: false
  scope: full_fx
  dynamic: true
  inductor_options: {}
float32_matmul_precision: highest
allow_tf32: false
train_autocast:
  enabled: false
  dtype: bfloat16
```

Rerun the same seed/data-order comparison after rollback. Do not weaken parity
tolerances to make a failed optimization appear acceptable.
