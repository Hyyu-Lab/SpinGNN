# Deploy a model with PT2

Packaged AOTInductor (`.pt2`) is the recommended GPU inference path for the
`spin/allegro_spin` and `spin/nequip_spin` LAMMPS pair styles. The non-spin
`allegro` style also supports PT2 for structure/BEC models through the ordered
`pos edge_index atom_types` inputs. TorchScript (`.pth`) is an alternative
for every PT2-capable style and is required by non-spin
`nequip`.

## Artifact contract

Select the training session and the deployment device. From the repository root, use the intended
Python environment and the [source launcher](SOURCE_CHECKOUT.md) so both
NequIP and Allegro resolve to this checkout:

```bash
python tools/ci/source_python.py nequip-deploy build-pt2 --device cuda \
  --train-dir results/spin_nosoc/example deployed-model.pt2
```

`--device auto` selects CUDA when it is available and otherwise CPU. The
command writes `deployed-model.pt2` and `deployed-model.pt2.meta`. They are one
artifact set and must be moved, archived, and published together.

A `.pt2` package is tied to the checkpoint it was exported from. Model weights
and learned rescale tensors (`scale_by` and `shift_by`) are serialized into the
package. Export a new package for a different checkpoint; never reuse an old
package with another checkpoint.

The sidecar binds to the package with `package_sha256`. Before constructing the
AOTI loader, LAMMPS verifies that package SHA-256, reconciles
`provided_outputs` with the ordered `output_signature`, rejects duplicate
names, and checks target device. The first model call also verifies the exact
output count and each consumed tensor's float32 dtype, rank, and shape before
constructing an accessor.

CUDA packages additionally record `cuda_compute_capability` and
`torch_cuda_version`. The pair rejects a package exported for a different GPU
architecture or LibTorch CUDA ABI. Export on the target architecture with the
same CUDA/PyTorch environment used by LAMMPS.

## Learned scaling and dynamic shapes

Normal checkpoints wrapped by `RescaleEnergyEtc` are supported. The exporter
preserves nested outer `RescaleOutput` layers and replays runtime scale/shift
operations after derivative outputs, from the inner wrapper outwards. The CLI
must report `PT2 learned scaling: included (N layer(s))` for a checkpoint known
to use learned scaling. Unsupported output wrappers fail explicitly and are
never silently discarded.

The exported ABI has dynamic atom and edge axes. The declared range is one or
more atoms and at least two directed edges, capped at 2**40 on both axes
(`_DYNAMIC_DIM_MAX` in `nequip/utils/pt2_deploy.py`). The finite cap lets
Inductor select safe indexing when `n_edges * features_per_edge` exceeds
the 32-bit range. Maintained tests exercise
2, 4, 24, and 25 edges plus a real two-atom LAMMPS run. LAMMPS supplies an
edge list already filtered to the model cutoff, and export rejects decreasing
per-layer cutoffs that would invalidate that contract.

## Verify an export

Use the [deployment tests](../tests_lammps/README.md) and the per-device
command below to check a fresh export:

1. Compare TorchScript and PT2 in Python on identical graph tensors, including
   energy, force, spin force, virial, atom virial, and model-specific outputs.
2. Build LAMMPS from the matching sources and compare `.pth` and
   `.pt2` through the appropriate pair style, including small-edge inputs.
3. Exercise package/sidecar mismatch, output-signature mutation, wrong device,
   and wrong CUDA architecture as fail-fast negative tests.
4. For nonzero global energy shifts, compare Python total energy with serial,
   MPI, host, and Kokkos-SERIAL LAMMPS paths.

Record the source revision and the hashes of the checkpoint, package, sidecar,
and LAMMPS executable when comparing runs. Deterministic CUDA comparisons use:

```bash
export SPINGNN_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### Per-device correctness checks

`tools/ci/gpu_pt2_threeway.py` exports TorchScript, CPU-target PT2, and
CUDA-target PT2 from the NoSOC, SOC, and BEC fixtures under
`tests_lammps/fixtures/configs` using deterministic weights. Run it on the
target GPU with a serial CUDA LAMMPS build and an empty output directory.

```bash
python tools/ci/source_python.py tools/ci/gpu_pt2_threeway.py --gpu h100 --devices cpu cuda \
  --lammps-bin "$PWD/build/lammps-serial/lmp" \
  --output-dir "$PWD/validation/pt2-correctness-h100"
```

Select `--gpu 4090` or `--gpu 5090` on those devices. The CUDA sidecar binds
the export to the device capability and Torch CUDA ABI. The script compares
the following routes and writes `threeway.json` in the output directory.

1. Python TorchScript against Python CPU PT2.
2. Python TorchScript against Python CUDA PT2.
3. LAMMPS TorchScript and LAMMPS CUDA PT2 against the same physical reference.
4. LAMMPS CUDA PT2 against LAMMPS TorchScript.

The same GPU run verifies the loader diagnostics for a CPU-target package and
for a package with an incompatible CUDA capability. A CPU-only smoke checks
the Python TorchScript and CPU PT2 route. The result marks the CUDA and
LAMMPS routes `NOT_RUN`.

```bash
python tools/ci/source_python.py tools/ci/gpu_pt2_threeway.py --line bec --devices cpu \
  --output-dir "$PWD/validation/pt2-correctness-cpu"
```

## Boundaries

- PT2 loading is implemented by `allegro`, `spin/allegro_spin`, and
  `spin/nequip_spin`; non-spin `nequip` remains TorchScript-only.
- Validate the exported potential with its target pair style and GPU
  architecture before running a production simulation.
- A package targets one device class, CUDA compute capability, and Torch CUDA
  ABI; cross-architecture portability is not claimed.
- The package and sidecar are written sequentially rather than with a
  filesystem-wide atomic transaction. A failed export must be removed rather
  than partially reused.
- For Kokkos CUDA, see the [LAMMPS build options](../lammps/README.md#opt-in-kokkos-cuda).
  The host/Kokkos parity tests use deterministic mode on one device. Use
  serial CUDA PT2 for the standard GPU inference path.

