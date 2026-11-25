# SpinGNN2 compact contract v1

Export an eager training checkpoint from its source checkout. A training directory supplies
`best_model.pth` and `config.yaml`. A different state dictionary can use the same config.

```sh
python tools/ci/source_python.py spingnn2-export runs/example -o models/example
python tools/ci/source_python.py spingnn2-export runs/example/last_model.pth \
    --config runs/example/config.yaml -o models/last
```

The installed `spingnn2-export` command accepts the same arguments and
`--source-root PATH`. Export uses the training dependencies. Install
`src/spingnn2` on an inference host to run the reference evaluator with NumPy and SciPy.

```python
from spingnn2.reference import ReferenceModel

model = ReferenceModel.load("models/example.sp2.json")
result = model.evaluate(graph, tile_atoms=256, radial_mode="analytic")

cache = model.evaluate_stage1_lattice(graph, tile_atoms=256, radial_mode="spline")
updated = model.evaluate_stage2_spin_cached(cache, new_spins)
```

## Graph and output conventions

Supply `pos[N,3]`, zero-based `atom_types[N]` (or `[N,1]`), and directed
`edge_index[2,E]` in the training type order. Row zero indexes centers.
Magnetic models also require Cartesian `spin[N,3]`. A periodic graph supplies a
row-vector `cell[3,3]`, `edge_cell_shift[E,3]` in lattice coordinates, and three true
`pbc` flags. Multiple structures can share a call with dense frame indices in
`batch[N]` and one cell per frame. Edges can arrive in any order. The caller supplies
the full neighborhood of each center, including periodic images.

The result uses the eager field names and shapes. `forces` equals the negative
position derivative. The SpinGNN convention multiplies the negative spin derivative
by the squared spin length for `spin_forces`. The per-center `atom_virial`
retains its antisymmetric part. The frame `virial` is symmetric and
`stress = -virial / abs(det(cell))`. Without a cell, stress is an empty array.
DREAM writes nine row-major Cartesian BEC components per atom and per-frame mean.
Hard ASR subtracts each frame's componentwise mean. Soft and none retain the
physical predictions. Evidential outputs include nodewise parameters and covariances,
the frame sum of epistemic covariances, and mean trace over valid atoms.

Ordinary evaluation releases intermediate features after each tile's reverse pass.
The lattice cache explicitly retains geometry, lattice readouts and the intermediates
needed for their force derivatives across spin updates. Build a new cache after
changes to coordinates, cell, neighbor list, species, or model.

## Wire representation

The JSON object identifies `format` (`"spingnn2.contract.v1"`), `format_version` (`1`), a companion binary filename,
and its SHA-256. `arrays` describes immutable binary arrays by little-endian dtype,
shape, and byte offset. Arrays have C row-major layout and eight-byte-aligned starts.
Inside `model`, `{"$array": i}` refers to the zero-based descriptor at index `i`.

`model.layers` lists the source graph operations in execution order with field names
and parameters. Each trunk stores resolved tensor-product paths as six integer columns
`(i_in1,i_in2,i_out,m1,m2,m3)` and a parallel `scaled_w3j` column. The first three
indices select irreps. The last three select zero-based components within those
irreps in the source real basis. The exported irrep offsets map them to flat
feature indices. Linear records contain per-irrep matrix blocks, normalization
and component slices. Weighted channels use the serialized `w_index`. Cartesian
heads store their actual change-of-basis matrices. MLP records store folded
weight matrices and post-multiply activations.

Radial records contain trained basis parameters, normalization statistics when
applicable, and cutoff parameters. Each Hermite table contains values and first
derivatives at uniformly spaced knots, its domain, and calibration errors.
Spin dot products outside the tabulated interval use the exported analytical
Fourier expression. Outer inference rescales are applied in their nested
execution order, including squared scaling of evidential covariance fields.

## Native inference

Build one shared library per backend and CUDA architecture. The build pins
Kokkos and the JSON reader, and requires CMake 3.24+, a C++17 compiler, OpenSSL
development headers, and either a CPU CBLAS implementation or the CUDA toolkit.
Cluster builds and capacity runs belong on Slurm allocations.

```sh
python tools/native/build_native.py --backend cpu
python tools/native/build_native.py --backend cuda --arch all-gpu --cuda-root /data/app/cuda/12.8
python tools/native/build_native.py --backend cuda --arch sm120 --cuda-root /data/app/cuda/12.8
```

`NativeModel` accepts the same graphs and returns the same field shapes as
`ReferenceModel`. Select the library using `library=...` or
`SPINGNN2_LIBRARY`. When unset, the CUDA loader prefers
`validation/native/build-cuda-all-gpu/libspingnn2.so` or detects the active
device compute capability via `libcuda.so.1` (override with `SPINGNN2_ARCH=sm89`).

```python
from spingnn2.native import NativeModel

with NativeModel.load("model.sp2.json", backend="cuda", precision="float32",
                      tile_atoms=256, radial_mode="spline") as model:
    outputs = model.evaluate(graph)
    print(model.info, model.diagnostics)
    with model.evaluate_stage1_lattice(graph) as cache:
        midpoint = model.evaluate_stage2_spin_cached(cache, midpoint_spins)
```

CUDA uploads atom and cell arrays once, streams CSR edge slices through
bounded pinned buffers, and downloads one packed array of complete results.
Device scratch grows to fit the largest tile and is reused between
evaluations. Each sparse tensor-product term indexes its center's environment
directly. Cache creation records stage-1 intermediates, including those required
to propagate updated spin-dependent forces back to coordinates. On CUDA,
stage 2 reads resident lattice tapes in place. The default residency budget is
the smaller of 4 GiB and one quarter of currently free GPU memory. Tiles that
exceed the tape budget can retain their forward tensors for spin-only calls.
Full evaluation recomputes those tapes on the GPU. Set
`cache_device_limit_bytes=0` for host-only snapshots or pass a positive byte
limit to tune residency. Diagnostics report device-resident
cache, host snapshot payload and reusable tile workspace separately. Use `profile=True`
to measure synchronized forward geometry, MLP, tensor-product/linear and reverse
stages. Graph preparation, result assembly and host transfers also contribute
to end-to-end time.

The C header `libspingnn2/include/spingnn2_c_api.h` defines opaque handles,
precision and options, host array layouts, ownership, and error handling. The
versioned ABI uses explicit dimensions, row-major arrays, and zero-based indices.
Close or free handles after outstanding calls finish. Python copies output
arrays before freeing the native result, so they survive subsequent evaluations.

Install the Python package with the `ase` extra to use the calculator.

```python
from spingnn2.calculator import SpinGNN2Calculator

atoms.calc = SpinGNN2Calculator("model.sp2.json", backend="cpu", radial_mode="spline")
energy, forces = atoms.get_potential_energy(), atoms.get_forces()
```

Supply `type_names` in training order when the contract lacks chemical names.
Magnetic models accept Cartesian `atoms.arrays["spin"]` or ASE
`initial_magmoms`. Scalar magnetic moments point along z. The calculator returns
stress in ASE Voigt order for fully periodic structures, BECs as
`[N,3,3]`, and evidential outputs through `atoms.calc.get_property(name, atoms)`.

## Model-specialized execution

The generic mode is the default and uses the portable Kokkos program. Python
`auto` prepares a cache entry on the first use and loads it. For a deployment
without a compiler, prepare once on a build node and use `direct` with the
returned manifest. Never load a shared library or cubin supplied by an
untrusted producer. The cache checks integrity and rejects writable shared
entries. Its hashes do not authenticate an external producer.

```sh
python tools/ci/source_python.py spingnn2-prepare-jit-artifact \
    --model model.sp2.json --backend cuda --arch sm120 --precision float32
```

```python
from spingnn2.native import NativeModel

with NativeModel.load("model.sp2.json", backend="cuda", precision="float32",
                      execution_mode="direct", artifact=artifact_manifest,
                      execution_profile="speed", radial_mode="spline") as model:
    full = model.evaluate(graph)
    print(model.execution_plan)
    with model.evaluate_stage1_lattice(graph) as lattice:
        full_midpoint = model.evaluate_stage2_spin_cached(lattice, new_spins)
    with model.evaluate_stage1_lattice(graph, spin_only=True) as lattice:
        frozen = model.evaluate_stage2_spin_cached(lattice, new_spins, spin_only=True)
```

Frozen-lattice evaluation returns energy, magnetic forces and forward readouts.
It stops the energy adjoint at the cached lattice outputs. Set `spin_only=True`
at creation to store just the Stage-2 forward inputs. A full evaluation from
this cache recomputes the Stage-1 tape and returns all derivatives. A magnetic
model and a cache of the original coordinates and
neighbor list are required. Switching an existing model to a different artifact
requires recreating its cache.

Both `NativeModel` and `SpinGNN2Calculator` accept `execution_profile`.
When tile size is automatic, `capacity` caps the initial tile at 256 centers,
while `speed` allows 512, or 1,024 for CUDA FP32 DREAM. The planner can reduce the width using the largest
center degree, the exported channel widths and available memory. The estimate
is a sizing heuristic. `model.diagnostics` reports the measured workspace
after evaluation. An explicit `tile_atoms` overrides automatic sizing; passing
`None` requests the full graph in one tile.

CUDA models use FP32 GEMM inputs and FP32 accumulation by default.
`NativeModel(..., precision="float32", allow_tf32=True)` opts in to TF32
GEMMs. This can change forces and BECs. `model.execution_plan["math_mode"]`
records the active policy. A fixed-lattice cache retains the policy used at
creation for subsequent spin updates.
