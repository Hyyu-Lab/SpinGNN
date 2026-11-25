# libspingnn2

The C ABI consumes immutable `.sp2.json` contract manifests (`spingnn2.contract.v1`) and verifies the companion SHA-256
before decoding any weights. Configure one build tree per execution space and
GPU architecture. The library exports ABI version 1 with opaque model, cache
and result handles. See [the public header](include/spingnn2_c_api.h) and
[the Python build and inference examples](../src/spingnn2/README.md).

## Build the native runtime

The C ABI library uses C++17, Kokkos 4.7.02, nlohmann/json 3.11.3, OpenSSL
libcrypto, and CBLAS or cuBLAS. The builder pins the Kokkos and JSON archives
by SHA-256. Populate the default cache from a host with internet access before
building on an offline compute node.

```sh
python tools/native/build_native.py --prepare-dependencies
```

The default source cache is `validation/native/deps`. Use `--dependency-dir` to
select a different shared cache or provide local sources with
`--kokkos-source` and `--json-source`.

### Cluster builds

Run the commands below on a Slurm allocation. They use the cluster GCC 10.5.0
compiler by default. The CUDA jobs use the host compiler through Kokkos
`nvcc_wrapper`. Set `--cxx`, `--mkl-root`, `--cuda-root`, or `--cmake`
for a different installation.

```sh
srun -p xiang -c 8 --mem=16G python tools/native/build_native.py --backend cpu
srun -p xiang -c 16 --mem=32G python tools/native/build_native.py \
    --backend cuda --arch all-gpu --cuda-root /data/app/cuda/12.8
# Or target a single architecture (sm70|sm80|sm86|sm89|sm90|sm120):
srun -p xiang -c 8 --mem=24G python tools/native/build_native.py \
    --backend cuda --arch sm120 --cuda-root /data/app/cuda/12.8
```

OpenMP and CPU BLAS run on the compute node. Compiling CUDA (including
multi-architecture `all-gpu` fat binaries covering `sm_70`, `sm_80`, `sm_86`,
`sm_89`, `sm_90`, and `sm_120` plus `compute_120` PTX) needs only the toolkit and runs
cleanly on CPU allocations. Execution needs a matching GPU and driver.

The build runs out of tree and prints the shared library path. Default build
directories are `validation/native/build-cpu`, `validation/native/build-cuda-all-gpu`,
and `validation/native/build-cuda-sm*`. Set `--install-prefix` to install the
header, versioned shared library and license notices. Set
`SPINGNN2_LIBRARY` to select the shared library from Python, or pass the
path explicitly to `NativeModel.load`.

A CPU build on another system uses system BLAS and requires a CBLAS header and
library, or accepts oneMKL through `--mkl-root`. Builds made with the GNU C++
toolchain keep their C++ standard-library symbols private inside the library.
Applications use the functions and structures in
[`spingnn2_c_api.h`](include/spingnn2_c_api.h). The library handles errors
internally and exposes a thread-local error string.

### Standalone toolchain probe

Append `--probe` to the same build command to build the
`sp2_toolchain_probe` executable. Run it with a `.sp2.json` contract manifest to verify SHA-256
and FP32/FP64 GEMM. With a CUDA build, run the executable on the corresponding
GPU partition. Use `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to match the CPU
allocation.

[Third-party notices](../THIRD_PARTY_LICENSES.md)

## Execution and memory

One model owns immutable device weights and a reusable workspace. A graph call
preserves neighbor order within each center, then streams complete neighborhoods
through that workspace. Both the sparse product and its environment adjoint index
the center tile directly. GEMMs operate on folded scalar weights or packed
irrep blocks. The reverse tape contains explicit adjoints for the energy
path. Covariance and BEC heads run forward. Frame corrections and extensive
reductions run after center assembly, including hard acoustic sum rules.

The standard graph ABI takes host arrays and returns host results. CUDA uploads node inputs and
CSR row pointers once and gathers edge inputs from resident arrays. It streams
contiguous targets and image shifts through bounded pinned host storage and two
device edge slots. A center-owned reduction accumulates atomic virials and
combines force contributions with neighbor updates on the device. The complete
result array is copied back once per evaluation. Already grouped uncached
inputs can borrow the caller's edge arrays for the duration of the call. Caches
own their fixed geometry. Device memory includes the model weights and
graph-sized atom inputs/outputs in addition to the tile workspace. CUDA caches
keep a bounded set of lattice tapes on the device. Frozen
tile storage holds the input and intermediates needed by the adjoint. Stage 2
reads them directly and reuses a separate workspace for spin-dependent operations.
Additional tiles spill into host snapshots and restore to stable device addresses.
CPU caches use host snapshots. A cache and
a regular evaluation can coexist on one model. The cache is valid across
changes of spin with fixed positions, cell, graph and species.

The timings in `sp2_diagnostics` use execution-space fences only when profiling
is enabled. The three forward categories record geometry/bases, MLP and tensor
product/linear work. Reverse measures the tile adjoint. Call wall time also
includes CSR preparation, transfers, field assembly and the frame reductions.
`workspace_bytes` measures the active device workspace including graph inputs,
outputs and tile storage. `cache_host_bytes` counts the saved host snapshot payload.
`cache_device_bytes` counts retained CUDA cache allocations. These counters
exclude host graph storage and C++ tape metadata. For total device memory
use the independent CUDA measurement in the benchmark.

## Interface conventions

Set `struct_size` on each input structure. Use `sp2_model_info` to read the
selected floating-point precision, training species order when available, and
required cutoff. All real input and output arrays use that selected precision.
A result field's name, shape and data remain valid through the lifetime of its
result handle. Shape rank and extents distinguish graph, atom and tensor fields.
A failed function returns a nonzero status and a thread-local diagnostic
available via `sp2_last_error`. Callers own handles and must coordinate
destruction with active calls. Cache handles keep their native model alive.

Kokkos initializes on first load if the linked runtime is uninitialized. An
external owner retains responsibility for its runtime. Each library is built
with private backend symbols so callers use the C ABI across compiler and
execution-space boundaries.

### Domain-decomposed evaluation

Set `sp2_graph.num_global` and `sp2_graph.num_owned` for a one-frame rank
partition with owned atoms followed by ghost images. Supply complete cutoff
neighborhoods for the owned centers. Ghost images carry their already
translated coordinates if the application handles periodic boundaries.
`owned_energy` contains the rank's center contributions to physical total
energy, including any share of the model's global shift. Virials and readouts
span owned centers. Forces and spin forces span all endpoints and require
reverse communication from each ghost to its owner. For hard BEC ASR,
subtract the global mean tensor from each owned readout, then add the
`bec_asr_shift` reported by `sp2_model_info`.

The CUDA neighbor ABI uses `sp2_model_load_device` with the caller's device
ordinal and `sp2_evaluate_neighbors` with strided device pointers. The input
is a full neighbor list at cutoff plus skin, using the caller's device atom
types and a device species map. Rebuilding the list, reordering atoms, or
changing atom types requires a new topology epoch. The engine keeps the skin
list on the device and builds a compact CSR of physical neighbors on each
geometry update. Edges at or beyond the model cutoff are removed before
neural-network evaluation.

The CUDA neighbor call requires the prepared JIT image. It returns
model-precision device pointers ready for use on the caller's stream.
Gather, scatter and readouts execute on the device. On each geometry update,
one batched transfer of 32-bit tile boundaries sets the host-side GEMM
dimensions. The diagnostics include these bytes and additional metadata
read when the skin list is rebuilt. A matching nonzero lattice epoch reuses
the compact edges without a metadata transfer. The engine retains a bounded
number of Stage-1 tapes on the device and reevaluates other tiles. Each
cache hit counts one reused tape. A frozen evaluation can omit lattice
derivatives through `SP2_SKIP_LATTICE_FORCES`. CUDA Graph replay reads the
current compact edge arrays. A change in tile edge count requires a new capture.

## Prepared JIT artifacts

```sh
python tools/ci/source_python.py spingnn2-prepare-jit-artifact \
    --model model.sp2.json --precision float32 --backend cuda --arch sm120 \
    --library validation/native/build-cuda-sm120/libspingnn2.so
```

The command prints the manifest path. The cache uses
`SPINGNN2_JIT_CACHE` or `~/.cache/spingnn2/jit`, with `--cache-dir` as an
explicit override. Keep the manifest and its named kernel image together.
Copy both files to the destination inference machine and load the manifest
with `sp2_model_load_jit_artifact` or Python `NativeModel(artifact=...,
execution_mode="direct")`. Files and their containing directory must belong
to the caller or root and must be unwritable by group or others. The loader
hashes the contract, checks the image SHA-256 and embedded identity, and
checks every required symbol. Only load artifacts produced by a trusted
preparer. These checks bind model and code but do not authenticate an
external publisher.

CUDA preparation uses `libspingnn2_jit_prepare.so`, which loads NVRTC
from the toolkit without a GPU or driver. `SPINGNN2_NVRTC_LIBRARY` selects
a specific NVRTC library. The inference library loads the validated cubin
through the CUDA Driver API and can run with the preparation library absent.
For CPU preparation, use `--backend cpu` with an OpenMP-capable C++ compiler.
