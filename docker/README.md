# SpinGNN CUDA 12.8 container

This image packages the validated Python 3.10.20 / PyTorch 2.11.0+cu128
environment and prebuilt serial and MPI LAMMPS executables. It is intended for
Linux x86-64 NVIDIA systems whose host driver supports CUDA 12.8.

The image supports NVIDIA A100 (Ampere, `sm_80`),
RTX 4090 (Ada, `sm_89`), and H100 (Hopper, `sm_90`). The image exports
`TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"` so runtime JIT extension builds cover all
three. The supported architectures are recorded in the
`org.spingnn.cuda.architectures` OCI label. The PyTorch wheel uses native
kernels on A100 and H100 and compiles kernels from PTX on RTX 4090, which
can add startup time. V100 requires a different PyTorch/CUDA environment
because the wheel starts at `sm_75`.

For RTX 5090 (Blackwell, `sm_120`), use the repository-local environment
built by `tools/ci/install_gpu_env.sh`. The image's architecture list stops
at `sm_90`. See [the CUDA installation guide](../docs/INSTALL_CUDA128.md).

Export AOTInductor `.pt2` packages on the GPU architecture that will run
them. LAMMPS validates the architecture and LibTorch CUDA ABI in the sidecar.
See the [PT2 deployment guide](../docs/PT2_MIGRATION.md) for details.

## Host requirements

- Docker Engine (the recipe supports both the legacy builder and BuildKit),
  or Buildah for daemonless image construction;
- an NVIDIA driver compatible with CUDA 12.8;
- NVIDIA Container Toolkit configured so `docker run --gpus all` works;
- network access to the base-image registry, the PyTorch cu128 index, and PyPI.

The Dockerfile uses NVIDIA's public `nvcr.io/nvidia/pytorch:25.02-py3` base
image. `docker/build.sh` archives the local SpinGNN checkout and the local
[T-e3nn checkout](https://github.com/Hongyu-yu/T-e3nn) pinned to v1.0 at
`75334027fffc38f2268db8ab1eed281a18ffdbf2`.

## Build

Run the [source checkout steps](../INSTALL.md#21-clone-the-sources-on-shared-storage)
to place `SpinGNN/` and `validation/vendor/T-e3nn/` under the same workspace
directory. Use clean checkouts because the build script archives both source
revisions into the image.

```bash
cd /shared/path/spingnn/SpinGNN
docker/build.sh spingnn:cuda128
```

For a daemonless build, use Buildah:

```bash
SPINGNN_CONTAINER_BUILDER=buildah \
BUILDAH_STORAGE_DRIVER=vfs BUILDAH_ISOLATION=chroot \
  docker/build.sh spingnn:cuda128
```

The command produces a Docker-format image. Buildah uses TLS verification
for registry access by default.

For a network-isolated builder, stage the uv-managed CPython directory and a
complete wheelhouse under an assets directory, then force offline resolution:

```text
/shared/path/docker-assets/
|-- python/cpython-3.10.20-linux-x86_64-gnu/
`-- wheelhouse/
```

```bash
SPINGNN_OFFLINE=1 \
SPINGNN_OFFLINE_ASSETS=/shared/path/docker-assets \
  docker/build.sh spingnn:cuda128
```

Offline package installation requires all wheels in the supplied wheelhouse.
The build still needs the base image and Debian/Ubuntu package repository
access for the operating-system packages.

The build script records the SpinGNN and T-e3nn commits in OCI labels. It
installs T-e3nn with `--no-build-isolation` first, then installs Spin-NequIP and
Spin-Allegro together with `--no-deps`. This ordering is part of the runtime
contract.

To publish through a registry account you control:

```bash
export REGISTRY=registry.example.com
export IMAGE_NAMESPACE=your-namespace
export SPINGNN_IMAGE="$REGISTRY/$IMAGE_NAMESPACE/spingnn:cuda128"
docker login "$REGISTRY"
docker tag spingnn:cuda128 "$SPINGNN_IMAGE"
docker push "$SPINGNN_IMAGE"
```

Do not put registry passwords or access tokens in the Dockerfile, build
arguments, image labels, or shell history.

## Pull and run

For a local build, use the tag passed to the build script. To run a published
image, set `SPINGNN_IMAGE` to its registry-qualified name and pull it first.

```bash
export SPINGNN_IMAGE=spingnn:cuda128
docker pull "$SPINGNN_IMAGE"  # omit for a local build
docker run --rm --gpus all "$SPINGNN_IMAGE" spingnn-check
```

For checkout-based verification rather than the image smoke command, select
the hardware contract explicitly:

```bash
SPINGNN_GPU_EXPECT=h100 tools/ci/gpu_full_tests.sh
SPINGNN_GPU_EXPECT=4090 tools/ci/gpu_full_tests.sh
```

Run the Python and training commands directly:

```bash
docker run --rm --gpus all "$SPINGNN_IMAGE" \
  python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name())'
docker run --rm --gpus all "$SPINGNN_IMAGE" nequip-train --help
```

The serial executable is the default `lmp`. `lmp-mpi` selects the MPI build:

```bash
docker run --rm --gpus all -v "$PWD:/work" "$SPINGNN_IMAGE" \
  lmp -in /work/in.lammps
docker run --rm --gpus all "$SPINGNN_IMAGE" \
  mpirun --allow-run-as-root -np 2 lmp-mpi -h
```

The smoke command takes a directory containing `nosoc.pth` (Mn/Ge) and
`bec.pth` (Li/Nb/O), and a directory containing `nosoc.data`, `bec.data`,
and `bec.bec` for the two systems. Its defaults are `/validation/models`
and `/validation/lammps`:

```bash
docker run --rm --gpus all \
  -v /shared/path/spingnn/validation:/validation:ro \
  "$SPINGNN_IMAGE" spingnn-smoke
```

This runs a NoSOC energy and force check, three `spin/sib` steps, and a
BEC energy and force check. Supply the models and inputs when starting
the container.

## Deterministic execution

Enable deterministic CUDA algorithms with the project switch and the cuBLAS
workspace configuration:

```bash
docker run --rm --gpus all \
  -e SPINGNN_DETERMINISTIC=1 \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$SPINGNN_IMAGE" spingnn-check
```

Deterministic mode can reduce performance and can reject operations without a
deterministic CUDA implementation, so it is not the image default.

## Image layout

| Path | Contents |
| --- | --- |
| `/opt/spingnn/src` | SpinGNN source snapshot |
| `/opt/spingnn/vendor/T-e3nn` | T-e3nn source snapshot |
| `/opt/spingnn/env` | Python 3.10.20 virtual environment |
| `/opt/spingnn/lammps/serial/lmp` | Serial LAMMPS executable |
| `/opt/spingnn/lammps/mpi/lmp` | MPI LAMMPS executable |
| `/opt/spingnn/bin` | GPU check, LAMMPS wrappers, and smoke test |
