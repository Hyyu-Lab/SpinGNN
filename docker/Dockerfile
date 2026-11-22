# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/pytorch:25.02-py3

ARG SPINGNN_SOURCE_COMMIT=unknown
ARG TE3NN_SOURCE_COMMIT=75334027fffc38f2268db8ab1eed281a18ffdbf2
ARG UV_VERSION=0.11.6
ARG SPINGNN_OFFLINE=0
ARG LAMMPS_BUILD_JOBS=4

LABEL org.opencontainers.image.title="SpinGNN CUDA 12.8 runtime" \
      org.opencontainers.image.description="SpinGNN Python and serial/MPI LAMMPS runtime for NVIDIA Ampere, Ada, and Hopper" \
      org.opencontainers.image.revision="${SPINGNN_SOURCE_COMMIT}" \
      org.opencontainers.image.source="https://github.com/Hyyu-Lab/SpinGNN" \
      org.spingnn.te3nn.revision="${TE3NN_SOURCE_COMMIT}" \
      org.spingnn.cuda.architectures="sm_80;sm_89;sm_90"

# Deterministic CUDA execution is opt-in for release-gate checks.
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    VIRTUAL_ENV=/opt/spingnn/env \
    PATH=/opt/spingnn/env/bin:/opt/spingnn/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/spingnn/env/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH} \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0" \
    UV_CACHE_DIR=/tmp/uv-cache \
    SPINGNN_DETERMINISTIC=0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libopenmpi-dev \
        ninja-build \
        openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# The build context is prepared by docker/build.sh from local source archives.
# The archives exclude Git metadata.
COPY T-e3nn /opt/spingnn/vendor/T-e3nn
COPY SpinGNN /opt/spingnn/src
COPY python /opt/spingnn/python
COPY wheelhouse /opt/spingnn/wheelhouse

RUN if [[ "${SPINGNN_OFFLINE}" == 1 ]]; then \
        test -x /opt/spingnn/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10; \
        python3 -m pip install --no-index --find-links=/opt/spingnn/wheelhouse \
            "uv==${UV_VERSION}"; \
    else \
        python3 -m pip install --no-cache-dir "uv==${UV_VERSION}"; \
        uv python install 3.10.20 --install-dir /opt/spingnn/python --no-bin; \
    fi \
    && uv venv --clear \
        --python /opt/spingnn/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10 \
        "${VIRTUAL_ENV}" \
    && if [[ "${SPINGNN_OFFLINE}" == 1 ]]; then \
        uv pip install --python "${VIRTUAL_ENV}/bin/python" \
            --no-index --find-links=/opt/spingnn/wheelhouse \
            "torch==2.11.0+cu128"; \
    else \
        uv pip install --python "${VIRTUAL_ENV}/bin/python" \
            --index-url https://download.pytorch.org/whl/cu128 \
            "torch==2.11.0+cu128"; \
    fi \
    && uv pip install --python "${VIRTUAL_ENV}/bin/python" \
        $([[ "${SPINGNN_OFFLINE}" == 1 ]] && printf '%s' '--no-index --find-links=/opt/spingnn/wheelhouse') \
        "cuda-pathfinder==1.6.0" \
        "filelock==3.32.3" \
        "numpy==1.26.4" \
        "ase==3.29.0" \
        "pandas==2.3.3" \
        "PyYAML==6.0.3" \
        "tqdm==4.70.0" \
        "torch-runstats==0.2.0" \
        "torch-ema==0.3" \
        "opt-einsum==3.4.0" \
        "opt-einsum-fx==0.1.4" \
        "packaging==26.3" \
        "pytest==9.1.1" \
        "scikit-learn==1.7.2" \
        "matplotlib==3.10.9" \
        "SciencePlots==2.2.2" \
        "scipy==1.15.3" \
        "pip==26.2.1" \
        "setuptools==78.1.0" \
    && if [[ "${SPINGNN_OFFLINE}" == 1 ]]; then \
        export PIP_NO_INDEX=1 PIP_FIND_LINKS=/opt/spingnn/wheelhouse; \
    fi \
    && "${VIRTUAL_ENV}/bin/python" -m pip install --no-build-isolation \
        /opt/spingnn/vendor/T-e3nn \
    && "${VIRTUAL_ENV}/bin/python" -m pip install --no-deps \
        /opt/spingnn/src/src/core /opt/spingnn/src/src/models \
    && rm -rf "${UV_CACHE_DIR}" /root/.cache/pip /opt/spingnn/wheelhouse

RUN set -eux; \
    torch_prefix=$(python -c 'import torch; print(torch.utils.cmake_prefix_path.splitlines()[0])'); \
    for variant in serial mpi; do \
        if [[ "${variant}" == mpi ]]; then build_mpi=yes; else build_mpi=no; fi; \
        cmake --fresh -S /opt/spingnn/src/lammps/cmake -B "/tmp/lammps-${variant}" \
            -G Ninja \
            -D PKG_SPIN=yes \
            -D PKG_ASPHERE=yes \
            -D PKG_BROWNIAN=yes \
            -D BUILD_MPI="${build_mpi}" \
            -D CMAKE_BUILD_TYPE=Release \
            -D CMAKE_CXX_STANDARD=17 \
            -D CMAKE_CXX_STANDARD_REQUIRED=yes \
            -D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
            -D "CMAKE_PREFIX_PATH=${torch_prefix}"; \
        cmake --build "/tmp/lammps-${variant}" --parallel "${LAMMPS_BUILD_JOBS}"; \
        install -D -m 0755 "/tmp/lammps-${variant}/lmp" "/opt/spingnn/lammps/${variant}/lmp"; \
    done; \
    /opt/spingnn/lammps/serial/lmp -h >/tmp/lammps-help.txt; \
    grep -Eq '(^|[[:space:]])allegro([[:space:]]|$)' /tmp/lammps-help.txt; \
    grep -q 'spin/allegro_spin' /tmp/lammps-help.txt; \
    grep -Eq '(^|[[:space:]])bec([[:space:]]|$)' /tmp/lammps-help.txt; \
    ldd /opt/spingnn/lammps/mpi/lmp | grep -qi mpi; \
    rm -rf /tmp/lammps-serial /tmp/lammps-mpi /tmp/lammps-help.txt

RUN install -d -m 0755 /opt/spingnn/bin \
    && install -m 0755 /opt/spingnn/src/docker/bin/spingnn-check /opt/spingnn/bin/spingnn-check \
    && install -m 0755 /opt/spingnn/src/docker/bin/spingnn-lmp /opt/spingnn/bin/spingnn-lmp \
    && install -m 0755 /opt/spingnn/src/docker/bin/spingnn-smoke /opt/spingnn/bin/spingnn-smoke \
    && ln -s spingnn-lmp /opt/spingnn/bin/lmp \
    && ln -s spingnn-lmp /opt/spingnn/bin/lmp-mpi \
    && python -c 'import allegro, e3nn, nequip, torch; \
assert torch.__version__ == "2.11.0+cu128"; \
assert torch.version.cuda == "12.8"; \
assert e3nn.__version__ == "0.5.0+spin"; \
assert nequip.__version__ == "0.5.6+spin"; \
assert allegro.__version__ == "0.2.0+spin"; \
print("SPINGNN_IMAGE_IMPORT_OK")'

WORKDIR /work
CMD ["bash"]
