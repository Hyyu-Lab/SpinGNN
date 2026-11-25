# SpinGNN2 Native Dependencies & Third-Party Notices

- **Kokkos 4.7.02** provides OpenMP and CUDA kernels and device storage for
  `libspingnn2`. Its runtime is linked statically into the C ABI library. The
  release includes Desul atomics and mdspan selected internally by Kokkos.
  [Release source](https://github.com/kokkos/kokkos/tree/4.7.02) ·
  [Apache-2.0 with LLVM exception](LICENSES/Kokkos-4.7.02.txt) ·
  [Kokkos copyright](LICENSES/Kokkos-Copyright.txt) ·
  [Desul notices](https://github.com/kokkos/kokkos/tree/4.7.02/tpls/desul) ·
  [mdspan notices](https://github.com/kokkos/kokkos/tree/4.7.02/tpls/mdspan)
- **nlohmann/json 3.11.3** provides the host-only parser for `.sp2.json`
  contract manifests.
  [Release source](https://github.com/nlohmann/json/tree/v3.11.3) ·
  [MIT license](LICENSES/nlohmann-json-3.11.3.txt)
- **OpenSSL libcrypto** supplies SHA-256 verification through its EVP
  interface. The library links against the OpenSSL installation selected by
  CMake. [OpenSSL license](https://openssl-library.org/source/license/)
- **BLAS and cuBLAS** supply GEMM. CPU builds link to an installed CBLAS
  implementation such as OpenBLAS or Intel oneMKL. CUDA builds link to cuBLAS
  and the CUDA runtime installed with the toolkit.
  [OpenBLAS license](https://github.com/OpenMathLib/OpenBLAS/blob/develop/LICENSE) ·
  [Intel oneMKL license](https://www.intel.com/content/www/us/en/developer/articles/license/end-user-license-agreement.html) ·
  [CUDA Toolkit license](https://docs.nvidia.com/cuda/eula/index.html)
- **NVIDIA NVRTC and CUDA Driver API** are dynamically loaded from the
  installed toolkit and driver (`libspingnn2_jit_prepare.so` for offline
  compilation and `libcuda.so.1` for loading pre-compiled cubins at runtime).

The LAMMPS overlay sources and patches in `lammps_spingnn2/` use GNU GPL
version 2. They target the upstream annotated tag `stable_30Sep2026_update1`
(`17ea1b9effad496ec60d0f079e8b0aaa4c5b6d89`), which resolves to commit
`098ffdefa525cabc82272925de02e9884f58c657`. Ported SPIN and DREAM files retain
their original notices. The native library and Python runtime retain the MIT
license and the dependency notices above.
[Upstream tag](https://github.com/lammps/lammps/releases/tag/stable_30Sep2026_update1) ·
[GPL-2.0](LICENSES/LAMMPS-GPL-2.0.txt)
