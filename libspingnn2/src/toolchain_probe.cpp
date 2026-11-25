#include "contract.hpp"

#include <Kokkos_Core.hpp>

#ifdef SP2_ENABLE_CUDA
#include <cublas_v2.h>
#elif defined(SP2_USE_MKL)
#include <mkl.h>
#else
#include <cblas.h>
#endif

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

#ifdef SP2_ENABLE_CUDA
template<class T>
struct Cublas;
template<>
struct Cublas<float> {
    static constexpr auto gemm = cublasSgemm;
};
template<>
struct Cublas<double> {
    static constexpr auto gemm = cublasDgemm;
};
#else
template<class T>
struct Cblas;
template<>
struct Cblas<float> {
    static constexpr auto gemm = cblas_sgemm;
};
template<>
struct Cblas<double> {
    static constexpr auto gemm = cblas_dgemm;
};
#endif

template<class T>
void exercise() {
    Kokkos::View<T*> a("a", 4), b("b", 4), out("out", 4);
    Kokkos::parallel_for("populate", 4, KOKKOS_LAMBDA(int i) {
        a(i) = static_cast<T>(i + 1);
        b(i) = static_cast<T>(i + 5);
    });
    Kokkos::fence();
#ifdef SP2_ENABLE_CUDA
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS initialization failed");
    }
    const T alpha = 1;
    const T beta = 0;
    const auto status = Cublas<T>::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        2, 2, 2, &alpha, b.data(), 2, a.data(), 2, &beta, out.data(), 2);
    cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS GEMM failed");
    }
#else
    Cblas<T>::gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1,
                   a.data(), 2, b.data(), 2, 0, out.data(), 2);
#endif
    const auto result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
    if (result(0) != 19 || result(1) != 22 || result(2) != 43 || result(3) != 50) {
        throw std::runtime_error("GEMM produced unexpected results");
    }
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard runtime;
    try {
        std::cout << "Kokkos " << KOKKOS_VERSION << " default backend "
                  << Kokkos::DefaultExecutionSpace::name() << '\n';
        exercise<float>();
        exercise<double>();
        if (argc == 2) {
            const sp2::Contract contract(argv[1]);
            std::cout << "Verified contract family " << contract.model.at("family") << '\n';
        }
        std::cout << "FP32 and FP64 GEMM passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
