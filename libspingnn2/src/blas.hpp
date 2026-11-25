#ifndef SPINGNN2_BLAS_HPP
#define SPINGNN2_BLAS_HPP

#include "tensor.hpp"

#ifdef SP2_ENABLE_CUDA
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cstdint>
#include <unordered_map>
#elif defined(SP2_USE_MKL)
#include <mkl.h>
#else
#include <cblas.h>
#endif
#include <type_traits>

namespace sp2 {
#ifdef SP2_ENABLE_CUDA
inline void check_blas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS failed with status " + std::to_string(status));
    }
}

struct LtGemmKey {
    int device;
    int m, n, k;
    int lda, ldb, ldc;
    uint8_t flags; // bit0: tf32, bit1: trans_a, bit2: trans_b
    uint8_t align_a, align_b, align_c;

    bool operator==(const LtGemmKey &o) const noexcept {
        return device == o.device && m == o.m && n == o.n && k == o.k &&
               lda == o.lda && ldb == o.ldb && ldc == o.ldc &&
               flags == o.flags && align_a == o.align_a &&
               align_b == o.align_b && align_c == o.align_c;
    }
};

struct LtGemmKeyHash {
    std::size_t operator()(const LtGemmKey &k) const noexcept {
        std::size_t h = std::hash<int>{}(k.device);
        auto mix = [&](std::size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(static_cast<std::size_t>(k.m));
        mix(static_cast<std::size_t>(k.n));
        mix(static_cast<std::size_t>(k.k));
        mix(static_cast<std::size_t>(k.lda));
        mix(static_cast<std::size_t>(k.ldb));
        mix(static_cast<std::size_t>(k.ldc));
        mix((static_cast<std::size_t>(k.flags) << 24) |
            (static_cast<std::size_t>(k.align_a) << 16) |
            (static_cast<std::size_t>(k.align_b) << 8) |
            static_cast<std::size_t>(k.align_c));
        return h;
    }
};

struct LtPlan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr; // col-major B^T
    cublasLtMatrixLayout_t b_desc = nullptr; // col-major A^T
    cublasLtMatrixLayout_t c_desc = nullptr; // col-major C^T
    cublasLtMatmulAlgo_t algo{};
    bool valid = false;

    LtPlan() = default;
    LtPlan(const LtPlan &) = delete;
    LtPlan &operator=(const LtPlan &) = delete;
    LtPlan(LtPlan &&o) noexcept
        : op(std::exchange(o.op, nullptr)),
          a_desc(std::exchange(o.a_desc, nullptr)),
          b_desc(std::exchange(o.b_desc, nullptr)),
          c_desc(std::exchange(o.c_desc, nullptr)),
          algo(o.algo),
          valid(o.valid) {}
    ~LtPlan() {
        if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
        if (op) cublasLtMatmulDescDestroy(op);
    }
};

class BlasHandle {
public:
    static constexpr std::size_t lt_workspace_bytes = 4u * 1024u * 1024u;

    explicit BlasHandle(const Exec& exec) : device(exec.cuda_device()) {
        check_blas(cublasCreate(&handle));
        try {
            check_blas(cublasSetStream(handle, exec.cuda_stream()));
            check_blas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
            stream = exec.cuda_stream();
            check_blas(cublasLtCreate(&lt_handle));
            cuda_check(cudaMalloc(&lt_workspace, lt_workspace_bytes),
                       "Allocate persistent cuBLASLt workspace");
        } catch (...) {
            if (lt_workspace) cudaFree(lt_workspace);
            if (lt_handle) cublasLtDestroy(lt_handle);
            cublasDestroy(handle);
            throw;
        }
    }
    ~BlasHandle() {
        lt_cache.clear();
        if (lt_workspace) cudaFree(lt_workspace);
        if (lt_handle) cublasLtDestroy(lt_handle);
        cublasDestroy(handle);
    }
    BlasHandle(const BlasHandle&) = delete;
    BlasHandle& operator=(const BlasHandle&) = delete;

    const LtPlan *get_lt_plan(const LtGemmKey &key, bool tf32, bool trans_a, bool trans_b,
                              int a_rows, int a_cols, int b_rows, int b_cols,
                              int m, int n, cudaStream_t active_stream) {
        auto it = lt_cache.find(key);
        if (it != lt_cache.end()) return it->second.valid ? &it->second : nullptr;

        // Heuristic selection must never run inside CUDA Graph capture.
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(active_stream, &capture_status) == cudaSuccess &&
            capture_status != cudaStreamCaptureStatusNone) {
            return nullptr;
        }

        if (lt_cache.size() >= 512) lt_cache.clear();

        LtPlan plan;
        const cublasComputeType_t compute =
            tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
        const cublasOperation_t ta = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasOperation_t tb = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

        cublasLtMatmulPreference_t pref = nullptr;
        auto cleanup_pref = [&]() {
            if (pref) cublasLtMatmulPreferenceDestroy(pref);
        };
        try {
            check_blas(cublasLtMatmulDescCreate(&plan.op, compute, CUDA_R_32F));
            check_blas(cublasLtMatmulDescSetAttribute(
                plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &tb, sizeof(tb)));
            check_blas(cublasLtMatmulDescSetAttribute(
                plan.op, CUBLASLT_MATMUL_DESC_TRANSB, &ta, sizeof(ta)));
            // Column-major operands: C^T(n, m) = op(B^T) * op(A^T)
            check_blas(cublasLtMatrixLayoutCreate(&plan.a_desc, CUDA_R_32F, b_cols, b_rows, b_cols));
            check_blas(cublasLtMatrixLayoutCreate(&plan.b_desc, CUDA_R_32F, a_cols, a_rows, a_cols));
            check_blas(cublasLtMatrixLayoutCreate(&plan.c_desc, CUDA_R_32F, n, m, n));

            check_blas(cublasLtMatmulPreferenceCreate(&pref));
            const std::size_t ws = lt_workspace_bytes;
            check_blas(cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
            const uint32_t align_a = key.align_b; // left operand in cublasLt is B^T
            const uint32_t align_b = key.align_a; // right operand in cublasLt is A^T
            const uint32_t align_c = key.align_c;
            check_blas(cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &align_a, sizeof(align_a)));
            check_blas(cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &align_b, sizeof(align_b)));
            check_blas(cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &align_c, sizeof(align_c)));
            check_blas(cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &align_c, sizeof(align_c)));

            cublasLtMatmulHeuristicResult_t results[4]{};
            int returned = 0;
            const auto st = cublasLtMatmulAlgoGetHeuristic(
                lt_handle, plan.op, plan.a_desc, plan.b_desc, plan.c_desc, plan.c_desc,
                pref, 4, results, &returned);
            cleanup_pref();
            if (st == CUBLAS_STATUS_SUCCESS && returned > 0) {
                for (int r = 0; r < returned; ++r) {
                    if (results[r].state == CUBLAS_STATUS_SUCCESS &&
                        results[r].workspaceSize <= lt_workspace_bytes) {
                        plan.algo = results[r].algo;
                        plan.valid = true;
                        break;
                    }
                }
            }
        } catch (...) {
            cleanup_pref();
            plan.valid = false;
        }
        auto [inserted, _] = lt_cache.emplace(key, std::move(plan));
        return inserted->second.valid ? &inserted->second : nullptr;
    }

    int device = 0;
    cublasHandle_t handle{};
    cublasLtHandle_t lt_handle{};
    void *lt_workspace = nullptr;
    cudaStream_t stream{};
    cublasMath_t math = CUBLAS_DEFAULT_MATH;
    std::unordered_map<LtGemmKey, LtPlan, LtGemmKeyHash> lt_cache;
};

// cuBLAS handles and their math modes are thread-local. Each evaluation,
// including a cached one, selects its own mode before enqueueing its GEMMs.
inline thread_local bool use_tf32 = false;
inline thread_local BlasHandle *capture_blas = nullptr;

class UseCaptureBlas {
    BlasHandle *previous;
public:
    explicit UseCaptureBlas(BlasHandle &handle) : previous(capture_blas) { capture_blas = &handle; }
    ~UseCaptureBlas() { capture_blas = previous; }
    UseCaptureBlas(const UseCaptureBlas &) = delete;
    UseCaptureBlas &operator=(const UseCaptureBlas &) = delete;
};

inline BlasHandle &blas_handle(const Exec &exec) {
    auto normal_handle = [&]() -> BlasHandle & {
        thread_local BlasHandle handle(exec);
        return handle;
    };
    auto &blas = capture_blas ? *capture_blas : normal_handle();
    if (blas.stream != exec.cuda_stream()) {
        check_blas(cublasSetStream(blas.handle, exec.cuda_stream()));
        blas.stream = exec.cuda_stream();
    }
    const auto mode = use_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    if (blas.math != mode) {
        check_blas(cublasSetMathMode(blas.handle, mode));
        blas.math = mode;
    }
    return blas;
}

inline cublasHandle_t cublas(const Exec& exec) {
    return blas_handle(exec).handle;
}

inline uint8_t ptr_alignment(const void *p) noexcept {
    const auto addr = reinterpret_cast<std::uintptr_t>(p);
    if ((addr & 15u) == 0u) return 16u;
    if ((addr & 7u) == 0u) return 8u;
    return 4u;
}
#endif

template<class T>
void gemm(const Exec& exec, Tensor<T> out, Tensor<T> a, Tensor<T> b,
          bool trans_a = false, bool trans_b = false, T alpha = T(1), T beta = T(0)) {
    const int m = trans_a ? a.cols : a.rows;
    const int k = trans_a ? a.rows : a.cols;
    const int n = trans_b ? b.rows : b.cols;
    if ((trans_b ? b.cols : b.rows) != k || out.rows != m || out.cols != n) {
        throw std::logic_error("Invalid tile GEMM dimensions");
    }
    if (!m || !n) return;
    if (!k) {
        for_each(exec, "scale empty GEMM", out.size(), KOKKOS_LAMBDA(int i) {
            out.ptr[i] = beta == T(0) ? T(0) : beta * out.ptr[i];
        });
        return;
    }
#ifdef SP2_ENABLE_CUDA
    auto &blas = blas_handle(exec);
    const auto ta = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto tb = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    // cuBLAS uses column-major storage, so compute C^T = B^T A^T.
    if constexpr (std::is_same_v<T,float>) {
        // Use cached cuBLASLt heuristic selection with strict CUBLAS_COMPUTE_32F
        // default (TF32 only on explicit opt-in), avoiding split-K scal_kernel
        // overhead while remaining CUDA-Graph-capture safe.
        const LtGemmKey key{
            blas.device, m, n, k, a.cols, b.cols, n,
            static_cast<uint8_t>((use_tf32 ? 1u : 0u) | (trans_a ? 2u : 0u) | (trans_b ? 4u : 0u)),
            ptr_alignment(a.ptr), ptr_alignment(b.ptr), ptr_alignment(out.ptr)};
        if (const LtPlan *plan = blas.get_lt_plan(
                key, use_tf32, trans_a, trans_b, a.rows, a.cols, b.rows, b.cols,
                m, n, exec.cuda_stream())) {
            const auto st = cublasLtMatmul(
                blas.lt_handle, plan->op, &alpha,
                b.ptr, plan->a_desc, a.ptr, plan->b_desc,
                &beta, out.ptr, plan->c_desc, out.ptr, plan->c_desc,
                &plan->algo, blas.lt_workspace, BlasHandle::lt_workspace_bytes,
                exec.cuda_stream());
            if (st == CUBLAS_STATUS_SUCCESS) return;
        }
        const auto compute = use_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
        check_blas(cublasGemmEx(blas.handle, tb, ta, n, m, k, &alpha, b.ptr, CUDA_R_32F, b.cols,
                               a.ptr, CUDA_R_32F, a.cols, &beta, out.ptr, CUDA_R_32F, n,
                               compute, CUBLAS_GEMM_DEFAULT));
    } else {
        check_blas(cublasDgemm(blas.handle, tb, ta, n, m, k, &alpha, b.ptr, b.cols,
                              a.ptr, a.cols, &beta, out.ptr, n));
    }
#else
    (void)exec;
    const auto ta = trans_a ? CblasTrans : CblasNoTrans;
    const auto tb = trans_b ? CblasTrans : CblasNoTrans;
    if constexpr (std::is_same_v<T,float>) {
        cblas_sgemm(CblasRowMajor, ta, tb, m, n, k, alpha, a.ptr, a.cols,
                    b.ptr, b.cols, beta, out.ptr, n);
    } else {
        cblas_dgemm(CblasRowMajor, ta, tb, m, n, k, alpha, a.ptr, a.cols,
                    b.ptr, b.cols, beta, out.ptr, n);
    }
#endif
}
}  // namespace sp2
#endif
