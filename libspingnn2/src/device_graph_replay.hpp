#pragma once

#ifdef SP2_ENABLE_CUDA
#include "blas.hpp"
#include <optional>

namespace sp2 {

// A mode is warmed eagerly once before capture. Multiple graphs may share
// workspace because every evaluation and graph launch uses one ordered stream.
struct NeighborGraphKey {
    const void *positions, *spins, *output;
    std::array<int64_t, 6> layout;
    uint64_t lattice, workspace;
    uint32_t flags;

    bool operator==(const NeighborGraphKey &other) const {
        return positions == other.positions && spins == other.spins &&
               output == other.output && layout == other.layout && lattice == other.lattice &&
               workspace == other.workspace && flags == other.flags;
    }
};

class NeighborGraph {
    std::optional<NeighborGraphKey> key;
    cudaGraphExec_t exec = nullptr;
    bool failed = false;

public:
    uint64_t peak_tile_edges = 0, cache_hits = 0;

    NeighborGraph() = default;
    NeighborGraph(const NeighborGraph &) = delete;
    NeighborGraph &operator=(const NeighborGraph &) = delete;
    ~NeighborGraph() { reset(); }

    void reset() {
        if (exec) cudaGraphExecDestroy(std::exchange(exec, nullptr));
        key.reset();
        failed = false;
    }

    bool match(const NeighborGraphKey &value) const { return key && *key == value; }
    bool available() const { return !failed; }

    void warm(const NeighborGraphKey &value) {
        reset();
        key = value;
    }

    void disable() { failed = true; }

    template<class Enqueue>
    sp2_cuda_graph_status run(cudaStream_t stream, Enqueue enqueue) {
        if (exec) {
            if (cudaGraphLaunch(exec, stream) == cudaSuccess) return SP2_GRAPH_REPLAYED;
        } else if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
            bool queued = false;
            try {
                enqueue();
                queued = true;
            } catch (...) {
                // Eager retry reports real evaluation errors after ending capture.
            }
            cudaGraph_t graph = nullptr;
            const auto status = cudaStreamEndCapture(stream, &graph);
            if (queued && status == cudaSuccess && graph)
                cudaGraphInstantiate(&exec, graph, 0);
            if (graph) cudaGraphDestroy(graph);
            if (exec && cudaGraphLaunch(exec, stream) == cudaSuccess) return SP2_GRAPH_CAPTURED;
        }
        if (exec) cudaGraphExecDestroy(std::exchange(exec, nullptr));
        cudaGetLastError();
        failed = true;
        return SP2_GRAPH_FALLBACK;
    }
};

class GraphBlas {
    std::unique_ptr<BlasHandle> handle;
    void *scratch = nullptr;
    static constexpr size_t capacity = 32u * 1024u * 1024u;
public:
    GraphBlas() = default;
    GraphBlas(const GraphBlas &) = delete;
    GraphBlas &operator=(const GraphBlas &) = delete;
    ~GraphBlas() {
        handle.reset();
        if (scratch) cudaFree(scratch);
    }

    BlasHandle &get(const Exec &exec) {
        if (!handle) {
            auto next = std::make_unique<BlasHandle>(exec);
            // With user workspace, cuBLAS does not capture per-GEMM memory
            // nodes and the graph remains reusable across ordinary launches.
            if (!scratch) cuda_check(cudaMalloc(&scratch, capacity), "Allocate cuBLAS graph workspace");
            check_blas(cublasSetWorkspace(next->handle, scratch, capacity));
            handle = std::move(next);
        }
        return *handle;
    }

    size_t bytes() const { return scratch ? capacity : 0; }
};

} // namespace sp2
#endif
