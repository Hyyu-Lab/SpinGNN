#pragma once
#include "spingnn2_c_api.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace sp2 {
template <class Message> inline void require(bool ok, const Message &msg) {
    if (!ok)
        throw std::invalid_argument(msg);
}

// Stable counting sort retains the edge order of each center. All topology
// and graph-sized outputs reside on the host. Device input uses one CSR tile.
template <class T> struct Graph {
    int n, e, frames, owned;
    int64_t global;
    std::vector<T> pos, shifts, cell, volume;
    std::vector<int> types, batch, offsets, target;
    std::vector<int64_t> counts;
    // Non-cached evaluations can borrow an already grouped input. Caches always
    // retain their own graph, so later spin evaluations need no caller-owned memory.
    const int32_t* input_target = nullptr;
    const T* input_shifts = nullptr;
    const int32_t* neighbors() const { return input_target ? input_target : target.data(); }
    const T* images() const { return input_shifts ? input_shifts : shifts.data(); }

    Graph(const sp2_graph &g, int num_types, bool borrow_ordered = false)
        : n(g.num_atoms), e(g.num_edges), frames(g.num_frames),
          global(g.struct_size >= sizeof(sp2_graph) ? g.num_global : 0) {
        owned = global ? g.num_owned : n;
        require(e >= 0 && global >= 0 &&
                    (global ? frames == 1 && owned >= 0 && owned <= n && owned <= global :
                              n > 0 && frames > 0 && frames <= n),
                "Invalid graph dimensions");
        require((!n || (g.positions && g.types)) && (!e || g.edge_index),
                "Missing positions, types or edges");
        require(frames == 1 || g.batch, "Multiple frames require batch indices");
        require(g.num_cells == 0 || g.num_cells == 1 || g.num_cells == frames,
                "Invalid cell count");
        require(!g.num_cells || (g.cells && (!e || g.edge_cell_shifts)),
                "Missing periodic geometry");
        const T *xyz = static_cast<const T *>(g.positions);
        if (n) {
            pos.assign(xyz, xyz + 3ull * n);
            types.assign(g.types, g.types + n);
        }
        batch.resize(n);
        counts.resize(frames);
        for (int i = 0; i < n; ++i) {
            int f = g.batch ? g.batch[i] : 0;
            require(types[i] >= 0 && types[i] < num_types && f >= 0 && f < frames,
                    "Type or batch index out of bounds");
            batch[i] = f;
            if (i < owned) ++counts[f];
        }
        if (global) counts[0] = global;
        require(*std::min_element(counts.begin(), counts.end()) > 0, "Empty frame in batch");
        offsets.resize(size_t(owned) + 1);
        bool grouped = borrow_ordered;
        for (int k = 0; k < e; ++k) {
            const int i = g.edge_index[k], j = g.edge_index[size_t(e) + k];
            if (k && i < g.edge_index[k - 1]) grouped = false;
            require(i >= 0 && i < owned && j >= 0 && j < n && batch[i] == batch[j],
                    "Invalid edge endpoints");
            ++offsets[size_t(i) + 1];
        }
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());

        if (g.num_cells) {
            const T *cells = static_cast<const T *>(g.cells);
            cell.assign(cells, cells + 9ull * g.num_cells);
            volume.resize(frames);
            for (int f = 0; f < frames; ++f) {
                const T *a = &cell[9ull * (g.num_cells == 1 ? 0 : f)];
                T det = a[0] * (a[4] * a[8] - a[5] * a[7]) + a[1] * (a[5] * a[6] - a[3] * a[8]) +
                        a[2] * (a[3] * a[7] - a[4] * a[6]);
                volume[f] = std::abs(det);
                require(volume[f] > std::numeric_limits<T>::epsilon() && std::isfinite(det),
                        "Invalid cell volume");
            }
        }
        const T *shift = static_cast<const T *>(g.edge_cell_shifts);
        if (grouped && e) {
            input_target = g.edge_index + e;
            input_shifts = shift;
            return;
        }
        auto cursor = offsets;
        target.resize(e);
        if (g.num_cells) shifts.resize(3ull * e);
        for (int k = 0; k < e; ++k) {
            int at = cursor[g.edge_index[k]]++;
            target[at] = g.edge_index[size_t(e) + k];
            if (g.num_cells)
                std::copy_n(shift + 3ull * k, 3, shifts.data() + 3ull * at);
        }
    }
    void displacement(int i, int edge, T *r) const {
        int j = neighbors()[edge];
        for (int k = 0; k < 3; ++k) {
            r[k] = pos[3ull * j + k] - pos[3ull * i + k];
            if (!cell.empty()) {
                const T *c = cell.data() + 9ull * (cell.size() == 9 ? 0 : batch[i]);
                const T *s = images() + 3ull * edge;
                r[k] += s[0] * c[k] + s[1] * c[3 + k] + s[2] * c[6 + k];
            }
        }
    }
};
} // namespace sp2
