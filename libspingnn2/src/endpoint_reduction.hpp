#pragma once
#include "graph.hpp"
#include "tile.hpp"

namespace sp2 {
// Index the touched atoms of a center tile. Sorted CSR incidence within each
// row preserves the eager center/neighbor order, including periodic self edges.
template <class T> class EndpointReduction {
    std::vector<int> lookup, atoms, offsets, signed_edges, cursor;
    DeviceArray<int> device_offsets, device_edges;
    std::vector<T> force, spin_force;

  public:
    template <class Buffers>
    void apply(Buffers &b, const Graph<T> &graph, int start, const TileInput<T> &input,
               const TileResult<T> &tile, std::vector<T> *forces, std::vector<T> *spins,
               std::vector<T> *virials) {
        lookup.resize(graph.n, -1);
        atoms.clear();
        offsets.assign(1, 0);
        const int stop = start + input.ncenters;
        auto count = [&](int atom) {
            if (lookup[atom] < 0) {
                lookup[atom] = int(atoms.size());
                atoms.push_back(atom);
                offsets.push_back(0);
            }
            ++offsets[lookup[atom] + 1];
        };
        for (int i = start; i < stop; ++i)
            for (int k = graph.offsets[i]; k < graph.offsets[i + 1]; ++k) {
                count(i);
                count(graph.target[k]);
            }
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        cursor = offsets;
        signed_edges.resize(offsets.back());
        for (int i = start; i < stop; ++i)
            for (int k = graph.offsets[i]; k < graph.offsets[i + 1]; ++k) {
                int code = k - graph.offsets[start] + 1;
                signed_edges[cursor[lookup[i]]++] = code;
                signed_edges[cursor[lookup[graph.target[k]]]++] = -code;
            }
        auto transfer = [&](DeviceArray<int> &dst,
                            const std::vector<int> &source) -> DeviceArray<int> {
            if (dst.extent(0) < source.size())
                dst = DeviceArray<int>("endpoint incidence", source.size());
            b.send(dst.data(), source.data(), source.size());
            return Kokkos::subview(dst, std::make_pair(size_t(0), source.size()));
        };
        auto rows = transfer(device_offsets, offsets), edges = transfer(device_edges, signed_edges);
        auto gather = [&](std::vector<T> *from, std::vector<T> &temp) -> Tensor<T> {
            if (!from)
                return {};
            temp.resize(atoms.size() * 3);
            for (size_t p = 0; p < atoms.size(); ++p)
                std::copy_n(from->data() + 3ull * atoms[p], 3, temp.data() + 3 * p);
            return b.tensor(temp, 3);
        };
        auto f = gather(forces, force), s = gather(spins, spin_force);
        auto v = virials ? b.w.make(input.ncenters, 9) : Tensor<T>{};
        jit_scatter(b.w, input, tile, rows, edges, f, s, v, input.frozen);
        if (forces)
            b.receive(force.data(), f);
        if (spins)
            b.receive(spin_force.data(), s);
        if (virials)
            b.receive(virials->data() + 9ull * start, v);
        b.w.exec.fence("read compact endpoint reduction");
        for (size_t p = 0; p < atoms.size(); ++p) {
            int atom = atoms[p];
            if (forces)
                std::copy_n(force.data() + 3 * p, 3, forces->data() + 3ull * atom);
            if (spins)
                std::copy_n(spin_force.data() + 3 * p, 3, spins->data() + 3ull * atom);
            lookup[atom] = -1;
        }
    }
    size_t bytes() const {
        return sizeof(int) * (device_offsets.extent(0) + device_edges.extent(0));
    }
};
} // namespace sp2
