#pragma once

#include "jit_runtime.hpp"

namespace sp2 {

inline bool jit_basis_op(const std::string& op) {
    return op == "RadialBasisEdgeEncoding" || op == "EdgeScalarAddSpin" ||
           op == "RadialBasisEdgeEncoding_spin";
}

inline bool jit_harmonic_op(const std::string& op) {
    return op == "SphericalHarmonicEdgeAttrs" ||
           op == "SphericalHarmonicEdgeAttrs_spin_nosoc" ||
           op == "SphericalHarmonicEdgeAttrs_spin_soc_simple";
}

template<class T> constexpr bool jit_cooperative_geometry() {
#ifdef SP2_ENABLE_CUDA
    return sizeof(T) == sizeof(float);
#else
    return false;
#endif
}

template<class T> const char* jit_geometry_forward_suffix() {
#ifdef SP2_ENABLE_CUDA
    return jit_cooperative_geometry<T>() ? "_resident_warp8_fwd" : "_resident_fwd";
#else
    return "_fwd";
#endif
}

template<class T> const char* jit_geometry_reverse_suffix() {
#ifdef SP2_ENABLE_CUDA
    return jit_cooperative_geometry<T>() ? "_warp8_rev" : "_rev";
#else
    return "_rev";
#endif
}

template<class T> std::vector<std::string> geometry_jit_symbols(const Json& model) {
#ifdef SP2_ENABLE_CUDA
    std::vector<std::string> names{"sp2_geom_resident_displacement", "sp2_geom_scatter"};
#else
    std::vector<std::string> names{"sp2_geom_displacement", "sp2_geom_scatter"};
#endif
    const auto& layers = model.at("layers");
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        if (jit_harmonic_op(layer.at("op")))
            throw std::invalid_argument("JIT spherical harmonics require an adjacent edge basis");
        if (!jit_basis_op(layer.at("op"))) continue;
        const auto name = "sp2_geom_" + layer.at("name").get<std::string>();
        names.push_back(name + jit_geometry_forward_suffix<T>());
        names.push_back(name + jit_geometry_reverse_suffix<T>());
        if (i + 1 < layers.size() && jit_harmonic_op(layers[i + 1].at("op"))) ++i;
    }
    return names;
}

template<class T> void jit_displacement(
    Workspace<T>& work, const TileInput<T>& input, Tensor<T> length) {
#ifdef SP2_ENABLE_CUDA
    work.jit->launch(work.exec, "sp2_geom_resident_displacement", input.r.rows, 128,
        input.center.data(), input.targets.data(), input.frames.data(), input.start,
        static_cast<int>(input.shared_cell), input.positions, input.shifts.ptr, input.graph_cells,
        input.offsets.data(), input.ncenters, input.r.ptr, length.ptr);
#else
    work.jit->launch(work.exec, "sp2_geom_displacement", input.r.rows, 128,
        input.center.data(), input.pos_i.ptr, input.pos_j.ptr, input.shifts.ptr, input.cells.ptr,
        input.r.ptr, length.ptr);
#endif
}

template<class T> void jit_scatter(
    Workspace<T>& work, const TileInput<T>& input, const TileResult<T>& tile,
    const DeviceArray<int>& offsets, const DeviceArray<int>& signed_edges, Tensor<T> force,
    Tensor<T> spin_force, Tensor<T> atom_virial, bool frozen) {
    ProfileScope<T> profile(work, Profile::reverse);
    const int touched = frozen ? spin_force.rows : force.rows;
    const int virial_rows = frozen ? 0 : atom_virial.rows;
    if (virial_rows > INT_MAX - touched) throw std::length_error("Too many tile output rows");
    work.jit->launch(work.exec, "sp2_geom_scatter", touched + virial_rows, 128,
        touched, offsets.data(), signed_edges.data(), input.offsets.data(),
        tile.r.ptr, tile.length.ptr, tile.dr0.ptr, tile.dl.ptr, tile.ds_i.ptr, tile.ds_j.ptr,
        force.ptr, spin_force.ptr, atom_virial.ptr);
}

template<class T> void jit_geometry_layer(
    Workspace<T>& w, const TileInput<T>& input, TileState<T>& state,
    const std::string& symbol, const std::string& field, const std::string& out,
    const std::string& sh_out, BasisDevice<T> b, BasisDevice<T> spin_table,
    int sh_width, bool magnetic) {
    ProfileScope<T> profile(w, Profile::geometry);
    const auto length = state.fields.at("length");
    const bool combined = spin_table.width != 0;
    const auto prev = field.empty() ? Tensor<T>{} : state.fields.at(field);
    const int old_width = prev.cols, rows = length.rows;
    const int spin_width = combined ? spin_table.width : (magnetic ? b.width : 0);
    const auto value = w.make(rows, old_width + b.width + (combined ? spin_width : 0));
    const auto slopes = w.make(rows, b.width);
    const auto spin_local = magnetic ? w.make(rows, 4) : Tensor<T>{};
    const auto spin_values = magnetic ? w.make(rows, spin_width) : Tensor<T>{};
    const auto spin_slopes = combined ? w.make(rows, spin_width) : Tensor<T>{};
    const auto features = sh_width ? w.make(rows, sh_width) : Tensor<T>{};
    const auto jacobian = sh_width ? w.make(rows, sh_width * 3) : Tensor<T>{};
    const auto si = input.spins_i, sj = input.spins_j;
    const auto tile_offsets = input.offsets;
    const int ncenters = input.ncenters;
    constexpr int block = 128;
    const std::string forward = symbol + jit_geometry_forward_suffix<T>();
    const auto table_ptr = [](const BasisDevice<T>& basis) -> const T* {
        if constexpr (std::is_same_v<T, float>) return basis.values_f;
        else return basis.values;
    };
    const auto deriv_ptr = [](const BasisDevice<T>& basis) -> const T* {
        if constexpr (std::is_same_v<T, float>) return basis.derivatives_f;
        else return basis.derivatives;
    };
    const auto launch = [&](auto... args) {
        if constexpr (jit_cooperative_geometry<T>()) {
            const int edges_per_block = block / 8;
            const int blocks = rows ? (rows - 1) / edges_per_block + 1 : 0;
            w.jit->launch_grid(w.exec, forward, blocks, block, 0, rows, args...);
        } else {
            w.jit->launch(w.exec, forward, rows, block, args...);
        }
    };
    launch(
        static_cast<int>(input.spline), old_width, input.center.data(),
#ifdef SP2_ENABLE_CUDA
        input.targets.data(), input.frames.data(), input.start, static_cast<int>(input.shared_cell),
        input.positions, input.shifts.ptr, input.graph_cells, input.graph_spins,
#else
        input.pos_i.ptr, input.pos_j.ptr, input.shifts.ptr, input.cells.ptr, si.ptr, sj.ptr,
#endif
        prev.ptr, b.weights, b.mean, b.inv_std, input.r.ptr, length.ptr, value.ptr,
        slopes.ptr, spin_local.ptr, spin_values.ptr, spin_slopes.ptr, features.ptr, jacobian.ptr,
        table_ptr(b), deriv_ptr(b), table_ptr(spin_table), deriv_ptr(spin_table),
        spin_table.weights, tile_offsets.data(), ncenters);

    const bool frozen = input.frozen;
    const std::string reverse_sym = symbol + jit_geometry_reverse_suffix<T>();
    std::vector<std::string> in{field, "length", "r", "i", "j"}, outputs{out};
    std::vector<Tensor<T>> values{value};
    if (sh_width) {
        outputs.push_back(sh_out);
        values.push_back(features);
    }
    record(state, std::move(in), std::move(outputs), values,
        TilePullback<T>([=](Workspace<T>& work, const auto& grad) {
            const auto dprevious = old_width && grad[0] ? work.make(rows, old_width) : Tensor<T>{};
            const auto dl = !frozen ? work.make(rows, 1) : Tensor<T>{};
            const auto dr = sh_width && !frozen ? work.make(rows, 3) : Tensor<T>{};
            const auto di = magnetic ? work.make(rows, 3) : Tensor<T>{};
            const auto dj = magnetic ? work.make(rows, 3) : Tensor<T>{};
            if constexpr (jit_cooperative_geometry<T>()) {
                const int edges_per_block = block / 8;
                const int blocks = rows ? (rows - 1) / edges_per_block + 1 : 0;
                work.jit->launch_grid(work.exec, reverse_sym, blocks, block, 0, rows,
                    old_width, si.ptr, sj.ptr, grad[0].ptr, sh_width ? grad[1].ptr : nullptr,
                    slopes.ptr, spin_local.ptr, spin_values.ptr, spin_slopes.ptr, jacobian.ptr,
                    dprevious.ptr, dl.ptr, dr.ptr, di.ptr, dj.ptr, tile_offsets.data(), ncenters);
            } else {
                work.jit->launch(work.exec, reverse_sym, rows, 128, old_width, si.ptr, sj.ptr,
                    grad[0].ptr, sh_width ? grad[1].ptr : nullptr, slopes.ptr, spin_local.ptr,
                    spin_values.ptr, spin_slopes.ptr, jacobian.ptr, dprevious.ptr, dl.ptr, dr.ptr,
                    di.ptr, dj.ptr, tile_offsets.data(), ncenters);
            }
            return std::vector<Tensor<T>>{dprevious, dl, dr, di, dj};
        }));
}
}  // namespace sp2
