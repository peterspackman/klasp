#pragma once

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace klasp {

#ifdef KLASP_SINGLE_PRECISION
using RealType = float;
#else
using RealType = double;
#endif
using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = ExecSpace::memory_space;
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using HostMemSpace = HostExecSpace::memory_space;

template <typename T>
using View1D = Kokkos::View<T*, MemSpace>;

template <typename T>
using View2D = Kokkos::View<T**, Kokkos::LayoutRight, MemSpace>;

template <typename T>
using View3D = Kokkos::View<T***, Kokkos::LayoutRight, MemSpace>;

// Space-parameterized views for hybrid GPU+CPU execution
template <typename T, typename Space>
using SView1D = Kokkos::View<T*, typename Space::memory_space>;

template <typename T, typename Space>
using SView2D = Kokkos::View<T**, Kokkos::LayoutRight, typename Space::memory_space>;

template <typename T, typename Space>
using SView3D = Kokkos::View<T***, Kokkos::LayoutRight, typename Space::memory_space>;

// ---------------------------------------------------------------------------
// Host <-> device marshalling. copy_to_view fills and pushes only the first
// src.size() rows, leaving any padded tail of the view untouched.
// ---------------------------------------------------------------------------

// A host mirror of `v` holding its data. On a host-space build this aliases `v`,
// so writing through the result may write through to `v`.
template <typename V>
inline auto to_host(const V& v) {
    return Kokkos::create_mirror_view_and_copy(HostMemSpace(), v);
}

// Rank-1 view -> std::vector.
template <typename V>
inline std::vector<typename V::non_const_value_type> to_host_vector(const V& v) {
    static_assert(V::rank == 1, "to_host_vector takes a rank-1 view");
    auto h = to_host(v);
    std::vector<typename V::non_const_value_type> out(v.extent(0));
    for (std::size_t i = 0; i < v.extent(0); ++i) out[i] = h(i);
    return out;
}

namespace detail {
template <typename V>
inline void check_fits(const V& dst, std::size_t n) {
    if (n > dst.extent(0))
        throw std::runtime_error("copy_to_view: source of " + std::to_string(n) +
                                 " does not fit view '" + std::string(dst.label()) +
                                 "' of extent " + std::to_string(dst.extent(0)));
}
}  // namespace detail

// std::vector -> the first src.size() entries of an existing rank-1 view.
template <typename V, typename T>
inline void copy_to_view(const V& dst, const std::vector<T>& src) {
    static_assert(V::rank == 1, "this copy_to_view takes a rank-1 view");
    const std::size_t n = src.size();
    detail::check_fits(dst, n);
    if (n == 0) return;
    using value_type = typename V::non_const_value_type;
    auto sub = Kokkos::subview(dst, Kokkos::make_pair(std::size_t(0), n));
    auto h = Kokkos::create_mirror_view(sub);
    for (std::size_t i = 0; i < n; ++i) h(i) = static_cast<value_type>(src[i]);
    Kokkos::deep_copy(sub, h);
}

// std::vector<std::array<T, N>> -> the first src.size() rows of a rank-2 view.
template <typename V, typename T, std::size_t N>
inline void copy_to_view(const V& dst, const std::vector<std::array<T, N>>& src) {
    static_assert(V::rank == 2, "this copy_to_view takes a rank-2 view");
    const std::size_t n = src.size();
    detail::check_fits(dst, n);
    if (dst.extent(1) != N)
        throw std::runtime_error("copy_to_view: row width mismatch for view '" +
                                 std::string(dst.label()) + "'");
    if (n == 0) return;
    using value_type = typename V::non_const_value_type;
    auto sub = Kokkos::subview(dst, Kokkos::make_pair(std::size_t(0), n), Kokkos::ALL);
    auto h = Kokkos::create_mirror_view(sub);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t c = 0; c < N; ++c) h(i, c) = static_cast<value_type>(src[i][c]);
    Kokkos::deep_copy(sub, h);
}

// std::vector -> a newly allocated device view of exactly its size.
template <typename T>
inline View1D<T> to_view(const std::string& label, const std::vector<T>& src) {
    View1D<T> v(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), src.size());
    copy_to_view(v, src);
    return v;
}

template <typename T, std::size_t N>
inline View2D<T> to_view(const std::string& label,
                         const std::vector<std::array<T, N>>& src) {
    View2D<T> v(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), src.size(), N);
    copy_to_view(v, src);
    return v;
}

// Pre-allocated workspace for a single structure evaluation, allocated once
// with max sizes and reused.
struct Workspace {
    View2D<RealType> positions;     // (max_atoms, 3)
    View1D<RealType> charges;       // (max_atoms)
    View1D<int>      types;         // (max_atoms)
    View2D<RealType> cell_matrix;   // (3, 3)

    View2D<int>      shifts;        // (max_shifts, 3)
    View2D<RealType> shift_cart;    // (max_shifts, 3) precomputed Cartesian shifts

    View2D<RealType> kvecs;         // (max_kvecs, 3)
    View1D<RealType> kprefactors;   // (max_kvecs)

    View2D<RealType> buck_A;        // (max_types, max_types)
    View2D<RealType> buck_B;
    View2D<RealType> buck_C;
    View2D<RealType> buck_Rmin;     // Buckingham minimum distance (0 = no cap)

    // The tabulated pair potential, or null for the closed form. An incomplete
    // type so types.hpp need not know RadialSpline (defined in pair_model.hpp).
    std::shared_ptr<struct TabulatedPotentials> pair_tables;

    // Pairs (i,j) with the same mol_id at zero shift are intramolecular and
    // skipped; -1 disables exclusion.
    View1D<int>      mol_id;        // (max_atoms)

    View2D<RealType> multipoles;    // (max_atoms, 35) Cartesian multipoles in lab frame
    View2D<RealType> dipoles;       // (max_atoms, 3) Cartesian dipoles (x,y,z) for Ewald
    int max_multipole_rank;         // 0 = charge-only (default), 4 = full

    // Actual sizes for the current structure
    int n_atoms;
    int n_shifts;
    int n_kvecs;
    int n_types;

    RealType kappa;
    RealType rcut;
    RealType buck_cutoff;
    // Atom type index reserved for padded (ghost) sites, or -1 if none.
    int ghost_type = -1;
    // Relative truncation target for both Ewald sums (see ewald_split_width).
    RealType ewald_accuracy;
};

Workspace allocate_workspace(int max_atoms, int max_shifts,
                             int max_kvecs, int max_types,
                             int max_multipole_rank = 0);

}  // namespace klasp
