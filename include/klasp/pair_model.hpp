#pragma once

// Pair-physics interface: the short-range energy of one atom pair at separation r.
// A PairModel evaluates the untapered potential; the caller applies the taper
// (apply_taper in energy_kernels.hpp). Models are trivially copyable structs passed
// by value as template parameters, chosen on the host:
//
//     struct Model {
//         KOKKOS_INLINE_FUNCTION
//         void eval(RealType r, RealType r2, RealType inv_r, int ti, int tj,
//                   RealType& e, RealType& de_dr) const;
//         KOKKOS_INLINE_FUNCTION
//         void eval(RealType r, RealType r2, RealType inv_r, int ti, int tj,
//                   RealType& e, RealType& de_dr, RealType& d2e_dr2) const;
//     };
//
// r2 must be the caller's exact rx^2 + ry^2 + rz^2, not r*r recomputed. The
// d2e_dr2 overload is needed for the trust-region Hessian seed.

#include "klasp/energy_kernels.hpp"
#include "klasp/interp.hpp"
#include "klasp/types.hpp"

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace klasp {

// Which pair model a run evaluates. Auto uses the table when the input built one,
// else the closed form; the explicit values are for debugging and comparison.
// Resolved on the host to a template parameter (use_tabulated_pair_model).
enum class PairModelKind : uint8_t {
    Auto = 0,
    Buckingham = 1,
    Tabulated = 2,
};

// Closed-form Buckingham.
template <typename Space>
struct BuckinghamModelT {
    SView2D<RealType, Space> A, B, C, Rmin;   // (n_types, n_types)
    TangToenniesDamping damping = TangToenniesDamping::off();

    KOKKOS_INLINE_FUNCTION
    void eval(RealType r, RealType /*r2*/, RealType inv_r, int ti, int tj,
              RealType& e, RealType& de_dr) const {
        buckingham_untapered(r, inv_r, A(ti, tj), B(ti, tj), C(ti, tj),
                             damping, Rmin(ti, tj), e, de_dr);
    }

    KOKKOS_INLINE_FUNCTION
    void eval(RealType r, RealType /*r2*/, RealType inv_r, int ti, int tj,
              RealType& e, RealType& de_dr, RealType& d2e_dr2) const {
        buckingham_untapered(r, inv_r, A(ti, tj), B(ti, tj), C(ti, tj),
                             damping, Rmin(ti, tj), e, de_dr, &d2e_dr2);
    }
};

using BuckinghamModel = BuckinghamModelT<ExecSpace>;

// Tabulated: one C2 cubic spline per ordered type pair, indexed
// type_i * n_types + type_j (see interp.hpp). A pair with no table keeps a
// default RadialSpline, which returns zero everywhere.
template <typename Space>
struct TabulatedModelT {
    SView1D<RadialSpline<>, Space> splines;   // (n_types * n_types)
    int n_types = 0;

    KOKKOS_INLINE_FUNCTION
    void eval(RealType r, RealType /*r2*/, RealType inv_r, int ti, int tj,
              RealType& e, RealType& de_dr, RealType& d2e_dr2) const {
        splines(ti * n_types + tj).eval(r, inv_r, e, de_dr, d2e_dr2);
    }

    KOKKOS_INLINE_FUNCTION
    void eval(RealType r, RealType r2, RealType inv_r, int ti, int tj,
              RealType& e, RealType& de_dr) const {
        RealType d2;   // unused; the interval's second derivative folds away
        eval(r, r2, inv_r, ti, tj, e, de_dr, d2);
    }
};

using TabulatedModel = TabulatedModelT<ExecSpace>;

// The `potentials.tabulated` block. Absent means the closed form.
struct TabulatedSpec {
    bool enabled = false;
    // Knots across [r_min, r_max].
    int n_knots = 1500;
    // Below r_min the spline continues exponentially from its own value and slope,
    // so a repulsive wall stays finite and repulsive.
    double r_min = 1.0;    // Bohr
    double r_max = 0.0;    // Bohr; 0 means "the potential's cutoff"
};

// Device spline headers plus the flat coefficient array they point into. Keep
// alive while any kernel can see `splines`.
struct TabulatedPotentials {
    View1D<RealType> coeffs;
    View1D<RadialSpline<>> splines;
    int n_types = 0;
    int n_tables = 0;   // pairs that actually got one; the rest answer zero
    int n_knots = 0;
    double r_min = 0, r_max = 0;
    // Worst relative error against the tabulated function, at interval midpoints.
    double worst_rel_error = 0;
    std::string summary() const;
};

// Cross-space copy: moves the coefficients and repoints every spline. No-op when
// both spaces share memory.
template <typename Space>
void rebase_pair_tables(const SView1D<RadialSpline<>, Space>& splines,
                        const RealType* old_base, RealType* new_base) {
    if (splines.extent(0) == 0 || old_base == new_base) return;
    auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), splines);
    for (std::size_t i = 0; i < h.extent(0); ++i)
        if (h(i).c != nullptr) h(i).c = new_base + (h(i).c - old_base);
    Kokkos::deep_copy(splines, h);
}

}  // namespace klasp
