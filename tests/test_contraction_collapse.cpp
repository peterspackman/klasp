// The contracted rigid-body gradient path: each asymmetric-unit atom's force is
// contracted against a Jacobian and multiplied by the symmetry multiplicity,
// which requires every symmetry operation to be an isometry of the cell. Checks
// the isometry in every lattice type (with a negative control), the refusal of
// unequal copy counts, and the gradients against finite differences.
#include "kokkos_listener.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/sgdata.hpp"
#include "klasp/molecule.hpp"
#include "klasp/cell.hpp"
#include "klasp/rotation.hpp"
#include "klasp/multipole.hpp"
#include "test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <vector>
#include <cmath>

using namespace klasp;
using Catch::Matchers::WithinRel;

namespace {

// A cell obeying the metric constraints of each lattice type, with distinct
// lengths where allowed so no accidental higher symmetry exists.
void metric_cell(LatticeType lat, double p[6]) {
    const double a = 6.1, b = 7.3, c = 8.9;
    // LatticeType is unscoped, so zero p on every path.
    for (int i = 0; i < 6; ++i) p[i] = 0.0;
    switch (lat) {
        case TRICLINIC:    p[0]=a; p[1]=b; p[2]=c; p[3]=88;  p[4]=95; p[5]=103; break;
        case MONOCLINIC:   p[0]=a; p[1]=b; p[2]=c; p[3]=90;  p[4]=97; p[5]=90;  break;
        case ORTHORHOMBIC: p[0]=a; p[1]=b; p[2]=c; p[3]=90;  p[4]=90; p[5]=90;  break;
        case TETRAGONAL:   p[0]=a; p[1]=a; p[2]=c; p[3]=90;  p[4]=90; p[5]=90;  break;
        case RHOMBOHEDRAL: p[0]=a; p[1]=a; p[2]=a; p[3]=83;  p[4]=83; p[5]=83;  break;
        case HEXAGONAL:    p[0]=a; p[1]=a; p[2]=c; p[3]=90;  p[4]=90; p[5]=120; break;
        case CUBIC:        p[0]=a; p[1]=a; p[2]=a; p[3]=90;  p[4]=90; p[5]=90;  break;
        // An unhandled lattice type fails here rather than testing a zero cell.
        default:
            FAIL("metric_cell has no cell for lattice type "
                 << static_cast<int>(lat));
            break;
    }
}

// Largest element of |S^T S - I| over every symmetry operation of `sg`, with the
// cell built from `params`. Zero means every operation is an isometry.
double worst_orthogonality_error(int sg_number, const double params[6]) {
    const auto sg = get_space_group(sg_number);
    RealType cp[6];
    for (int i = 0; i < 6; ++i) cp[i] = RealType(params[i]);
    RealType cell[3][3], cell_inv[3][3] = {};
    cell_params_to_matrix(cp, cell);
    mat3_inv(cell, cell_inv);

    double worst = 0.0;
    for (int op = 0; op < sg.n_symops; ++op) {
        RealType S_frac[3][3], S_cart[3][3];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                S_frac[r][c] = RealType(sg.symops[op].rot[r][c]);
        frac_to_cart_symop(cell, cell_inv, S_frac, S_cart);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                double d = 0;
                for (int k = 0; k < 3; ++k) d += double(S_cart[k][r]) * double(S_cart[k][c]);
                worst = std::max(worst, std::abs(d - (r == c ? 1.0 : 0.0)));
            }
    }
    return worst;
}

}  // namespace

TEST_CASE("symmetry operations are isometries in every lattice type",
          "[contraction][chain]") {
    // The fine25 search set plus a cubic group, covering all seven lattice types.
    const int groups[] = {1, 2, 4, 5, 7, 9, 13, 14, 15, 18, 19, 20, 29, 33, 43,
                          56, 60, 61, 76, 86, 88, 96, 145, 148, 169, 205};
    bool saw[7] = {false, false, false, false, false, false, false};

    for (int n : groups) {
        const auto sg = get_space_group(n);
        double p[6];
        metric_cell(sg.lattice, p);
        const double err = worst_orthogonality_error(n, p);
        INFO("space group " << n << " (" << lattice_name(sg.lattice)
             << ", " << sg.n_symops << " symops), worst |S^T S - I| = " << err);
        REQUIRE(err < 1e-12);
        saw[static_cast<int>(sg.lattice)] = true;
    }

    // The claim is about every lattice type, so the test has to have seen them.
    for (int i = 0; i < 7; ++i) {
        INFO("no space group in the list reported lattice type " << i);
        REQUIRE(saw[i]);
    }
}

TEST_CASE("a cell off its lattice metric breaks the isometry",
          "[contraction][chain]") {
    // Negative control: R-3 in a triclinic cell, where the three-fold axis is not
    // an isometry.
    double bad[6];
    metric_cell(TRICLINIC, bad);
    const double err = worst_orthogonality_error(148, bad);
    INFO("worst |S^T S - I| on a deliberately wrong cell = " << err);
    REQUIRE(err > 1e-3);
}

// Molecules with unequal numbers of copies (e.g. one on a special position) are
// refused: the asymmetric-unit pair enumeration needs equal copy counts for
// cross-molecule pairs to have a single multiplicity.
TEST_CASE("a structure with unequal copy counts is refused, not miscomputed",
          "[contraction][chain][zprime]") {
    const std::vector<int> sym_map = {0, 1, 0, 1, 0, 1, 0};   // 4 copies vs 3
    auto f = testing::make_fixture_zprime(2, 3, 6.4, 7.2, 8.1, 96.0, sym_map);
    REQUIRE_THROWS_AS(validate_symmetry_multiplicity(f.mol), std::runtime_error);

    // An even split is accepted, so the check is about the counts.
    auto ok = testing::make_fixture_zprime(2, 3, 6.4, 7.2, 8.1, 96.0);
    REQUIRE_NOTHROW(validate_symmetry_multiplicity(ok.mol));
}

// Self is zero atom by atom (|mu_a|^2 is rotation-invariant); intra cancels
// within each rigid molecule. Each half alone is a noticeable fraction of the
// gradient, so dropping one without the other is a silent error.
TEST_CASE("self and intra add nothing to the rigid-body gradient",
          "[contraction][zeroterms]") {
    const int n_indep = 2, n_sites = 4;

    auto grad_with = [&](uint32_t components) {
        auto f = testing::make_fixture_zprime(n_indep, n_sites, 6.4, 7.2, 8.1, 96.0, {}, 4);
        f.settings.model.ewald_components = components;
        batch_expand_asymmetric_unit<4>(f.bws.tables, f.bws.geom, f.bws.dofs);
        batch_energy_gradient_asymunit<4>(f.bws.tables, f.bws.geom, f.bws.dofs, f.settings, 0);
        auto gt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f.bws.dofs.d_gt);
        auto gr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f.bws.dofs.d_gr);
        double worst = 0;
        for (int k = 0; k < n_indep; ++k)
            for (int d = 0; d < 3; ++d)
                worst = std::max(worst, std::max(std::abs(double(gt(0,k,d))),
                                                 std::abs(double(gr(0,k,d)))));
        return worst;
    };

    // Each alone, so nothing else can mask it.
    const double self_only  = grad_with(EWALD_SELF);
    const double intra_only = grad_with(EWALD_INTRA);
    // For scale: what a real gradient looks like on this fixture.
    const double full = grad_with(EWALD_BUCKINGHAM | EWALD_PAIR_ERFC | EWALD_MULTIPOLE_H);

    INFO("self " << self_only << "  intra " << intra_only << "  full " << full);
    REQUIRE(full > 1e-3);                    // the comparison means something
    REQUIRE(self_only  < 1e-12 * std::max(1.0, full));
    REQUIRE(intra_only < 1e-12 * std::max(1.0, full));
}

// More than one molecule and more than one copy of each is where a symmetry
// multiplicity error would show.
TEST_CASE("Z' > 1 rigid-body gradients match finite differences at rank 4",
          "[contraction][zprime][fd]") {
    const int n_indep = 2, n_sites = 4;
    auto f = testing::make_fixture_zprime(n_indep, n_sites, 6.4, 7.2, 8.1, 96.0, {}, 4);
    REQUIRE(f.mol.n_symops > n_indep);      // there are copies
    REQUIRE(n_indep > 1);                   // and more than one molecule

    auto energy_now = [&]() {
        batch_expand_asymmetric_unit<4>(f.bws.tables, f.bws.geom, f.bws.dofs);
        batch_energy_gradient_asymunit<4>(f.bws.tables, f.bws.geom, f.bws.dofs, f.settings, 0);
        auto hE = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f.bws.geom.energies);
        return double(hE(0));
    };
    energy_now();
    auto h_gt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f.bws.dofs.d_gt);
    auto h_gr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f.bws.dofs.d_gr);
    std::vector<double> gt, gr;
    for (int k = 0; k < n_indep; ++k)
        for (int d = 0; d < 3; ++d) {
            gt.push_back(double(h_gt(0,k,d)));
            gr.push_back(double(h_gr(0,k,d)));
        }

    auto central = [&](auto& view, int k, int d, double h) {
        auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
        const RealType saved = hv(0,k,d);
        hv(0,k,d) = saved + RealType(h); Kokkos::deep_copy(view, hv);
        const double ep = energy_now();
        hv(0,k,d) = saved - RealType(h); Kokkos::deep_copy(view, hv);
        const double em = energy_now();
        hv(0,k,d) = saved;               Kokkos::deep_copy(view, hv);
        energy_now();
        return (ep - em) / (2.0 * h);
    };

    for (int k = 0; k < n_indep; ++k)
        for (int d = 0; d < 3; ++d) {
            INFO("translation, molecule " << k << " component " << d);
            REQUIRE_THAT(gt[k*3+d], WithinRel(central(f.bws.dofs.translations, k, d, 2e-5), 1e-5));
            INFO("rotation, molecule " << k << " component " << d);
            REQUIRE_THAT(gr[k*3+d], WithinRel(central(f.bws.dofs.rot_vectors, k, d, 2e-5), 1e-5));
        }
}
