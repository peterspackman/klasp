#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <vector>

#include "kokkos_listener.hpp"
#include "klasp/crystal_dof.hpp"
#include "klasp/device_optim.hpp"

using namespace klasp::crystdof;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// An anisotropic, sheared reference cell (columns = lattice vectors, row-major)
// so the fractional reconditioning (h0^T) is non-trivial.
constexpr double H0[9] = {
    8.0, 1.0, 0.5,
    0.0, 6.0, 1.2,
    0.0, 0.0, 4.0,
};

// Toy analytic energy: springs from each atom to a fixed Cartesian center.
//   E = 1/2 sum_i k |r_i - c_i|^2,  dE/dr_i = k (r_i - c_i),  F_i = -dE/dr_i.
// The virial (affine derivative at fixed fractional coords) is
//   W_ab = dE/deps_ab = sum_i (dE/dr_i)_a * r_i,b   (symmetrized to Voigt).
struct Toy {
    double k = 0.7;
    std::vector<std::array<double, 3>> c;  // fixed centers, Cartesian

    double energy(const std::vector<std::array<double, 3>>& r) const {
        double e = 0;
        for (size_t i = 0; i < r.size(); ++i)
            for (int a = 0; a < 3; ++a) {
                const double d = r[i][a] - c[i][a];
                e += 0.5 * k * d * d;
            }
        return e;
    }
    // PET-convention force F = -dE/dr.
    std::array<double, 3> force(const std::array<double, 3>& ri, int i) const {
        return {-k * (ri[0] - c[i][0]), -k * (ri[1] - c[i][1]), -k * (ri[2] - c[i][2])};
    }
};

// Rotationally invariant pairwise springs: unlike Toy, a symmetric virial, as
// the strain fold assumes.
struct PairSpring {
    double k = 0.5, r0 = 3.0;
    double energy(const std::vector<std::array<double, 3>>& r) const {
        double e = 0;
        for (size_t i = 0; i < r.size(); ++i)
            for (size_t j = i + 1; j < r.size(); ++j) {
                double d2 = 0;
                for (int c = 0; c < 3; ++c) { const double dd = r[i][c] - r[j][c]; d2 += dd * dd; }
                const double d = std::sqrt(d2);
                e += 0.5 * k * (d - r0) * (d - r0);
            }
        return e;
    }
    // PET-convention force F = -dE/dr_i.
    std::array<double, 3> force(const std::vector<std::array<double, 3>>& r, int i) const {
        std::array<double, 3> F{0, 0, 0};
        for (size_t j = 0; j < r.size(); ++j) {
            if ((int)j == i) continue;
            double dv[3], d2 = 0;
            for (int c = 0; c < 3; ++c) { dv[c] = r[i][c] - r[j][c]; d2 += dv[c] * dv[c]; }
            const double d = std::sqrt(d2);
            if (d < 1e-9) continue;
            const double coef = k * (d - r0) / d;  // dE/dr_i contribution coefficient
            for (int c = 0; c < 3; ++c) F[c] -= coef * dv[c];
        }
        return F;
    }
};

std::vector<std::array<double, 3>> cart_positions(const std::vector<double>& u,
                                                  int N, double lambda,
                                                  const double eps[6]) {
    double h[9];
    current_cell(H0, eps, h);
    std::vector<std::array<double, 3>> r(N);
    for (int i = 0; i < N; ++i) {
        double ri[3];
        cart_from_dof(h, lambda, &u[3 * i], ri);
        r[i] = {ri[0], ri[1], ri[2]};
    }
    return r;
}

}  // namespace

TEST_CASE("current_cell = (I+E) h0 and matches hand strain", "[crystdof]") {
    const double eps[6] = {0.02, -0.01, 0.03, 0.005, 0.0, -0.004};
    double h[9];
    current_cell(H0, eps, h);
    // Reconstruct (I+E) h0 by hand.
    double def[9];
    strain_to_def(eps, def);
    double ref[9];
    mat3_mul(def, H0, ref);
    for (int i = 0; i < 9; ++i) REQUIRE_THAT(h[i], WithinAbs(ref[i], 1e-13));
}

TEST_CASE("fractional round-trip: s = h0^{-1} p recovers p at zero strain", "[crystdof]") {
    const double lambda = precond_length(H0);
    const double eps[6] = {0, 0, 0, 0, 0, 0};
    // Pick a Cartesian position p, get its fractional s = h0^{-1} p, build the DOF
    // u = lambda*s, then cart_from_dof must return p exactly.
    const double p[3] = {3.1, -2.0, 1.7};
    double h0inv[9];
    REQUIRE(mat3_inv(H0, h0inv));
    double s[3];
    mat3_vec(h0inv, p, s);
    const double u[3] = {lambda * s[0], lambda * s[1], lambda * s[2]};
    double h[9];
    current_cell(H0, eps, h);
    double r[3];
    cart_from_dof(h, lambda, u, r);
    for (int c = 0; c < 3; ++c) REQUIRE_THAT(r[c], WithinAbs(p[c], 1e-11));
}

TEST_CASE("atom_grad matches finite-difference dE/du at fixed strain", "[crystdof]") {
    const int N = 3;
    const double lambda = precond_length(H0);
    const double eps[6] = {0.01, 0.0, -0.02, 0.003, 0.001, 0.0};

    Toy toy;
    toy.c = {{1.0, 0.5, -0.3}, {-2.0, 1.1, 0.9}, {0.4, -1.7, 2.2}};

    // Arbitrary fractional DOFs.
    std::vector<double> u = {lambda * 0.10, lambda * 0.20, lambda * 0.05,
                             lambda * 0.55, lambda * 0.40, lambda * 0.70,
                             lambda * 0.85, lambda * 0.15, lambda * 0.33};

    double h[9];
    current_cell(H0, eps, h);
    auto r = cart_positions(u, N, lambda, eps);

    // Analytic atomic gradient via the primitive.
    std::vector<double> g(3 * N, 0.0);
    for (int i = 0; i < N; ++i) {
        auto F = toy.force(r[i], i);
        double Fi[3] = {F[0], F[1], F[2]};
        atom_grad(h, lambda, Fi, &g[3 * i]);
    }

    // Central finite difference of E wrt each u component.
    const double dlt = 1e-6;
    for (int j = 0; j < 3 * N; ++j) {
        auto up = u, um = u;
        up[j] += dlt; um[j] -= dlt;
        const double ep = toy.energy(cart_positions(up, N, lambda, eps));
        const double em = toy.energy(cart_positions(um, N, lambda, eps));
        const double fd = (ep - em) / (2 * dlt);
        REQUIRE_THAT(g[j], WithinAbs(fd, 1e-6));
    }
}

TEST_CASE("strain_grad matches finite-difference dE/deps at fixed fractional", "[crystdof]") {
    const int N = 3;
    const double lambda = precond_length(H0);
    const double cell_factor = static_cast<double>(N);
    const double eps0[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    Toy toy;
    toy.c = {{1.0, 0.5, -0.3}, {-2.0, 1.1, 0.9}, {0.4, -1.7, 2.2}};

    std::vector<double> u = {lambda * 0.10, lambda * 0.20, lambda * 0.05,
                             lambda * 0.55, lambda * 0.40, lambda * 0.70,
                             lambda * 0.85, lambda * 0.15, lambda * 0.33};

    // Analytic virial W_ab = sum_i (dE/dr_i)_a r_i,b, symmetrized -> Voigt.
    double h[9];
    current_cell(H0, eps0, h);
    auto r = cart_positions(u, N, lambda, eps0);
    double Wm[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < N; ++i) {
        auto F = toy.force(r[i], i);
        const double dEdr[3] = {-F[0], -F[1], -F[2]};  // dE/dr = -F
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) Wm[a * 3 + b] += dEdr[a] * r[i][b];
    }
    // Pack Voigt [xx,yy,zz,xy,xz,yz] in the full tensor-virial convention (each
    // shear eps in both off-diagonal slots, so dE/deps_xy = Wm[1]+Wm[3]), as PET
    // returns and CellState uses.
    const double W[6] = {Wm[0], Wm[4], Wm[8],
                         Wm[1] + Wm[3],
                         Wm[2] + Wm[6],
                         Wm[5] + Wm[7]};

    const double vol = std::fabs(mat3_det(h));
    double g6[6];
    strain_grad(W, cell_factor, vol, g6);  // dE/d(eps*cf) = W/cf

    // Central FD of E wrt each Voigt strain component, fractional coords fixed.
    const double dlt = 1e-6;
    const int vi[6] = {0, 1, 2, 3, 4, 5};
    for (int k = 0; k < 6; ++k) {
        double epsp[6] = {0, 0, 0, 0, 0, 0}, epsm[6] = {0, 0, 0, 0, 0, 0};
        epsp[vi[k]] = dlt; epsm[vi[k]] = -dlt;
        const double ep = toy.energy(cart_positions(u, N, lambda, epsp));
        const double em = toy.energy(cart_positions(u, N, lambda, epsm));
        const double fd = (ep - em) / (2 * dlt);   // dE/deps_k
        // g6[k] is dE/d(eps_k*cf) = (dE/deps_k)/cf.
        REQUIRE_THAT(g6[k] * cell_factor, WithinAbs(fd, 1e-5));
    }
}

// The symmetry force-fold g_qa = -sum_op (def M0_op)^T F is the exact chain-rule
// gradient of E(reconstruct(q)) for any energy, so it must match FD through
// improper operators (det = -1) too.

// A SymLayout for the uniform case: asym_off[k] == k*ns, full_off[op] == op*ns,
// op_of_atom[il] == il/ns.
struct UniformLayout {
    std::vector<int> sites, asym_off, full_off, op_of_atom;
    UniformLayout(int n_indep, int no, int ns) {
        sites.assign(n_indep, ns);
        asym_off.resize(n_indep + 1);
        for (int k = 0; k <= n_indep; ++k) asym_off[k] = k * ns;
        full_off.resize(no + 1);
        for (int op = 0; op <= no; ++op) full_off[op] = op * ns;
        op_of_atom.resize(no * ns);
        for (int il = 0; il < no * ns; ++il) op_of_atom[il] = il / ns;
    }
    klasp::devopt::SymLayout get() const {
        return klasp::devopt::SymLayout{static_cast<int>(sites.size()), sites.data(),
                                        asym_off.data(), full_off.data(), op_of_atom.data()};
    }
};

TEST_CASE("sym_fold atomic gradient matches FD through an inversion operator", "[crystdof][sym]") {
    using klasp::devopt::sym_expand;
    using klasp::devopt::sym_fold;
    const int ns = 2;   // asym-unit atoms (Z'=1)
    const int no = 3;   // identity, inversion (improper), 2-fold about z (proper)
    const int nasym = ns;
    const int nf = 3 * nasym + 6;

    const double Rf[3][9] = {
        {1, 0, 0, 0, 1, 0, 0, 0, 1},        // identity
        {-1, 0, 0, 0, -1, 0, 0, 0, -1},     // inversion (det -1)
        {-1, 0, 0, 0, -1, 0, 0, 0, 1},      // 2-fold about z (det +1)
    };
    double h0inv[9];
    REQUIRE(mat3_inv(H0, h0inv));
    std::vector<double> M0(no * 9), Rall(no * 9), tc(no * 3, 0.0);
    for (int op = 0; op < no; ++op) {
        double t[9];
        mat3_mul(H0, Rf[op], t);
        mat3_mul(t, h0inv, &M0[op * 9]);  // M0_op = h0 R_op h0^-1 (Cartesian)
        for (int i = 0; i < 9; ++i) Rall[op * 9 + i] = Rf[op][i];
    }
    tc[3] = 0.3; tc[4] = -0.2; tc[5] = 0.15;  // Cartesian translations
    tc[6] = -0.4; tc[7] = 0.25; tc[8] = 0.10;
    std::vector<int> indep(no, 0);
    const UniformLayout LO(1, no, ns);

    const double eps[6] = {0.01, -0.005, 0.02, 0.004, -0.002, 0.001};
    double def[9];
    strain_to_def(eps, def);

    std::vector<double> xb = {1.2, -0.7, 0.4, -0.9, 1.1, -1.3, 0, 0, 0, 0, 0, 0};

    auto reconstruct = [&](const std::vector<double>& x) {
        std::vector<std::array<double, 3>> r(no * ns);
        for (int il = 0; il < no * ns; ++il) {
            double ri[3];
            sym_expand(def, x.data(), il, LO.get(), M0.data(), tc.data(), indep.data(), ri);
            r[il] = {ri[0], ri[1], ri[2]};
        }
        return r;
    };

    Toy toy;
    toy.c = {{0.5, 0.2, -0.1}, {-0.3, 0.8, 0.6}, {1.0, -0.4, 0.2},
             {-0.7, 0.1, -0.5}, {0.2, 0.9, -0.3}, {-0.6, -0.2, 0.7}};

    auto r = reconstruct(xb);
    std::vector<double> Fraw(no * ns * 3);
    for (int il = 0; il < no * ns; ++il) {
        auto F = toy.force(r[il], il);
        Fraw[3 * il] = F[0]; Fraw[3 * il + 1] = F[1]; Fraw[3 * il + 2] = F[2];
    }
    double Wdummy[6] = {0, 0, 0, 0, 0, 0};
    std::vector<double> grad(nf, 0.0);
    double maxF = 0, ms = 0;
    sym_fold(def, Fraw.data(), 1.0, Wdummy, 1.0, no, LO.get(), M0.data(), tc.data(), Rall.data(),
             indep.data(), H0, 3 * nasym, static_cast<double>(nasym), grad.data(), maxF, ms);

    const double dlt = 1e-6;
    for (int j = 0; j < 3 * nasym; ++j) {
        auto xp = xb, xm = xb;
        xp[j] += dlt; xm[j] -= dlt;
        const double ep = toy.energy(reconstruct(xp));
        const double em = toy.energy(reconstruct(xm));
        const double fd = (ep - em) / (2 * dlt);
        REQUIRE_THAT(grad[j], WithinAbs(fd, 1e-6));
    }
}

// Strain fold through an inversion. For Ci, Wsym = (1/2)(Wm + (-I)Wm(-I)^T) = Wm,
// so the symmetrized strain gradient must equal the raw dE/deps FD everywhere.
TEST_CASE("sym_fold strain gradient matches FD for a Ci (inversion) group", "[crystdof][sym]") {
    using klasp::devopt::sym_expand;
    using klasp::devopt::sym_fold;
    const int ns = 2, no = 2, nasym = ns, nf = 3 * nasym + 6;
    const double Rf[2][9] = {{1, 0, 0, 0, 1, 0, 0, 0, 1}, {-1, 0, 0, 0, -1, 0, 0, 0, -1}};
    double h0inv[9];
    REQUIRE(mat3_inv(H0, h0inv));
    std::vector<double> M0(no * 9), Rall(no * 9), tc(no * 3, 0.0);
    for (int op = 0; op < no; ++op) {
        double t[9];
        mat3_mul(H0, Rf[op], t);
        mat3_mul(t, h0inv, &M0[op * 9]);
        for (int i = 0; i < 9; ++i) Rall[op * 9 + i] = Rf[op][i];
    }
    tc[3] = 0.3; tc[4] = -0.2; tc[5] = 0.15;
    std::vector<int> indep(no, 0);
    const UniformLayout LO(1, no, ns);
    const double cellfac = static_cast<double>(nasym);
    std::vector<double> xb = {1.2, -0.7, 0.4, -0.9, 1.1, -1.3, 0, 0, 0, 0, 0, 0};
    PairSpring toy;

    auto reconstruct = [&](const double eps[6]) {
        double def[9];
        strain_to_def(eps, def);
        std::vector<std::array<double, 3>> r(no * ns);
        for (int il = 0; il < no * ns; ++il) {
            double ri[3];
            sym_expand(def, xb.data(), il, LO.get(), M0.data(), tc.data(), indep.data(), ri);
            r[il] = {ri[0], ri[1], ri[2]};
        }
        return r;
    };
    const double eps0[6] = {0.01, -0.005, 0.02, 0.004, -0.002, 0.001};
    double def0[9];
    strain_to_def(eps0, def0);
    auto r = reconstruct(eps0);
    std::vector<double> Fraw(no * ns * 3);
    double Wm[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int il = 0; il < no * ns; ++il) {
        auto F = toy.force(r, il);
        Fraw[3 * il] = F[0]; Fraw[3 * il + 1] = F[1]; Fraw[3 * il + 2] = F[2];
        const double dEdr[3] = {-F[0], -F[1], -F[2]};
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) Wm[a * 3 + b] += dEdr[a] * r[il][b];
    }
    // Raw (full-tensor Voigt) virial straight from the toy forces. sym_fold applies
    // the def^-T cumulative-strain correction internally.
    const double Wraw[6] = {Wm[0], Wm[4], Wm[8], Wm[1] + Wm[3], Wm[2] + Wm[6], Wm[5] + Wm[7]};
    std::vector<double> grad(nf, 0.0);
    double maxF = 0, ms = 0;
    sym_fold(def0, Fraw.data(), 1.0, Wraw, 1.0, no, LO.get(), M0.data(), tc.data(), Rall.data(),
             indep.data(), H0, 3 * nasym, cellfac, grad.data(), maxF, ms);

    const double dlt = 1e-6;
    for (int k = 0; k < 6; ++k) {
        double epsp[6], epsm[6];
        for (int i = 0; i < 6; ++i) { epsp[i] = eps0[i]; epsm[i] = eps0[i]; }
        epsp[k] += dlt; epsm[k] -= dlt;
        const double ep = toy.energy(reconstruct(epsp));
        const double em = toy.energy(reconstruct(epsm));
        const double fd = (ep - em) / (2 * dlt);
        REQUIRE_THAT(grad[3 * nasym + k] * cellfac, WithinAbs(fd, 1e-5));
    }
}

// Free-path (CellState / FreeDof) strain gradient at NONZERO cumulative strain.
// strain_grad alone (W/cf) is only correct at def=I; strain_correct_virial adds
// the def^-T factor the accumulating-strain DOF needs.
TEST_CASE("strain_correct_virial + strain_grad match FD at nonzero strain", "[crystdof][sym]") {
    const int N = 3;
    const double lambda = precond_length(H0);
    const double cell_factor = static_cast<double>(N);
    const double eps0[6] = {0.015, -0.008, 0.02, 0.005, -0.003, 0.006};
    double def[9];
    strain_to_def(eps0, def);

    PairSpring toy;
    std::vector<double> u = {lambda * 0.10, lambda * 0.20, lambda * 0.05,
                             lambda * 0.55, lambda * 0.40, lambda * 0.70,
                             lambda * 0.85, lambda * 0.15, lambda * 0.33};

    double h[9];
    current_cell(H0, eps0, h);
    auto r = cart_positions(u, N, lambda, eps0);
    double Wm[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < N; ++i) {
        auto F = toy.force(r, i);
        const double dEdr[3] = {-F[0], -F[1], -F[2]};
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) Wm[a * 3 + b] += dEdr[a] * r[i][b];
    }
    const double W[6] = {Wm[0], Wm[4], Wm[8], Wm[1] + Wm[3], Wm[2] + Wm[6], Wm[5] + Wm[7]};
    double Wc[6];
    strain_correct_virial(W, def, Wc);
    const double vol = std::fabs(mat3_det(h));
    double g6[6];
    strain_grad(Wc, cell_factor, vol, g6);

    const double dlt = 1e-6;
    for (int k = 0; k < 6; ++k) {
        double epsp[6], epsm[6];
        for (int i = 0; i < 6; ++i) { epsp[i] = eps0[i]; epsm[i] = eps0[i]; }
        epsp[k] += dlt; epsm[k] -= dlt;
        const double ep = toy.energy(cart_positions(u, N, lambda, epsp));
        const double em = toy.energy(cart_positions(u, N, lambda, epsm));
        const double fd = (ep - em) / (2 * dlt);
        REQUIRE_THAT(g6[k] * cell_factor, WithinAbs(fd, 1e-5));
    }
}

// The line-search step cap (devopt::step_cap) under both coordinate metrics.
TEST_CASE("step_cap bounds atomic and strain steps", "[crystdof]") {
    using klasp::devopt::free_cap;
    using klasp::devopt::step_cap;
    using klasp::devopt::sym_cap;
    const int N = 2, nf = 3 * N;
    const double cf = 2.0;
    // Atomic block: one atom proposes a 0.9-Bohr move, cap at 0.3 -> scale 1/3.
    std::vector<double> dir = {0.0, 0.0, 0.0,  0.9, 0.0, 0.0,
                               /*eps*cf*/ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double sc = step_cap(free_cap(dir.data(), nf), dir.data(), nf, 6, cf, 0.30, 0.02);
    REQUIRE_THAT(sc, WithinRel(0.30 / 0.9, 1e-12));
    // Same answer under the per-atom-norm metric here, the move being axial.
    sc = step_cap(sym_cap(dir.data(), N), dir.data(), nf, 6, cf, 0.30, 0.02);
    REQUIRE_THAT(sc, WithinRel(0.30 / 0.9, 1e-12));

    // Strain dominates: eps_xx*cf = 0.2 -> eps=0.1, cap 0.02 -> scale 0.2.
    dir = {0.0, 0.0, 0.0, 0.05, 0.0, 0.0,  0.2, 0.0, 0.0, 0.0, 0.0, 0.0};
    sc = step_cap(free_cap(dir.data(), nf), dir.data(), nf, 6, cf, 0.30, 0.02);
    // atomic max 0.05 < 0.30 (no cap); strain eps=0.1 -> 0.02/0.1 = 0.2.
    REQUIRE_THAT(sc, WithinRel(0.2, 1e-12));

    // The two metrics differ: a diagonal move of 0.2 per component is 0.2 by
    // component and 0.2*sqrt(3) by norm, so only the symmetric metric caps it.
    dir = {0.0, 0.0, 0.0,  0.2, 0.2, 0.2,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    REQUIRE_THAT(step_cap(free_cap(dir.data(), nf), dir.data(), nf, 6, cf, 0.30, 0.02),
                 WithinRel(1.0, 1e-12));
    REQUIRE_THAT(step_cap(sym_cap(dir.data(), N), dir.data(), nf, 6, cf, 0.30, 0.02),
                 WithinRel(0.30 / (0.2 * std::sqrt(3.0)), 1e-12));

    // nstrain = 0 is the fixed-cell case: the strain block is not even read.
    REQUIRE_THAT(step_cap(free_cap(dir.data(), nf), dir.data(), nf, 0, cf, 0.30, 0.02),
                 WithinRel(1.0, 1e-12));
}

// ---------------------------------------------------------------------------
// Ragged layout: independent molecules of different sizes, atoms compacted, so
// offsets are not multiples of a common stride.
// ---------------------------------------------------------------------------

// Two independent molecules, 2 and 3 atoms, each generated by identity and by
// inversion: asym = 5 atoms, full cell = 10, and no offset in the problem is a
// multiple of a common stride.
struct RaggedLayout {
    std::vector<int> sites, asym_off, full_off, op_of_atom;
    RaggedLayout(const std::vector<int>& sites_of_indep, const std::vector<int>& indep_of_op)
        : sites(sites_of_indep) {
        const int ni = static_cast<int>(sites.size());
        const int no = static_cast<int>(indep_of_op.size());
        asym_off.assign(ni + 1, 0);
        for (int k = 0; k < ni; ++k) asym_off[k+1] = asym_off[k] + sites[k];
        full_off.assign(no + 1, 0);
        for (int op = 0; op < no; ++op) full_off[op+1] = full_off[op] + sites[indep_of_op[op]];
        op_of_atom.assign(full_off.back(), 0);
        for (int op = 0; op < no; ++op)
            for (int i = full_off[op]; i < full_off[op+1]; ++i) op_of_atom[i] = op;
    }
    klasp::devopt::SymLayout get() const {
        return klasp::devopt::SymLayout{static_cast<int>(sites.size()), sites.data(),
                                        asym_off.data(), full_off.data(), op_of_atom.data()};
    }
};

TEST_CASE("Ragged offsets describe unequal components", "[crystdof][sym][ragged]") {
    const RaggedLayout L({2, 3}, {0, 1, 0, 1});
    REQUIRE(L.asym_off == std::vector<int>{0, 2, 5});
    REQUIRE(L.full_off == std::vector<int>{0, 2, 5, 7, 10});
    REQUIRE(L.op_of_atom == std::vector<int>{0, 0, 1, 1, 1, 2, 2, 3, 3, 3});
}

// With equal components the ragged offsets reduce to a plain stride.
TEST_CASE("Equal components reduce the ragged layout to a plain stride",
          "[crystdof][sym][ragged]") {
    const int ns = 4, no = 3;
    const RaggedLayout L({ns, ns}, {0, 1, 0});
    for (int k = 0; k <= 2; ++k) REQUIRE(L.asym_off[k] == k * ns);
    for (int op = 0; op <= no; ++op) REQUIRE(L.full_off[op] == op * ns);
    for (int il = 0; il < no * ns; ++il) REQUIRE(L.op_of_atom[il] == il / ns);
}

// Fold full-cell forces onto the asymmetric unit and compare against FD through
// sym_expand; unequal component sizes keep a wrong offset from cancelling.
TEST_CASE("sym_fold gradient matches FD with unequal components",
          "[crystdof][sym][ragged]") {
    using klasp::devopt::sym_expand;
    using klasp::devopt::sym_fold;

    const std::vector<int> sites_of_indep = {2, 3};
    const std::vector<int> indep = {0, 1, 0, 1};   // identity, identity, inversion, inversion
    const RaggedLayout L(sites_of_indep, indep);
    const int no = static_cast<int>(indep.size());
    const int nasym = L.asym_off.back();           // 5
    const int nfull = L.full_off.back();           // 10
    const int nf = 3 * nasym;
    REQUIRE(nasym == 5);
    REQUIRE(nfull == 10);

    const double Rf[4][9] = {
        {1, 0, 0, 0, 1, 0, 0, 0, 1},
        {1, 0, 0, 0, 1, 0, 0, 0, 1},
        {-1, 0, 0, 0, -1, 0, 0, 0, -1},
        {-1, 0, 0, 0, -1, 0, 0, 0, -1},
    };
    double h0inv[9];
    REQUIRE(mat3_inv(H0, h0inv));
    std::vector<double> M0(no * 9), Rall(no * 9), tc(no * 3, 0.0);
    for (int op = 0; op < no; ++op) {
        double t[9];
        mat3_mul(H0, Rf[op], t);
        mat3_mul(t, h0inv, &M0[op * 9]);
        for (int i = 0; i < 9; ++i) Rall[op * 9 + i] = Rf[op][i];
    }
    tc[3] = 0.30; tc[4] = -0.20; tc[5] = 0.15;
    tc[6] = -0.40; tc[7] = 0.25; tc[8] = 0.10;
    tc[9] = 0.12; tc[10] = 0.33; tc[11] = -0.21;

    const double eps[6] = {0.01, -0.005, 0.02, 0.004, -0.002, 0.001};
    double def[9];
    strain_to_def(eps, def);

    std::vector<double> xb(3 * nasym + 6, 0.0);
    const double q0[15] = {1.2, -0.7, 0.4,  -0.9, 1.1, -1.3,  0.6, 0.2, -0.8,
                           -1.1, 0.5, 0.9,  0.3, -1.2, 0.7};
    for (int i = 0; i < 3 * nasym; ++i) xb[i] = q0[i];

    auto reconstruct = [&](const std::vector<double>& x) {
        std::vector<std::array<double, 3>> r(nfull);
        for (int il = 0; il < nfull; ++il) {
            double ri[3];
            sym_expand(def, x.data(), il, L.get(), M0.data(), tc.data(), indep.data(), ri);
            r[il] = {ri[0], ri[1], ri[2]};
        }
        return r;
    };

    Toy toy;
    toy.c.clear();
    for (int i = 0; i < nfull; ++i)
        toy.c.push_back({0.5 - 0.11 * i, 0.2 + 0.07 * i, -0.1 + 0.13 * i});

    const auto r = reconstruct(xb);
    std::vector<double> Fraw(nfull * 3);
    for (int il = 0; il < nfull; ++il) {
        const auto F = toy.force(r[il], il);
        Fraw[3*il] = F[0]; Fraw[3*il+1] = F[1]; Fraw[3*il+2] = F[2];
    }
    double Wdummy[6] = {0, 0, 0, 0, 0, 0};
    std::vector<double> grad(nf + 6, 0.0);
    double maxF = 0, ms = 0;
    sym_fold(def, Fraw.data(), 1.0, Wdummy, 1.0, no, L.get(), M0.data(), tc.data(),
             Rall.data(), indep.data(), H0, nf, static_cast<double>(nfull),
             grad.data(), maxF, ms);

    const double dlt = 1e-6;
    for (int j = 0; j < nf; ++j) {
        auto xp = xb, xm = xb;
        xp[j] += dlt; xm[j] -= dlt;
        const double fd = (toy.energy(reconstruct(xp)) - toy.energy(reconstruct(xm))) / (2 * dlt);
        INFO("asym DOF " << j << " (atom " << j/3 << ")");
        REQUIRE_THAT(grad[j], WithinAbs(fd, 1e-6));
    }
}

// sym_precond_pairs builds a model Hessian over the asymmetric-unit DOFs: it
// must be symmetric and touch only the asym block.
TEST_CASE("sym_precond_pairs builds a symmetric matrix over the asym DOFs",
          "[crystdof][sym][ragged]") {
    using klasp::devopt::sym_precond_pairs;

    const std::vector<int> sites_of_indep = {2, 3};
    const std::vector<int> indep = {0, 1, 0, 1};
    const RaggedLayout L(sites_of_indep, indep);
    const int no = static_cast<int>(indep.size());
    const int nasym = L.asym_off.back();
    const int ndof = 3 * nasym + 6;

    double h0inv[9];
    REQUIRE(mat3_inv(H0, h0inv));
    std::vector<double> M0(no * 9), tc(no * 3, 0.0);
    const double Rf[4][9] = {
        {1, 0, 0, 0, 1, 0, 0, 0, 1}, {1, 0, 0, 0, 1, 0, 0, 0, 1},
        {-1, 0, 0, 0, -1, 0, 0, 0, -1}, {-1, 0, 0, 0, -1, 0, 0, 0, -1}};
    for (int op = 0; op < no; ++op) {
        double t[9];
        mat3_mul(H0, Rf[op], t);
        mat3_mul(t, h0inv, &M0[op * 9]);
    }
    tc[3] = 0.9; tc[4] = -0.6; tc[5] = 0.4;
    tc[6] = -1.2; tc[7] = 0.7; tc[8] = 0.3;
    tc[9] = 0.5; tc[10] = 1.1; tc[11] = -0.7;

    std::vector<double> xb(ndof, 0.0);
    const double q0[15] = {1.2, -0.7, 0.4,  -0.9, 1.1, -1.3,  0.6, 0.2, -0.8,
                           -1.1, 0.5, 0.9,  0.3, -1.2, 0.7};
    for (int i = 0; i < 3 * nasym; ++i) xb[i] = q0[i];

    double hi[9];
    REQUIRE(mat3_inv(H0, hi));
    std::vector<double> P(static_cast<std::size_t>(ndof) * ndof, 0.0);
    sym_precond_pairs(xb.data(), ndof, no, L.get(), M0.data(), tc.data(), indep.data(),
                      H0, hi, true, P.data());

    double worst_asym = 0, touched_outside = 0;
    for (int i = 0; i < ndof; ++i)
        for (int j = 0; j < ndof; ++j) {
            const double v = P[static_cast<std::size_t>(i) * ndof + j];
            if (i < 3 * nasym && j < 3 * nasym)
                worst_asym = std::max(worst_asym,
                                      std::fabs(v - P[static_cast<std::size_t>(j) * ndof + i]));
            else
                touched_outside = std::max(touched_outside, std::fabs(v));
        }
    INFO("worst asymmetry " << worst_asym);
    REQUIRE(worst_asym < 1e-12);
    REQUIRE(touched_outside == 0.0);   // the 6 strain rows/cols are the caller's job

    double diag = 0;
    for (int i = 0; i < 3 * nasym; ++i) diag += std::fabs(P[static_cast<std::size_t>(i) * ndof + i]);
    REQUIRE(diag > 0.0);               // it did something
}

// ---------------------------------------------------------------------------
// Rigid-body DOFs (devopt::rigid_expand / rigid_fold). The rotation gradient
// is the exact derivative of the Rodrigues map, which equals minus the torque
// only at w = 0, so it is checked at a large rotation.
// ---------------------------------------------------------------------------
TEST_CASE("the rigid-body gradient matches finite differences", "[crystdof]") {
    using klasp::devopt::rigid_expand;
    using klasp::devopt::rigid_fold;

    // Five atoms in a lopsided arrangement, so no symmetry hides a wrong term.
    const int n = 5;
    const std::vector<double> p0 = {
        0.30, -0.70,  0.10,
        1.90,  0.40, -0.35,
       -1.10,  1.30,  0.85,
        0.55,  2.10, -1.40,
       -0.80, -1.60,  1.05,
    };
    double c[3] = {0, 0, 0};
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) c[k] += p0[3 * i + k] / n;

    // A fixed anisotropic "force field": E = sum_i a_i . r_i + 0.5 * k * |r_i - b_i|^2,
    // per-atom force -a_i - k (r_i - b_i).
    const std::vector<double> a = {
        0.11, -0.24,  0.37,   -0.42, 0.13, 0.09,   0.28, 0.31, -0.17,
       -0.36, 0.07, 0.44,      0.19, -0.29, -0.05,
    };
    const std::vector<double> b = {
        0.9, 0.2, -0.4,   -0.3, 1.1, 0.6,   0.5, -0.8, 0.2,
        1.4, 0.3, 0.7,    -1.2, 0.4, -0.9,
    };
    const double kspring = 0.35;

    auto energy_forces = [&](const double* x, double* F) {
        std::vector<double> r(3 * n, 0.0);
        rigid_expand(x, p0.data(), c, n, r.data());
        double E = 0;
        for (int i = 0; i < 3 * n; ++i) {
            const double dr = r[i] - b[i];
            E += a[i] * r[i] + 0.5 * kspring * dr * dr;
            if (F) F[i] = -a[i] - kspring * dr;
        }
        return E;
    };

    // ~0.9 rad about a tilted axis, plus a translation.
    std::vector<double> x = {0.23, -0.41, 0.16,  0.55, -0.30, 0.62};

    std::vector<double> F(3 * n, 0.0), g(6, 0.0);
    energy_forces(x.data(), F.data());
    rigid_fold(x.data(), p0.data(), c, F.data(), n, g.data());

    const double h = 1e-6;
    for (int k = 0; k < 6; ++k) {
        std::vector<double> xp = x, xm = x;
        xp[k] += h; xm[k] -= h;
        const double fd = (energy_forces(xp.data(), nullptr) - energy_forces(xm.data(), nullptr)) / (2 * h);
        REQUIRE_THAT(g[k], WithinAbs(fd, 1e-6));
    }

    // At w = 0 the rotation gradient is minus the torque.
    std::vector<double> x0(6, 0.0);
    energy_forces(x0.data(), F.data());
    rigid_fold(x0.data(), p0.data(), c, F.data(), n, g.data());
    double tau[3] = {0, 0, 0}, fnet[3] = {0, 0, 0};
    for (int i = 0; i < n; ++i) {
        const double d[3] = {p0[3*i] - c[0], p0[3*i+1] - c[1], p0[3*i+2] - c[2]};
        const double* Fi = &F[3 * i];
        for (int kk = 0; kk < 3; ++kk) fnet[kk] += Fi[kk];
        tau[0] += d[1]*Fi[2] - d[2]*Fi[1];
        tau[1] += d[2]*Fi[0] - d[0]*Fi[2];
        tau[2] += d[0]*Fi[1] - d[1]*Fi[0];
    }
    for (int kk = 0; kk < 3; ++kk) {
        REQUIRE_THAT(g[kk], WithinAbs(-fnet[kk], 1e-12));
        REQUIRE_THAT(g[3 + kk], WithinAbs(-tau[kk], 1e-12));
    }
}
