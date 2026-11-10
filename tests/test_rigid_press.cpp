// The rigid press (--collision rigid_press) soft-sphere potential: force sign,
// derivative consistency and taper.

#include "kokkos_listener.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "klasp/rigid_press.hpp"
#include "klasp/energy_kernels.hpp"

#include <cmath>

using namespace klasp;
using Catch::Matchers::WithinRel;

namespace {

// A two-site vdW table, so sigma = vdw(0) + vdw(1) in Bohr.
SView1D<RealType, ExecSpace> make_vdw(double a_ang, double b_ang) {
    SView1D<RealType, ExecSpace> v("vdw", 2);
    auto h = Kokkos::create_mirror_view(v);
    h(0) = static_cast<RealType>(a_ang);
    h(1) = static_cast<RealType>(b_ang);
    Kokkos::deep_copy(v, h);
    return v;
}

// Evaluate the potential on a pair separated by (rx, 0, 0). Runs on the device:
// the vdW table is in execution-space memory.
bool eval_wca(const WCA12PairFn<ExecSpace>& fn, double rx,
              RealType& e, RealType& fs) {
    const WCA12PairFn<ExecSpace> f = fn;   // by value: a device lambda cannot capture a reference
    const RealType r = static_cast<RealType>(rx);
    Kokkos::View<RealType[3], MemSpace> out("eval_wca");
    Kokkos::parallel_for("eval_wca", Kokkos::RangePolicy<ExecSpace>(0, 1),
                         KOKKOS_LAMBDA(int) {
        RealType ee = 0, ff = 0;
        const bool ok = f.eval(/*body_i=*/0, /*body_j=*/1, r, r * r, ee, ff);
        out(0) = ok ? RealType(1) : RealType(0);
        out(1) = ee;
        out(2) = ff;
    });
    Kokkos::fence();
    const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace{}, out);
    e  = host(1);
    fs = host(2);
    return host(0) != RealType(0);
}

WCA12PairFn<ExecSpace> make_fn() {
    return WCA12PairFn<ExecSpace>{make_vdw(1.7, 1.7), RP_EPS, RP_ALPHA,
                                  RP_MAX_CUTOFF * RP_MAX_CUTOFF};
}

}  // namespace

// pair_iteration's convention is f_i = fs * g.rx with g.rx = pos(j) - pos(i), so
// a repulsive force on i is antiparallel to rx.
TEST_CASE("the soft-sphere force pushes overlapping atoms apart", "[rigid_press]") {
    const auto fn = make_fn();
    const double sigma = (1.7 + 1.7) * ANGSTROM_TO_BOHR;

    for (double frac : {0.5, 0.7, 0.9, 0.99, 1.1, 1.3}) {
        const double r = frac * sigma;
        RealType e = 0, fs = 0;
        REQUIRE(eval_wca(fn, r, e, fs));
        INFO("r / sigma = " << frac);
        // Repulsive: positive energy, and the force on i is along -rx.
        CHECK(e > 0.0);
        const double f_i_x = double(fs) * r;   // rx = +r here
        CHECK(f_i_x < 0.0);
    }
}

// fs must be exactly V'(r)/r, including through the quintic taper. Checked
// either side of r = sigma, where the taper switches on.
TEST_CASE("the soft-sphere force scalar is the energy's derivative", "[rigid_press]") {
    const auto fn = make_fn();
    const double sigma = (1.7 + 1.7) * ANGSTROM_TO_BOHR;
    const double h = 1e-6;

    for (double frac : {0.6, 0.9, 1.05, 1.2, 1.4}) {
        const double r = frac * sigma;
        RealType e = 0, fs = 0;
        REQUIRE(eval_wca(fn, r, e, fs));

        RealType ep = 0, em = 0, dummy = 0;
        REQUIRE(eval_wca(fn, r + h, ep, dummy));
        REQUIRE(eval_wca(fn, r - h, em, dummy));
        const double fd = (double(ep) - double(em)) / (2 * h);

        INFO("r / sigma = " << frac);
        CHECK_THAT(double(fs) * r, WithinRel(fd, 1e-5));
    }
}

// Outside alpha*sigma the pair contributes nothing, and the energy reaches zero
// smoothly.
TEST_CASE("the soft-sphere potential vanishes smoothly at its cutoff",
          "[rigid_press]") {
    const auto fn = make_fn();
    const double sigma = (1.7 + 1.7) * ANGSTROM_TO_BOHR;
    const double r_cut = RP_ALPHA * sigma;

    RealType e = 0, fs = 0;
    CHECK_FALSE(eval_wca(fn, r_cut * 1.001, e, fs));

    // Just inside the cutoff the quintic gives ~1e-10, not zero; untapered it
    // would be ~8e-6, so a bound two orders below that detects a step.
    REQUIRE(eval_wca(fn, r_cut * 0.999, e, fs));
    CHECK(std::abs(double(e)) < 1e-7);
    CHECK(std::abs(double(fs)) < 1e-7);

    // Monotonically repulsive on the way in.
    RealType e_near = 0, e_far = 0, d = 0;
    REQUIRE(eval_wca(fn, sigma * 0.8, e_near, d));
    REQUIRE(eval_wca(fn, sigma * 1.2, e_far, d));
    CHECK(double(e_near) > double(e_far));
}
