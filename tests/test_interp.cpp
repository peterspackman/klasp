// RadialTable as a potential: it reproduces the function, its first derivative
// matches what it reproduces, and its second derivative is continuous.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "klasp/interp.hpp"
#include "kokkos_listener.hpp"

#include <cmath>
#include <vector>

using namespace klasp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// A Buckingham-shaped wall spanning five orders of magnitude onto a small floor,
// the shape of every CEFF repulsion table.
double wall(double r) { return 500.0 * std::exp(-3.0 * r) + 1.0e-3; }

// Something with a sign change and real curvature, for the Direct path.
double mixed(double r) { return std::exp(-1.2 * r) - 0.5 / (r * r * r); }

template <typename F>
double fd1(F&& f, double r, double h) { return (f(r + h) - f(r - h)) / (2 * h); }

template <typename F>
double fd2(F&& f, double r, double h) {
    return (f(r + h) - 2 * f(r) + f(r - h)) / (h * h);
}

// Relative with an absolute floor.
void agrees(double got, double want, double rel, double floor_) {
    const double tol = rel * std::max(std::abs(want), std::abs(got)) + floor_;
    INFO("got " << got << " want " << want << " tol " << tol);
    REQUIRE(std::abs(got - want) <= tol);
}

// Finite-difference noise is about eps*|f|/h (eps*|f|/h^2 for the second
// difference); the factor covers the spline evaluation's own ulps.
constexpr double kFdSlack = 16.0 * 2.220446049250313e-16;
double fd1_floor(double f_mag, double h) { return kFdSlack * std::abs(f_mag) / (2 * h); }
double fd2_floor(double f_mag, double h) { return kFdSlack * std::abs(f_mag) / (h * h); }

}  // namespace

TEST_CASE("cubic spline reproduces a cubic exactly", "[interp]") {
    // Not-a-knot end conditions make the spline exact for a single cubic.
    auto f = [](double r) { return 2.0 - 0.75 * r + 0.4 * r * r - 0.11 * r * r * r; };
    auto df = [](double r) { return -0.75 + 0.8 * r - 0.33 * r * r; };
    auto d2f = [](double r) { return 0.8 - 0.66 * r; };

    RadialTable<> t(0.5, 6.0, 24, f, {RadialBelow::Linear, RadialAbove::Linear});
    auto s = t.view();

    for (double r = 0.55; r < 5.95; r += 0.037) {
        RealType v, dv, d2v;
        s.eval(r, v, dv, d2v);
        REQUIRE_THAT(double(v),   WithinAbs(f(r),   1e-12));
        REQUIRE_THAT(double(dv),  WithinAbs(df(r),  1e-11));
        REQUIRE_THAT(double(d2v), WithinAbs(d2f(r), 1e-10));
    }
}

TEST_CASE("derivatives agree with finite differences", "[interp]") {
    SECTION("linear grid, direct scale") {
        RadialTable<RadialGrid::Linear, RadialScale::Direct> t(
            0.6, 9.0, 600, mixed, {RadialBelow::Linear, RadialAbove::Linear});
        auto s = t.view();
        auto probe = [&](double r) { return double(s.value(r)); };
        for (double r = 1.0; r < 8.0; r += 0.31) {
            RealType v, dv, d2v;
            s.eval(r, v, dv, d2v);
            agrees(dv,  fd1(probe, r, 1e-5), 1e-7, fd1_floor(probe(r), 1e-5));
            agrees(d2v, fd2(probe, r, 1e-4), 1e-5, fd2_floor(probe(r), 1e-4));
        }
    }

    SECTION("log grid, log scale") {
        // Both chain rules active at once: t = ln r and f = exp(u).
        RadialTable<RadialGrid::Log, RadialScale::Log> t(
            0.4, 12.0, 90, wall, {RadialBelow::Exponential, RadialAbove::Linear});
        auto s = t.view();
        auto probe = [&](double r) { return double(s.value(r)); };
        for (double r = 0.6; r < 11.0; r += 0.43) {
            RealType v, dv, d2v;
            s.eval(r, v, dv, d2v);
            agrees(dv,  fd1(probe, r, 1e-6 * r), 1e-6, fd1_floor(probe(r), 1e-6 * r));
            agrees(d2v, fd2(probe, r, 1e-4 * r), 1e-4, fd2_floor(probe(r), 1e-4 * r));
        }
    }

    SECTION("log grid, direct scale") {
        RadialTable<RadialGrid::Log, RadialScale::Direct> t(
            0.6, 9.0, 300, mixed, {RadialBelow::Linear, RadialAbove::Linear});
        auto s = t.view();
        auto probe = [&](double r) { return double(s.value(r)); };
        for (double r = 1.0; r < 8.0; r += 0.29) {
            RealType v, dv, d2v;
            s.eval(r, v, dv, d2v);
            agrees(dv,  fd1(probe, r, 1e-6 * r), 1e-6, fd1_floor(probe(r), 1e-6 * r));
            agrees(d2v, fd2(probe, r, 1e-4 * r), 1e-4, fd2_floor(probe(r), 1e-4 * r));
        }
    }
}

TEST_CASE("the second derivative is continuous across every knot", "[interp]") {
    // A continuous d2 gives a gap across a knot that falls linearly with eps; a
    // step stays put. So the test is the ratio, not the size.
    RadialTable<> t(0.5, 10.0, 64, mixed, {RadialBelow::Linear, RadialAbove::Linear});
    auto s = t.view();
    const std::vector<double> r = radial_knots(0.5, 10.0, 64, RadialGrid::Linear);

    auto worst_jump = [&](double eps) {
        double jv = 0, jd = 0, jh = 0;
        for (size_t i = 1; i + 1 < r.size(); ++i) {
            RealType vl, dl, d2l, vr, dr_, d2r;
            s.eval(r[i] - eps, vl, dl, d2l);
            s.eval(r[i] + eps, vr, dr_, d2r);
            jv = std::max(jv, std::abs(double(vr) - double(vl)));
            jd = std::max(jd, std::abs(double(dr_) - double(dl)));
            jh = std::max(jh, std::abs(double(d2r) - double(d2l)));
        }
        INFO("eps " << eps << ": v " << jv << " d " << jd << " d2 " << jh);
        return jh;
    };

    const double j1 = worst_jump(1e-4);
    const double j2 = worst_jump(1e-5);
    const double j3 = worst_jump(1e-6);
    INFO("d2 jump ratios " << j1 / j2 << ", " << j2 / j3 << " (10 = continuous, 1 = a step)");
    REQUIRE(j2 < j1 / 8.0);
    REQUIRE(j3 < j2 / 8.0);
}

TEST_CASE("accuracy improves like h^4", "[interp]") {
    // A cubic spline is O(h^4) in the value, so halving the spacing should cut the
    // error by about 16.
    auto err = [](int n) {
        RadialTable<> t(0.8, 8.0, n, mixed, {RadialBelow::Linear, RadialAbove::Linear});
        auto s = t.view();
        double worst = 0.0;
        for (double r = 0.85; r < 7.95; r += 0.0137)
            worst = std::max(worst, std::abs(double(s.value(r)) - mixed(r)));
        return worst;
    };
    const double e1 = err(41);
    const double e2 = err(81);
    INFO("error " << e1 << " -> " << e2 << ", ratio " << e1 / e2);
    REQUIRE(e2 < e1 / 8.0);
}

TEST_CASE("a log grid buys accuracy per knot on a steep wall", "[interp]") {
    // Log knots against uniform knots on the same wall and knot count.
    auto worst_rel = [](auto&& table) {
        auto s = table.view();
        double worst = 0.0;
        for (double r = 0.45; r < 11.9; r += 0.0071)
            worst = std::max(worst, std::abs(double(s.value(r)) - wall(r)) / wall(r));
        return worst;
    };

    RadialTable<RadialGrid::Log, RadialScale::Log> geometric(
        0.4, 12.0, 60, wall, {RadialBelow::Exponential, RadialAbove::Linear});
    RadialTable<RadialGrid::Linear, RadialScale::Direct> uniform(
        0.4, 12.0, 60, wall, {RadialBelow::Linear, RadialAbove::Linear});

    const double log_err = worst_rel(geometric);
    const double lin_err = worst_rel(uniform);
    INFO("60 knots: log grid rel err " << log_err << ", linear grid " << lin_err);
    REQUIRE(log_err < lin_err / 5.0);
    REQUIRE(log_err < 1e-3);
}

TEST_CASE("the continuations join C1 and stay physical", "[interp]") {
    SECTION("exponential below r_min") {
        // Contacts closer than r_min: the continuation must stay positive,
        // monotone and repulsive.
        RadialTable<RadialGrid::Log, RadialScale::Log> t(
            0.4, 12.0, 90, wall, {RadialBelow::Exponential, RadialAbove::Zero});
        auto s = t.view();

        RealType v_in, d_in, d2_in, v_out, d_out, d2_out;
        s.eval(0.4 + 1e-8, v_in, d_in, d2_in);
        s.eval(0.4 - 1e-8, v_out, d_out, d2_out);
        REQUIRE_THAT(double(v_out), WithinRel(double(v_in), 1e-6));
        REQUIRE_THAT(double(d_out), WithinRel(double(d_in), 1e-5));

        double prev = -1.0;
        for (double r = 0.4; r > 0.02; r -= 0.01) {
            const double v = s.value(r);
            REQUIRE(v > 0.0);
            REQUIRE(v >= prev);      // rising as r falls
            prev = v;
        }
        REQUIRE(std::isfinite(double(s.value(1e-6))));

        // The wall is an exponential, so the edge-slope continuation should
        // reproduce it.
        for (double r = 0.35; r > 0.05; r -= 0.05)
            REQUIRE_THAT(double(s.value(r)), WithinRel(wall(r), 1e-4));
    }

    SECTION("linear below r_min") {
        RadialTable<> t(1.0, 8.0, 50, mixed, {RadialBelow::Linear, RadialAbove::Linear});
        auto s = t.view();
        RealType v_in, d_in, d2_in, v_out, d_out, d2_out;
        s.eval(1.0 + 1e-8, v_in, d_in, d2_in);
        s.eval(1.0 - 1e-8, v_out, d_out, d2_out);
        REQUIRE_THAT(double(v_out), WithinRel(double(v_in), 1e-6));
        REQUIRE_THAT(double(d_out), WithinRel(double(d_in), 1e-5));
    }

    SECTION("above r_max") {
        RadialTable<> zero_tail(1.0, 8.0, 50, mixed,
                                {RadialBelow::Linear, RadialAbove::Zero});
        REQUIRE(zero_tail.view().value(8.5) == RealType(0));
        REQUIRE_THAT(double(zero_tail.value_at_max()), WithinAbs(mixed(8.0), 1e-6));

        RadialTable<> linear_tail(1.0, 8.0, 50, mixed,
                                  {RadialBelow::Linear, RadialAbove::Linear});
        auto s = linear_tail.view();
        RealType v_in, d_in, d2_in, v_out, d_out, d2_out;
        s.eval(8.0 - 1e-8, v_in, d_in, d2_in);
        s.eval(8.0 + 1e-8, v_out, d_out, d2_out);
        REQUIRE_THAT(double(v_out), WithinRel(double(v_in), 1e-6));
        REQUIRE_THAT(double(d_out), WithinRel(double(d_in), 1e-5));
    }
}

TEST_CASE("value() is the same function as eval()", "[interp]") {
    // value() takes a shorter path through the polynomial than the derivative
    // evaluation; both must agree.
    RadialTable<RadialGrid::Log, RadialScale::Log> t(
        0.4, 12.0, 90, wall, {RadialBelow::Exponential, RadialAbove::Zero});
    auto s = t.view();
    for (double r = 0.2; r < 13.0; r += 0.0031) {
        RealType v, dv, d2v;
        s.eval(r, v, dv, d2v);
        REQUIRE(v == s.value(r));
    }
}

TEST_CASE("the reciprocal overload matches", "[interp]") {
    RadialTable<RadialGrid::Log, RadialScale::Log> t(
        0.4, 12.0, 90, wall, {RadialBelow::Exponential, RadialAbove::Zero});
    auto s = t.view();
    for (double r = 0.5; r < 11.5; r += 0.017) {
        RealType a, b, c, d, e, f;
        s.eval(r, a, b, c);
        s.eval(r, RealType(1) / RealType(r), d, e, f);
        REQUIRE(a == d);
        REQUIRE(b == e);
        REQUIRE(c == f);
    }
}

TEST_CASE("samples supplied by the caller match samples taken here", "[interp]") {
    const std::vector<double> r = radial_knots(0.7, 9.0, 40, RadialGrid::Log);
    std::vector<double> y(r.size());
    for (size_t i = 0; i < r.size(); ++i) y[i] = wall(r[i]);

    RadialTable<RadialGrid::Log, RadialScale::Log> from_samples(0.7, 9.0, y);
    RadialTable<RadialGrid::Log, RadialScale::Log> from_functor(0.7, 9.0, 40, wall);
    REQUIRE(from_samples.coefficients() == from_functor.coefficients());
}

TEST_CASE("build rejects what it cannot represent", "[interp]") {
    REQUIRE_THROWS(RadialTable<>(1.0, 1.0, 10, mixed));                  // empty range
    REQUIRE_THROWS(RadialTable<>(1.0, 5.0, std::vector<double>{1.0}));   // one sample
    REQUIRE_THROWS(RadialTable<RadialGrid::Log>(0.0, 5.0, 10, mixed));   // log grid at 0
    REQUIRE_THROWS((RadialTable<RadialGrid::Linear, RadialScale::Log>(0.6, 9.0, 40, mixed)));
}

TEST_CASE("small tables are still splines", "[interp]") {
    // n = 3 and n = 2 are separate branches in the moment solve.
    auto f = [](double r) { return 1.0 + 0.5 * r - 0.25 * r * r; };
    RadialTable<> three(1.0, 3.0, 3, f, {RadialBelow::Linear, RadialAbove::Linear});
    for (double r = 1.05; r < 2.95; r += 0.1)
        REQUIRE_THAT(double(three.view().value(r)), WithinAbs(f(r), 1e-12));

    RadialTable<> two(1.0, 3.0, 2, f, {RadialBelow::Linear, RadialAbove::Linear});
    RealType v, dv, d2v;
    two.view().eval(2.0, v, dv, d2v);
    REQUIRE_THAT(double(v), WithinAbs(0.5 * (f(1.0) + f(3.0)), 1e-12));  // linear
    REQUIRE_THAT(double(d2v), WithinAbs(0.0, 1e-12));
}

TEST_CASE("the same table evaluates identically on the device", "[interp]") {
    // Device evaluation agrees with the host to a relative tolerance with an
    // absolute floor, not bitwise: multiply-add contraction differs.
    RadialTable<RadialGrid::Log, RadialScale::Log> t(
        0.4, 12.0, 90, wall, {RadialBelow::Exponential, RadialAbove::Zero});
    auto dev = upload(t, "test_wall");
    auto spline = dev.spline;

    const int n = 512;
    Kokkos::View<RealType* [3], MemSpace> out("out", n);
    Kokkos::parallel_for("eval", n, KOKKOS_LAMBDA(int i) {
        const RealType r = RealType(0.3) + RealType(i) * RealType(13.0 / n);
        RealType v, dv, d2v;
        spline.eval(r, v, dv, d2v);
        out(i, 0) = v; out(i, 1) = dv; out(i, 2) = d2v;
    });
    Kokkos::fence();
    auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace{}, out);

    auto ref = t.view();
    for (int i = 0; i < n; ++i) {
        const RealType r = RealType(0.3) + RealType(i) * RealType(13.0 / n);
        RealType v, dv, d2v;
        ref.eval(r, v, dv, d2v);
        REQUIRE_THAT(double(host(i, 0)), WithinRel(double(v),   1e-11) || WithinAbs(double(v),   1e-15));
        REQUIRE_THAT(double(host(i, 1)), WithinRel(double(dv),  1e-11) || WithinAbs(double(dv),  1e-15));
        REQUIRE_THAT(double(host(i, 2)), WithinRel(double(d2v), 1e-11) || WithinAbs(double(d2v), 1e-15));
    }
}
