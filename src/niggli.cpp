#include "klasp/niggli.hpp"
#include "klasp/cell.hpp"

#include <cmath>
#include <algorithm>

namespace klasp {

namespace {

// The basis as three vectors; every reduction step is a swap, a negation or a
// subtraction, and the metric is recomputed from the vectors each pass.
struct Basis {
    RealType v[3][3];   // v[i] is lattice vector i
    int p[3][3];        // transform, p[i][j] = P(i,j), reduced = original * P

    void swap_vec(int i, int j) {
        for (int d = 0; d < 3; ++d) {
            std::swap(v[i][d], v[j][d]);
            std::swap(p[d][i], p[d][j]);
        }
    }
    void negate(int i) {
        for (int d = 0; d < 3; ++d) { v[i][d] = -v[i][d]; p[d][i] = -p[d][i]; }
    }
    void axpy(int i, int j, int s) {
        for (int d = 0; d < 3; ++d) {
            v[i][d] += RealType(s) * v[j][d];
            p[d][i] += s * p[d][j];
        }
    }
};

RealType dot(const RealType a[3], const RealType b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

}  // namespace

NiggliResult niggli_reduce(const RealType cell[3][3], RealType eps_in) {
    NiggliResult out{};
    Basis b{};
    for (int i = 0; i < 3; ++i)
        for (int d = 0; d < 3; ++d) b.v[i][d] = cell[d][i];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) b.p[i][j] = (i == j) ? 1 : 0;

    // Metric entries are squared lengths, so the tolerance is too. Fixed from the
    // input: a threshold that moved with the basis could let the steps cycle.
    const RealType A0 = dot(b.v[0], b.v[0]);
    const RealType B0 = dot(b.v[1], b.v[1]);
    const RealType C0 = dot(b.v[2], b.v[2]);
    RealType eps = eps_in;
    if (!(eps > 0)) eps = RealType(1e-5) * std::cbrt(A0 * B0 * C0);
    if (!(eps > 0)) eps = RealType(1e-10);

    auto sgn = [](RealType x) { return x < 0 ? -1 : 1; };

    constexpr int MAX_STEPS = 1000;
    int step = 0;
    for (; step < MAX_STEPS; ++step) {
        const RealType A  = dot(b.v[0], b.v[0]);
        const RealType B  = dot(b.v[1], b.v[1]);
        const RealType C  = dot(b.v[2], b.v[2]);
        const RealType xi = 2 * dot(b.v[1], b.v[2]);
        const RealType et = 2 * dot(b.v[0], b.v[2]);
        const RealType ze = 2 * dot(b.v[0], b.v[1]);

        // 1, 2: order the vectors by length, ties broken by the off-diagonal.
        if (A > B + eps || (std::abs(A - B) <= eps && std::abs(xi) > std::abs(et) + eps)) {
            b.swap_vec(0, 1);
            continue;
        }
        if (B > C + eps || (std::abs(B - C) <= eps && std::abs(et) > std::abs(ze) + eps)) {
            b.swap_vec(1, 2);
            continue;
        }

        // 3, 4: normalise the signs. Under diag(i,j,k) the products transform as
        // xi -> jk xi, eta -> ik eta, zeta -> ij zeta, so only four combinations are
        // distinct; try them all.
        {
            const bool type_i = (xi * et * ze > 0);
            int best = -1;
            for (int t = 0; t < 4; ++t) {
                const int f[3] = {t == 1 ? -1 : 1, t == 2 ? -1 : 1, t == 3 ? -1 : 1};
                const RealType x = f[1]*f[2]*xi, e = f[0]*f[2]*et, z = f[0]*f[1]*ze;
                const bool ok = type_i ? (x > eps && e > eps && z > eps)
                                       : (x <= eps && e <= eps && z <= eps);
                if (ok) { best = t; break; }
            }
            if (best > 0) { b.negate(best - 1); continue; }
        }

        // 5, 6, 7: shorten one vector against another.
        if (std::abs(xi) > B + eps
            || (std::abs(xi - B) <= eps && 2 * et < ze - eps)
            || (std::abs(xi + B) <= eps && ze < -eps)) {
            b.axpy(2, 1, -sgn(xi));
            continue;
        }
        if (std::abs(et) > A + eps
            || (std::abs(et - A) <= eps && 2 * xi < ze - eps)
            || (std::abs(et + A) <= eps && ze < -eps)) {
            b.axpy(2, 0, -sgn(et));
            continue;
        }
        if (std::abs(ze) > A + eps
            || (std::abs(ze - A) <= eps && 2 * xi < et - eps)
            || (std::abs(ze + A) <= eps && et < -eps)) {
            b.axpy(1, 0, -sgn(ze));
            continue;
        }

        // 8: the body diagonal is shorter than c.
        if (xi + et + ze + A + B < -eps
            || (std::abs(xi + et + ze + A + B) <= eps && 2 * (A + et) + ze > eps)) {
            b.axpy(2, 0, 1);
            b.axpy(2, 1, 1);
            continue;
        }

        out.converged = true;
        break;
    }
    out.iterations = step;

    if (!out.converged) {
        // Return the input untouched rather than a half-reduced basis.
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) out.cell[r][c] = cell[r][c];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) out.transform[i][j] = (i == j) ? 1 : 0;
        return out;
    }

    // Negating one vector spans the same lattice, so handedness is restored here.
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) out.cell[r][c] = b.v[c][r];
    if (mat3_det(out.cell) < 0) {
        b.negate(2);
        for (int r = 0; r < 3; ++r) out.cell[r][2] = b.v[2][r];
    }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out.transform[i][j] = b.p[i][j];
    return out;
}

bool niggli_reduce_params(const RealType params_in[6], RealType params_out[6],
                          int transform[3][3]) {
    RealType m[3][3];
    cell_params_to_matrix(params_in, m);
    const NiggliResult r = niggli_reduce(m);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) transform[i][j] = r.transform[i][j];
    if (!r.converged) {
        for (int k = 0; k < 6; ++k) params_out[k] = params_in[k];
        return false;
    }

    auto col = [&](int j, int d) { return r.cell[d][j]; };
    auto len = [&](int j) {
        return std::sqrt(col(j,0)*col(j,0) + col(j,1)*col(j,1) + col(j,2)*col(j,2));
    };
    const RealType la = len(0), lb = len(1), lc = len(2);
    auto ang = [&](int i, int j, RealType li, RealType lj) {
        RealType c = (col(i,0)*col(j,0) + col(i,1)*col(j,1) + col(i,2)*col(j,2)) / (li * lj);
        c = std::max(RealType(-1), std::min(RealType(1), c));
        return std::acos(c) * RealType(180.0 / M_PI);
    };
    params_out[0] = la;
    params_out[1] = lb;
    params_out[2] = lc;
    params_out[3] = ang(1, 2, lb, lc);
    params_out[4] = ang(0, 2, la, lc);
    params_out[5] = ang(0, 1, la, lb);
    return true;
}

void niggli_map_fractional(const int transform[3][3], const double frac_in[3],
                           double frac_out[3]) {
    // P is unimodular, so its inverse is the adjugate over det = +/-1, integral.
    const int (*p)[3] = transform;
    const int det = p[0][0]*(p[1][1]*p[2][2] - p[1][2]*p[2][1])
                  - p[0][1]*(p[1][0]*p[2][2] - p[1][2]*p[2][0])
                  + p[0][2]*(p[1][0]*p[2][1] - p[1][1]*p[2][0]);
    if (det == 0) { for (int i = 0; i < 3; ++i) frac_out[i] = frac_in[i]; return; }
    int inv[3][3];
    inv[0][0] =  (p[1][1]*p[2][2] - p[1][2]*p[2][1]) / det;
    inv[0][1] = -(p[0][1]*p[2][2] - p[0][2]*p[2][1]) / det;
    inv[0][2] =  (p[0][1]*p[1][2] - p[0][2]*p[1][1]) / det;
    inv[1][0] = -(p[1][0]*p[2][2] - p[1][2]*p[2][0]) / det;
    inv[1][1] =  (p[0][0]*p[2][2] - p[0][2]*p[2][0]) / det;
    inv[1][2] = -(p[0][0]*p[1][2] - p[0][2]*p[1][0]) / det;
    inv[2][0] =  (p[1][0]*p[2][1] - p[1][1]*p[2][0]) / det;
    inv[2][1] = -(p[0][0]*p[2][1] - p[0][1]*p[2][0]) / det;
    inv[2][2] =  (p[0][0]*p[1][1] - p[0][1]*p[1][0]) / det;
    for (int i = 0; i < 3; ++i)
        frac_out[i] = inv[i][0]*frac_in[0] + inv[i][1]*frac_in[1] + inv[i][2]*frac_in[2];
}

}  // namespace klasp
