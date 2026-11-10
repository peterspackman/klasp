#pragma once
// Trust-region dogleg in a ball of radius delta over a limited-memory BFGS
// model (as chmpy.opt.TrustRegion).
//
// B = B0 + sum over the last `mem` pairs of
//   y_i y_i^T / (y_i.s_i) - a_i a_i^T / (s_i.a_i),   a_i = B_{i-1} s_i,
// with B0 = gamma * shape, the shape being dense (Cholesky factor and matrix),
// diagonal, or the identity. B^-1 v is the two-loop recursion through the same
// B0, so the Newton step and predicted reduction describe one model. Updates
// are Powell-damped (y blended towards B s until y.s = 0.2 s.B.s), keeping B
// positive definite.
//
// Lane contexts: Team spreads one structure's vectors over a warp, Serial runs
// one structure per thread.

#include <Kokkos_Core.hpp>

namespace klasp::tr {

// How gamma follows the pairs. Scaled: y.y / y.s after each update (for a shape
// normalised to mean diagonal one). Fixed: gamma stays at its initial value.
enum class Gamma { Scaled, Fixed };

template <class T>
struct Model {
    T *S = nullptr, *Y = nullptr, *A = nullptr, *PS = nullptr;   // pairs [mem*n], PS = shape * s
    T *ys = nullptr, *sa = nullptr;                              // [mem]
    T* alpha = nullptr;     // [mem] scratch: the two-loop recursion's coefficients
    T *q = nullptr, *bg = nullptr, *tmp = nullptr;               // [n] scratch
    const T* L = nullptr;   // dense shape: Cholesky factor ...
    const T* P = nullptr;   // ... and the matrix, L L^T
    T* D = nullptr;         // diagonal shape (used when P is null)
    int n = 0;
    T gamma = 1;
    int hc = 0, hh = 0;
    int mem = 12;           // pairs kept
    Gamma rule = Gamma::Scaled;
    // Full memory: drop the oldest pair (rebuilding every a_i, O(m^2 n)), or fold
    // the model's diagonal into a diagonal shape and start the pairs again (O(m n)).
    bool fold = false;
};

struct Serial {
    template <class T, class F>
    KOKKOS_INLINE_FUNCTION T sum(int n, F&& f) const {
        T s = 0;
        for (int k = 0; k < n; ++k) s += f(k);
        return s;
    }
    template <class F>
    KOKKOS_INLINE_FUNCTION void each(int n, F&& f) const {
        for (int k = 0; k < n; ++k) f(k);
    }
    template <class F>
    KOKKOS_INLINE_FUNCTION void single(F&& f) const { f(); }
};

// One structure per team, its vectors across the vector lanes.
template <class Member>
struct Team {
    const Member& t;
    template <class T, class F>
    KOKKOS_INLINE_FUNCTION T sum(int n, F&& f) const {
        T s = 0;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, n), [&](int k, T& u) { u += f(k); }, s);
        return s;
    }
    template <class F>
    KOKKOS_INLINE_FUNCTION void each(int n, F&& f) const {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, n), f);
        t.team_barrier();   // what one lane wrote, the next primitive reads on another
    }
    template <class F>
    KOKKOS_INLINE_FUNCTION void single(F&& f) const {
        Kokkos::single(Kokkos::PerThread(t), f);
        t.team_barrier();
    }
};

template <class C, class T>
KOKKOS_INLINE_FUNCTION T lb_dot(const C& c, const Model<T>& m, const T* a, const T* b) {
    return c.template sum<T>(m.n, [&](int k) { return a[k] * b[k]; });
}

template <class T>
KOKKOS_INLINE_FUNCTION T* lb_pair(const Model<T>& m, T* base, int pos) { return base + pos * m.n; }

template <class C, class T>
KOKKOS_INLINE_FUNCTION void lb_shape_matvec(const C& c, const Model<T>& m, const T* v, T* out) {
    const int n = m.n;
    if (m.P) {
        c.each(n, [&](int i) {
            T u = 0;
            for (int k = 0; k < n; ++k) u += m.P[k * n + i] * v[k];   // symmetric: column reads coalesce
            out[i] = u;
        });
    } else if (m.D) {
        c.each(n, [&](int i) { out[i] = m.D[i] * v[i]; });
    } else {
        c.each(n, [&](int i) { out[i] = v[i]; });
    }
}

template <class C, class T>
KOKKOS_INLINE_FUNCTION void lb_base_solve(const C& c, const Model<T>& m, T* v) {
    const int n = m.n;
    if (m.P) {
        for (int i = 0; i < n; ++i) {   // L y = v (y in tmp)
            const T u = c.template sum<T>(i, [&](int k) { return m.L[i * n + k] * m.tmp[k]; });
            const T yi = (v[i] - u) / m.L[i * n + i];
            c.single([&] { m.tmp[i] = yi; });
        }
        for (int i = n - 1; i >= 0; --i) {   // L^T v = y
            // Zero-based range offset by hand: some Kokkos releases ignore a vector
            // range's begin on CUDA.
            const T u = c.template sum<T>(n - i - 1, [&](int kk) { const int k = i + 1 + kk; return m.L[k * n + i] * v[k]; });
            const T xi = (m.tmp[i] - u) / m.L[i * n + i];
            c.single([&] { v[i] = xi; });
        }
    } else if (m.D) {
        c.each(n, [&](int k) { v[k] /= m.D[k]; });
    }
    const T ig = T(1) / m.gamma;
    c.each(n, [&](int k) { v[k] *= ig; });
}

// out = B v (out must not alias v).
template <class C, class T>
KOKKOS_INLINE_FUNCTION void lb_matvec(const C& c, const Model<T>& m, const T* v, T* out) {
    const int n = m.n;
    lb_shape_matvec(c, m, v, out);
    const T g = m.gamma;
    c.each(n, [&](int k) { out[k] *= g; });
    for (int i = 0; i < m.hc; ++i) {
        const int pos = (m.hh + i) % m.mem;
        const T* y = lb_pair(m, m.Y, pos); const T* a = lb_pair(m, m.A, pos);
        const T cy = lb_dot(c, m, y, v) / m.ys[pos];
        const T ca = lb_dot(c, m, a, v) / m.sa[pos];
        c.each(n, [&](int k) { out[k] += cy * y[k] - ca * a[k]; });
    }
}

// a_i = B_{i-1} s_i for the stored pairs from `from` on, oldest first, under the
// current gamma.
template <class C, class T>
KOKKOS_INLINE_FUNCTION void lb_refresh(const C& c, Model<T>& m, int from = 0) {
    const int n = m.n;
    for (int i = from; i < m.hc; ++i) {
        const int pos = (m.hh + i) % m.mem;
        T* a = lb_pair(m, m.A, pos);
        const T* si = lb_pair(m, m.S, pos);
        const T* psi = lb_pair(m, m.PS, pos);
        const T g = m.gamma;
        c.each(n, [&](int k) { a[k] = g * psi[k]; });
        for (int j = 0; j < i; ++j) {
            const int pj = (m.hh + j) % m.mem;
            const T* yj = lb_pair(m, m.Y, pj); const T* aj = lb_pair(m, m.A, pj);
            const T cy = lb_dot(c, m, yj, si) / m.ys[pj];
            const T ca = lb_dot(c, m, aj, si) / m.sa[pj];
            c.each(n, [&](int k) { a[k] += cy * yj[k] - ca * aj[k]; });
        }
        const T sa = lb_dot(c, m, si, a);
        c.single([&] { m.sa[pos] = sa; });
    }
}

// v <- B^-1 v by the two-loop recursion through B0.
template <class C, class T>
KOKKOS_INLINE_FUNCTION void lb_solve(const C& c, const Model<T>& m, T* v) {
    const int n = m.n, cnt = m.hc;
    for (int i = cnt - 1; i >= 0; --i) {
        const int pos = (m.hh + i) % m.mem;
        const T* sv = lb_pair(m, m.S, pos); const T* y = lb_pair(m, m.Y, pos);
        const T ai = lb_dot(c, m, sv, v) / m.ys[pos];
        c.single([&] { m.alpha[i] = ai; });
        c.each(n, [&](int k) { v[k] -= ai * y[k]; });
    }
    lb_base_solve(c, m, v);
    for (int i = 0; i < cnt; ++i) {
        const int pos = (m.hh + i) % m.mem;
        const T* sv = lb_pair(m, m.S, pos); const T* y = lb_pair(m, m.Y, pos);
        const T cf = m.alpha[i] - lb_dot(c, m, y, v) / m.ys[pos];
        c.each(n, [&](int k) { v[k] += cf * sv[k]; });
    }
}

// Diagonal of B: B0's, plus each pair's y_k^2 / ys - a_k^2 / sa.
template <class C, class T>
KOKKOS_INLINE_FUNCTION void lb_diagonal(const C& c, const Model<T>& m, T* out) {
    const int n = m.n;
    c.each(n, [&](int k) {
        T d = m.gamma * (m.P ? m.P[k * n + k] : (m.D ? m.D[k] : T(1)));
        for (int i = 0; i < m.hc; ++i) {
            const int pos = (m.hh + i) % m.mem;
            const T yk = lb_pair(m, m.Y, pos)[k], ak = lb_pair(m, m.A, pos)[k];
            d += yk * yk / m.ys[pos] - ak * ak / m.sa[pos];
        }
        out[k] = d;
    });
}

template <class T>
KOKKOS_INLINE_FUNCTION void lb_reset(Model<T>& m, T gamma0) { m.hc = 0; m.hh = 0; m.gamma = gamma0; }

// Learn from step p and the gradient change across it (gtrial - gfrom),
// Powell-damped. Uses q, and bg, which must hold B p on entry when bp_ready
// (lb_dogleg leaves it there).
template <class C, class T>
KOKKOS_INLINE_FUNCTION void lb_update(const C& c, Model<T>& m, const T* p, const T* gtrial, const T* gfrom,
                                      bool bp_ready = false) {
    const int n = m.n;
    if (!bp_ready) lb_matvec(c, m, p, m.bg);
    const T curv = lb_dot(c, m, p, m.bg);
    if (!(curv > 0)) return;
    c.each(n, [&](int k) { m.q[k] = gtrial[k] - gfrom[k]; });
    const T along = lb_dot(c, m, m.q, p);
    if (along < T(0.2) * curv) {
        const T th = T(0.8) * curv / (curv - along);
        c.each(n, [&](int k) { m.q[k] = th * m.q[k] + (T(1) - th) * m.bg[k]; });
    }
    const T den = lb_dot(c, m, m.q, p);
    if (!(den > T(1e-30))) return;
    if (m.fold && m.D && m.hc == m.mem) {
        // q (the new y) is still needed, so the diagonal goes through bg.
        lb_diagonal(c, m, m.bg);
        const T ig = T(1) / m.gamma;
        c.each(n, [&](int k) { m.D[k] = m.bg[k] * ig; });
        m.hc = 0; m.hh = 0;
    }
    int pos;
    const bool evict = m.hc == m.mem;
    if (!evict) { pos = (m.hh + m.hc) % m.mem; m.hc += 1; }
    else { pos = m.hh; m.hh = (m.hh + 1) % m.mem; }
    T* sp = lb_pair(m, m.S, pos); T* yp = lb_pair(m, m.Y, pos);
    c.each(n, [&](int k) { sp[k] = p[k]; yp[k] = m.q[k]; });
    lb_shape_matvec(c, m, sp, lb_pair(m, m.PS, pos));
    c.single([&] { m.ys[pos] = den; });
    if (m.rule == Gamma::Scaled) m.gamma = lb_dot(c, m, m.q, m.q) / den;
    lb_refresh(c, m, (m.rule == Gamma::Fixed && !evict) ? m.hc - 1 : 0);
}

// -(g.p + p.B.p / 2), with bg = B p.
template <class C, class T>
KOKKOS_INLINE_FUNCTION T lb_reduction(const C& c, const Model<T>& m, const T* g, const T* p) {
    return -(lb_dot(c, m, g, p) + T(0.5) * lb_dot(c, m, p, m.bg));
}

// Same, computing B p into bg first.
template <class C, class T>
KOKKOS_INLINE_FUNCTION T lb_predicted(const C& c, const Model<T>& m, const T* g, const T* p) {
    lb_matvec(c, m, p, m.bg);
    return lb_reduction(c, m, g, p);
}

// Minimise the model in ||p|| <= delta (Powell's dogleg) from gradient g. Writes
// p, leaves B p in bg, and returns the predicted reduction. Uses nw (scratch [n]).
template <class C, class T>
KOKKOS_INLINE_FUNCTION T lb_dogleg(const C& c, const Model<T>& m, const T* g, T delta, T* p, T* nw) {
    const int n = m.n;
    const T gg = lb_dot(c, m, g, g), gn = Kokkos::sqrt(gg);
    if (gn <= T(1e-30)) { c.each(n, [&](int k) { p[k] = 0; m.bg[k] = 0; }); return 0; }
    c.each(n, [&](int k) { nw[k] = -g[k]; });
    lb_solve(c, m, nw);                                   // Newton step
    if (Kokkos::sqrt(lb_dot(c, m, nw, nw)) <= delta) {
        c.each(n, [&](int k) { p[k] = nw[k]; m.bg[k] = -g[k]; });
        return lb_reduction(c, m, g, p);
    }
    lb_matvec(c, m, g, m.bg);                             // bg = B g
    const T curv = lb_dot(c, m, g, m.bg);
    if (!(curv > 0) || (gg / curv) * gn >= delta) {      // steepest descent to the boundary
        const T cb = -delta / gn;
        c.each(n, [&](int k) { p[k] = cb * g[k]; m.bg[k] *= cb; });
        return lb_reduction(c, m, g, p);
    }
    const T cc = -gg / curv;                              // Cauchy point = cc * g
    const T a = lb_dot(c, m, nw, nw) - 2 * cc * lb_dot(c, m, nw, g) + cc * cc * gg;   // |nw - cg|^2
    const T b = T(2) * (cc * lb_dot(c, m, g, nw) - cc * cc * gg);                     // 2 cg.(nw - cg)
    const T cq = cc * cc * gg - delta * delta;
    const T disc = b * b - 4 * a * cq;
    T tau = (a > 0 && disc > 0) ? (-b + Kokkos::sqrt(disc)) / (2 * a) : T(1);
    tau = Kokkos::fmin(Kokkos::fmax(tau, T(0)), T(1));
    // p = cc g + tau (nw - cc g); B p = cc B g + tau (-g - cc B g).
    c.each(n, [&](int k) {
        const T ck = cc * g[k], bgk = cc * m.bg[k];
        p[k] = ck + tau * (nw[k] - ck);
        m.bg[k] = bgk + tau * (-g[k] - bgk);
    });
    return lb_reduction(c, m, g, p);
}

// Accept or reject from actual against predicted reduction; changes below the
// energy's resolution (noise) are not used as evidence.
template <class T>
KOKKOS_INLINE_FUNCTION void lb_judge(T e_trial, T e, T pred, T eta, T rho_max, T noise,
                                     T& rho, bool& accept, bool& unusable) {
    const bool finite = Kokkos::isfinite(e_trial);
    if (!(finite && pred > T(1e-30))) { rho = -1; accept = false; unusable = !finite; return; }
    unusable = false;
    if (pred <= noise) { accept = e_trial <= e + noise; rho = accept ? T(1) : T(-1); return; }
    rho = (e - e_trial + noise) / pred;
    unusable = rho > rho_max;
    accept = !unusable && rho > eta;
}

}  // namespace klasp::tr
