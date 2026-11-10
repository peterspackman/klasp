#pragma once
/*
 * Clash detection and resolution between two sets of spheres, built on
 * KokkosSphereHull.
 *
 * 1. findClashes(A, B): exact sphere-pair overlaps, count + deepest pair, one
 *    kernel over the N*M pairs.
 *
 * 2. hullMTV(A, B): exact signed distance / minimum translation vector between
 *    the convex hulls of the two sets. The Minkowski difference is the ball hull
 *    of the difference spheres {c_i - d_j, r_i + s_j}, with support function
 *        h(u) = max_i(c_i.u + r_i) + max_j(-d_j.u + s_j),
 *    and s* = min_{|u|=1} h(u) (> 0 penetration, < 0 clearance); shifting B by
 *    s*.u* makes the hulls touch. The minimiser is found by enumerating the
 *    critical points of h (facet normals, cap directions -m^, closed-form arc
 *    minima), each re-evaluated against the full sets so the result is feasible.
 *
 * 3. resolve(A, B): union-level resolution. Along a direction u each pair
 *    forbids the s-interval between the roots of |v_ij - s u| = r_i + s_j; the
 *    first uncovered s >= 0 is the exact minimal step. The direction is chosen
 *    by multi-start plus shrinking-cone refinement; the result is feasible and
 *    never worse than the hull MTV.
 *
 * upload() can hull-reduce a set once; the reduction is invariant under rigid
 * motion, so hullMTV works on H_A * H_B difference spheres.
 */

#include "kokkos_quickhull.hpp"
#include "kokkos_spherehull.hpp"

#include <algorithm>
#include <cstring>
#include <random>

namespace kokkos_quickhull {

template <typename T> struct ClashReport {
  size_t count = 0;     // overlapping pairs (depth > margin-adjusted 0)
  T worstDepth = 0;     // max over pairs of (r_i + s_j + margin - |c_i - d_j|)
  size_t worstA = 0, worstB = 0;
  bool clashing() const { return count > 0; }
};

template <typename T> struct HullMTV {
  T signedDepth = 0;    // >0: hull penetration depth; <0: -clearance
  Vec3<T> direction{};  // unit u*; shift set B by (signedDepth)*direction
                        // (plus any desired margin) to make the hulls touch
  bool overlapping() const { return signedDepth > 0; }
  Vec3<T> translation(T margin = 0) const {
    const T s = signedDepth > 0 ? signedDepth + margin : T(0);
    return scaled(direction, s);
  }
};

template <typename T> struct Resolution {
  T distance = 0;       // |t|; 0 if there was nothing to resolve
  Vec3<T> direction{};  // unit
  Vec3<T> translation{};// apply to set B
  T hullDepth = 0;      // convex MTV magnitude for comparison/upper bound
};

// ===========================================================================
template <typename T, typename IndexT = std::uint32_t,
          typename ExecSpace = Kokkos::DefaultExecutionSpace>
class KokkosSphereClash {
public:
  using MemSpace = typename ExecSpace::memory_space;
  using CView = Kokkos::View<T *[3], Kokkos::LayoutLeft, MemSpace>;
  using RView = Kokkos::View<T *, MemSpace>;

  // A sphere set uploaded once and queried many times. `red*` hold the
  // hull-reduced subset (== the full set if reduce=false): the reduction is
  // invariant under rigid motion of the set, so it can be reused for every
  // placement of the same molecule/object.
  struct DeviceSet {
    CView c;
    RView r;
    size_t n = 0;
    std::vector<T> hc, hr;          // host copies (full)
    std::vector<T> redC, redR;      // host reduced subset (xyz / r)
    std::vector<size_t> hullIdx;    // reduced -> original indices
    size_t nRed() const { return redR.size(); }
  };

  DeviceSet upload(const T *centers, const T *radii, size_t n,
                   bool reduce = true, T eps = defaultEps<T>()) {
    DeviceSet S;
    S.n = n;
    S.c = CView(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                   std::string("clash::c")),
                n);
    S.r = RView(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                   std::string("clash::r")),
                n);
    auto hcm = Kokkos::create_mirror_view(S.c);
    auto hrm = Kokkos::create_mirror_view(S.r);
    S.hc.assign(centers, centers + 3 * n);
    S.hr.assign(radii, radii + n);
    for (size_t i = 0; i < n; i++) {
      hcm(i, 0) = centers[3 * i];
      hcm(i, 1) = centers[3 * i + 1];
      hcm(i, 2) = centers[3 * i + 2];
      hrm(i) = radii[i];
    }
    Kokkos::deep_copy(S.c, hcm);
    Kokkos::deep_copy(S.r, hrm);

    if (reduce && n > 4) {
      auto hull = m_hull.getSphereHull(centers, radii, n, true, eps);
      S.hullIdx = hull.sphereSourceIndices();
    } else {
      S.hullIdx.resize(n);
      for (size_t i = 0; i < n; i++)
        S.hullIdx[i] = i;
    }
    S.redC.reserve(3 * S.hullIdx.size());
    S.redR.reserve(S.hullIdx.size());
    for (size_t i : S.hullIdx) {
      S.redC.insert(S.redC.end(),
                    {centers[3 * i], centers[3 * i + 1], centers[3 * i + 2]});
      S.redR.push_back(radii[i]);
    }
    return S;
  }

  // -------------------------------------------------------------------------
  // 1) exact union-level clash detection: one kernel over the N*M pairs.
  // -------------------------------------------------------------------------
  ClashReport<T> findClashes(const DeviceSet &A, const DeviceSet &B,
                             T margin = T(0)) {
    ClashReport<T> rep;
    if (A.n == 0 || B.n == 0)
      return rep;
    const auto ac = A.c, bc = B.c;
    const auto ar = A.r, br = B.r;
    const size_t M = B.n, total = A.n * B.n;
    const T mg = margin;

    using Reducer = Kokkos::MaxLoc<T, size_t>;
    typename Reducer::value_type mx;
    size_t cnt = 0;
    Kokkos::parallel_reduce(
        "clash::pairs", Kokkos::RangePolicy<ExecSpace>(0, total),
        KOKKOS_LAMBDA(const size_t w, typename Reducer::value_type &upd,
                      size_t &c) {
          const size_t i = w / M, j = w % M;
          const T dx = ac(i, 0) - bc(j, 0), dy = ac(i, 1) - bc(j, 1),
                  dz = ac(i, 2) - bc(j, 2);
          const T R = ar(i) + br(j) + mg;
          const T depth = R - Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
          if (depth > 0)
            c++;
          if (depth > upd.val) {
            upd.val = depth;
            upd.loc = w;
          }
        },
        Reducer(mx), cnt);
    rep.count = cnt;
    rep.worstDepth = mx.val;
    rep.worstA = mx.loc / M;
    rep.worstB = mx.loc % M;
    return rep;
  }

  // -------------------------------------------------------------------------
  // 2) exact hull-level signed distance + minimum translation vector.
  // -------------------------------------------------------------------------
  HullMTV<T> hullMTV(const DeviceSet &A, const DeviceSet &B,
                     T eps = defaultEps<T>()) {
    HullMTV<T> out;
    const size_t Ha = A.nRed(), Hb = B.nRed();
    if (Ha == 0 || Hb == 0)
      return out;

    // Difference spheres over the hull-reduced sets:
    //   m_{ij} = c_i - d_j,  rho_{ij} = r_i + s_j
    m_diffC.resize(3 * Ha * Hb);
    m_diffR.resize(Ha * Hb);
    size_t w = 0;
    for (size_t i = 0; i < Ha; i++)
      for (size_t j = 0; j < Hb; j++, w++) {
        m_diffC[3 * w] = A.redC[3 * i] - B.redC[3 * j];
        m_diffC[3 * w + 1] = A.redC[3 * i + 1] - B.redC[3 * j + 1];
        m_diffC[3 * w + 2] = A.redC[3 * i + 2] - B.redC[3 * j + 2];
        m_diffR[w] = A.redR[i] + B.redR[j];
      }

    // Ball hull of the difference set -> the normal-cone cell structure of
    // h on the unit sphere. (The hull spheres are its cells, the facet
    // normals its vertices, the facet edges its arcs.)
    auto dh = m_hull.getSphereHull(m_diffC.data(), m_diffR.data(), Ha * Hb,
                                   true, eps);

    // candidate directions for argmin h
    m_cands.clear();
    const auto &hs = dh.sphereSourceIndices();
    // (a) per-cell interior critical point u = -m^
    for (size_t k : hs) {
      const Vec3<T> m{m_diffC[3 * k], m_diffC[3 * k + 1], m_diffC[3 * k + 2]};
      const T L2 = m.squaredNorm();
      if (L2 > T(0))
        m_cands.push_back(scaled(m, -T(1) / std::sqrt(L2)));
    }
    // (b) cell vertices = facet normals (already unit)
    for (const Plane<T> &P : dh.facePlanes())
      m_cands.push_back(P.n);
    // (c) cell arcs: per facet edge (a, b), the closed-form minimum of
    //     m_a.u + rho_a on the circle {u: (m_a - m_b).u = rho_b - rho_a}
    const auto &tri = dh.indices();
    m_edges.clear();
    for (size_t f = 0; f < tri.size() / 3; f++) {
      const size_t v[3] = {tri[3 * f], tri[3 * f + 1], tri[3 * f + 2]};
      for (int e = 0; e < 3; e++) {
        size_t a = v[e], b = v[(e + 1) % 3];
        if (a == b)
          continue; // degenerate facet
        if (a > b)
          std::swap(a, b);
        m_edges.push_back(a * (Ha * Hb) + b);
      }
    }
    std::sort(m_edges.begin(), m_edges.end());
    m_edges.erase(std::unique(m_edges.begin(), m_edges.end()), m_edges.end());
    for (size_t code : m_edges) {
      const size_t a = code / (Ha * Hb), b = code % (Ha * Hb);
      const Vec3<T> ma{m_diffC[3 * a], m_diffC[3 * a + 1], m_diffC[3 * a + 2]};
      const Vec3<T> mb{m_diffC[3 * b], m_diffC[3 * b + 1], m_diffC[3 * b + 2]};
      const Vec3<T> wv = ma - mb;
      const T wl2 = wv.squaredNorm();
      if (wl2 <= T(0))
        continue;
      const T wl = std::sqrt(wl2);
      const T t0 = (m_diffR[b] - m_diffR[a]) / wl; // u.w^ on the circle
      if (t0 <= T(-1) || t0 >= T(1))
        continue;
      const Vec3<T> wh = scaled(wv, T(1) / wl);
      // minimize ma.u on the circle: in-circle component opposite ma_perp
      const Vec3<T> maPerp = ma - scaled(wh, ma.dot(wh));
      const T pl2 = maPerp.squaredNorm();
      const T s = std::sqrt(std::max(T(0), T(1) - t0 * t0));
      Vec3<T> u;
      if (pl2 > T(0))
        u = scaled(wh, t0) + scaled(maPerp, -s / std::sqrt(pl2));
      else
        u = scaled(wh, t0); // ma parallel to w: any circle point; project
      const T ul2 = u.squaredNorm();
      if (ul2 > T(0))
        m_cands.push_back(scaled(u, T(1) / std::sqrt(ul2)));
    }

    // Evaluate every candidate against the FULL sets (exact h, so the
    // returned vector is feasible regardless of hull epsilon slop).
    T best = std::numeric_limits<T>::max();
    Vec3<T> bestU{0, 0, 1};
    for (const Vec3<T> &u : m_cands) {
      const T h = supportFull(A, u) + supportFullNeg(B, u);
      if (h < best) {
        best = h;
        bestU = u;
      }
    }
    out.signedDepth = best;
    out.direction = bestU;
    return out;
  }

  // -------------------------------------------------------------------------
  // 3) union-level resolution: smallest translation found that clears every
  //    sphere-pair overlap (gap-aware, so it can thread holes the convex
  //    hull cannot).
  // -------------------------------------------------------------------------
  Resolution<T> resolve(const DeviceSet &A, const DeviceSet &B,
                        T margin = T(0), int sampleDirs = 64,
                        int refineRounds = 3) {
    Resolution<T> res;
    const ClashReport<T> rep = findClashes(A, B, margin);
    const HullMTV<T> mtv = hullMTV(A, B);
    res.hullDepth = std::max(T(0), mtv.signedDepth + margin);
    if (!rep.clashing())
      return res; // nothing to do

    // PREFILTER: a pair can only forbid a step s <= bound0 if
    // |v| <= bound0 + R (its forbidden interval starts at s1 >= |v| - R).
    // The hull MTV is a feasible escape, so bound0 = hullDepth is sound and
    // shrinks the per-direction work from N*M pairs to the near pairs only.
    const T bound0 = res.hullDepth * (T(1) + T(1e-12)) + T(1e-12);
    gatherCandidatePairs(A, B, margin, bound0);

    // Cheap per-direction LOWER BOUND: every currently clashing pair's
    // forbidden interval contains s = 0 (it is overlapping now), so the
    // escape along u is at least its interval end s2(u) = u.v + sqrt(disc).
    // Cache the clashing pairs once (host) and use the bound to skip the
    // full interval kernel for hopeless directions.
    m_clashV.clear();
    m_clashR.clear();
    { // host-side filter of the gathered near pairs down to clashing ones
      auto hc = Kokkos::create_mirror_view(m_dCand);
      Kokkos::deep_copy(
          Kokkos::subview(hc, std::make_pair(size_t(0), m_candCount)),
          Kokkos::subview(m_dCand, std::make_pair(size_t(0), m_candCount)));
      for (size_t t = 0; t < m_candCount; t++) {
        const size_t w = hc(t), i = w / B.n, j = w % B.n;
        const Vec3<T> v{A.hc[3 * i] - B.hc[3 * j],
                        A.hc[3 * i + 1] - B.hc[3 * j + 1],
                        A.hc[3 * i + 2] - B.hc[3 * j + 2]};
        const T R = A.hr[i] + B.hr[j] + margin;
        if (v.squaredNorm() < R * R) {
          m_clashV.push_back(v);
          m_clashR.push_back(R);
        }
      }
    }
    auto lowerBound = [&](const Vec3<T> &u) {
      T lb = 0;
      for (size_t t = 0; t < m_clashV.size(); t++) {
        const T b = u.dot(m_clashV[t]);
        const T disc =
            b * b - (m_clashV[t].squaredNorm() - m_clashR[t] * m_clashR[t]);
        lb = std::max(lb, b + std::sqrt(disc)); // disc > 0: pair overlaps now
      }
      return lb;
    };

    // seed directions
    std::vector<Vec3<T>> dirs;
    if (mtv.signedDepth > 0)
      dirs.push_back(mtv.direction);
    { // deepest-clash direction: push B away from A's worst sphere
      const size_t i = rep.worstA, j = rep.worstB;
      Vec3<T> v{B.hc[3 * j] - A.hc[3 * i], B.hc[3 * j + 1] - A.hc[3 * i + 1],
                B.hc[3 * j + 2] - A.hc[3 * i + 2]};
      const T L2 = v.squaredNorm();
      if (L2 > T(0))
        dirs.push_back(scaled(v, T(1) / std::sqrt(L2)));
    }
    { // Fibonacci sample
      const T ga = T(M_PI) * (T(3) - std::sqrt(T(5)));
      for (int i = 0; i < sampleDirs; i++) {
        const T z = T(1) - T(2) * (T(i) + T(0.5)) / T(sampleDirs);
        const T rr = std::sqrt(std::max(T(0), T(1) - z * z));
        dirs.push_back({rr * std::cos(ga * T(i)), rr * std::sin(ga * T(i)), z});
      }
    }

    T bestS = bound0;
    Vec3<T> bestU = mtv.direction;
    size_t budget = m_candCount < size_t(100000)
                        ? std::numeric_limits<size_t>::max()
                        : (size_t)std::max(8, sampleDirs / 4);
    auto tryDir = [&](const Vec3<T> &u) {
      if (budget == 0 || lowerBound(u) >= bestS)
        return; // hopeless or out of budget; skip the kernel
      budget--;
      const T s = escapeAlong<true>(A, B, u, margin, bestS);
      if (s < bestS) {
        bestS = s;
        bestU = u;
      }
    };
    // Probe the seed directions, then the samples in lower-bound order
    // under a kernel budget (directions whose LB already exceeds the best
    // are free to skip; the budget caps the rest -- mainly a CPU concern,
    // each probe is one small kernel on a GPU).
    tryDir(dirs[0]);
    if (dirs.size() > 1)
      tryDir(dirs[1]);
    {
      std::vector<std::pair<T, size_t>> order;
      for (size_t i = 2; i < dirs.size(); i++)
        order.push_back({lowerBound(dirs[i]), i});
      std::sort(order.begin(), order.end());
      for (const auto &[lb, i] : order) {
        if (lb >= bestS || budget == 0)
          break;
        tryDir(dirs[i]);
      }
    }

    // shrinking-cone refinement around the best direction
    std::mt19937_64 rng(42);
    std::normal_distribution<T> g(0, 1);
    T cone = T(0.5);
    for (int round = 0; round < refineRounds; round++, cone *= T(0.4)) {
      const Vec3<T> u0 = bestU;
      for (int s = 0; s < 16; s++) {
        Vec3<T> u = u0 + scaled(Vec3<T>{g(rng), g(rng), g(rng)}, cone);
        const T L2 = u.squaredNorm();
        if (L2 <= T(0))
          continue;
        tryDir(scaled(u, T(1) / std::sqrt(L2)));
      }
    }

    // verification: recompute the chosen direction's escape over ALL pairs
    // with no bound (airtight feasibility, independent of the prefilter)
    bestS = escapeAlong<false>(A, B, bestU, margin,
                               std::numeric_limits<T>::max());

    res.distance = bestS;
    res.direction = bestU;
    res.translation = scaled(bestU, bestS);
    return res;
  }

  // Public because nvcc forbids KOKKOS_LAMBDA in private member functions.
  // Treat everything here as implementation detail.
public:
  KokkosSphereHull<T, IndexT, ExecSpace> m_hull;
  std::vector<T> m_diffC, m_diffR;
  std::vector<Vec3<T>> m_cands;
  std::vector<size_t> m_edges;
  Kokkos::View<T *, MemSpace> m_dIv; // gathered (s1, s2) interval pairs
  Kokkos::View<int, MemSpace> m_dIvCount;
  Kokkos::View<size_t *, MemSpace> m_dCand; // prefiltered near pairs
  Kokkos::View<T *[4], Kokkos::LayoutLeft, MemSpace> m_dPV; // packed v,R
  size_t m_candCount = 0;
  std::vector<Vec3<T>> m_clashV; // cached clashing pairs for the LB test
  std::vector<T> m_clashR;

  // h_A(u) over the full set (host scan; exact)
  static T supportFull(const DeviceSet &S, const Vec3<T> &u) {
    T best = -std::numeric_limits<T>::max();
    for (size_t i = 0; i < S.n; i++)
      best = std::max(best, u.x * S.hc[3 * i] + u.y * S.hc[3 * i + 1] +
                                u.z * S.hc[3 * i + 2] + S.hr[i]);
    return best;
  }
  // h_B(-u)
  static T supportFullNeg(const DeviceSet &S, const Vec3<T> &u) {
    T best = -std::numeric_limits<T>::max();
    for (size_t i = 0; i < S.n; i++)
      best = std::max(best, -(u.x * S.hc[3 * i] + u.y * S.hc[3 * i + 1] +
                              u.z * S.hc[3 * i + 2]) +
                                S.hr[i]);
    return best;
  }

  // Exact minimal step s >= 0 along unit u clearing every pair overlap:
  // pair (i,j) forbids the s-interval between the roots of
  //   s^2 - 2 s (u.v) + |v|^2 - R^2 = 0,  v = c_i - d_j, R = r_i + s_j + mg.
  // One kernel gathers the (s1, s2) intervals with s2 > 0 (and s1 below the
  // current best, since larger ones cannot affect the first gap <= bound);
  // the host merges and returns the first uncovered s. `bound` lets callers
  // prune directions that cannot beat the current best. UseCand iterates the
  // prefiltered near-pair list instead of all N*M pairs.
  template <bool UseCand>
  T escapeAlong(const DeviceSet &A, const DeviceSet &B, const Vec3<T> &u,
                T margin, T bound) {
    const size_t M = B.n;
    const size_t total = UseCand ? m_candCount : A.n * B.n;
    ensureIv(4096);
    for (;;) {
      const auto ac = A.c, bc = B.c;
      const auto ar = A.r, br = B.r;
      const auto iv = m_dIv;
      const auto ivc = m_dIvCount;
      const auto pv = m_dPV;
      const int cap = (int)(m_dIv.extent(0) / 2);
      const T ux = u.x, uy = u.y, uz = u.z, mg = margin;
      const T sBound = bound;
      Kokkos::deep_copy(m_dIvCount, 0);
      Kokkos::parallel_for(
          "clash::intervals", Kokkos::RangePolicy<ExecSpace>(0, total),
          KOKKOS_LAMBDA(const size_t ww) {
            T vx, vy, vz, R;
            if (UseCand) {
              vx = pv(ww, 0);
              vy = pv(ww, 1);
              vz = pv(ww, 2);
              R = pv(ww, 3);
            } else {
              const size_t i = ww / M, j = ww % M;
              vx = ac(i, 0) - bc(j, 0);
              vy = ac(i, 1) - bc(j, 1);
              vz = ac(i, 2) - bc(j, 2);
              R = ar(i) + br(j) + mg;
            }
            const T b = ux * vx + uy * vy + uz * vz;
            const T disc = b * b - (vx * vx + vy * vy + vz * vz - R * R);
            if (disc <= 0)
              return;
            const T sq = Kokkos::sqrt(disc);
            const T s1 = b - sq, s2 = b + sq;
            if (s2 <= 0 || s1 >= sBound)
              return; // cannot affect the first gap below the bound
            const int slot = Kokkos::atomic_fetch_add(&ivc(), 1);
            if (slot < cap) {
              iv(2 * slot) = s1;
              iv(2 * slot + 1) = s2;
            }
          });
      int count = 0;
      Kokkos::deep_copy(count, m_dIvCount);
      Kokkos::fence();
      if (count <= cap) {
        // merge on host
        const int n = count;
        auto hiv = Kokkos::create_mirror_view(m_dIv);
        Kokkos::deep_copy(
            Kokkos::subview(hiv, std::make_pair(size_t(0), size_t(2 * n))),
            Kokkos::subview(m_dIv, std::make_pair(size_t(0), size_t(2 * n))));
        m_iv.resize(n);
        for (int i = 0; i < n; i++)
          m_iv[i] = {hiv(2 * i), hiv(2 * i + 1)};
        std::sort(m_iv.begin(), m_iv.end());
        T s = 0;
        for (const auto &p : m_iv) {
          if (p.first > s)
            break; // gap found before this interval
          s = std::max(s, p.second);
          if (s >= bound)
            break; // can't beat current best; bail
        }
        return s;
      }
      ensureIv(size_t(2) * count + 64); // overflow: grow and retry
    }
  }

  // Gather pairs with |v| <= bound + R into m_dCand (packed w = i*M + j).
  void gatherCandidatePairs(const DeviceSet &A, const DeviceSet &B, T margin,
                            T bound) {
    const size_t M = B.n, total = A.n * B.n;
    if (m_dCand.extent(0) < 4096)
      m_dCand = Kokkos::View<size_t *, MemSpace>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             std::string("clash::cand")),
          4096);
    if (!m_dIvCount.data())
      m_dIvCount = Kokkos::View<int, MemSpace>(std::string("clash::ivcount"));
    for (;;) {
      const auto ac = A.c, bc = B.c;
      const auto ar = A.r, br = B.r;
      const auto cand = m_dCand;
      const auto cnt = m_dIvCount;
      const int cap = (int)m_dCand.extent(0);
      const T mg = margin, bd = bound;
      Kokkos::deep_copy(m_dIvCount, 0);
      Kokkos::parallel_for(
          "clash::gather", Kokkos::RangePolicy<ExecSpace>(0, total),
          KOKKOS_LAMBDA(const size_t w) {
            const size_t i = w / M, j = w % M;
            const T vx = ac(i, 0) - bc(j, 0), vy = ac(i, 1) - bc(j, 1),
                    vz = ac(i, 2) - bc(j, 2);
            const T reach = ar(i) + br(j) + mg + bd;
            if (vx * vx + vy * vy + vz * vz > reach * reach)
              return;
            const int slot = Kokkos::atomic_fetch_add(&cnt(), 1);
            if (slot < cap)
              cand(slot) = w;
          });
      int count = 0;
      Kokkos::deep_copy(count, m_dIvCount);
      Kokkos::fence();
      if (count <= cap) {
        m_candCount = (size_t)count;
        // pack (v, R) of the candidates into flat arrays: the per-direction
        // kernel then does 4 coalesced loads and no index arithmetic
        if (m_dPV.extent(0) < m_candCount)
          m_dPV = Kokkos::View<T *[4], Kokkos::LayoutLeft, MemSpace>(
              Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                 std::string("clash::pv")),
              m_candCount);
        const auto pv = m_dPV;
        Kokkos::parallel_for(
            "clash::pack", Kokkos::RangePolicy<ExecSpace>(0, m_candCount),
            KOKKOS_LAMBDA(const size_t t) {
              const size_t w = cand(t), i = w / M, j = w % M;
              pv(t, 0) = ac(i, 0) - bc(j, 0);
              pv(t, 1) = ac(i, 1) - bc(j, 1);
              pv(t, 2) = ac(i, 2) - bc(j, 2);
              pv(t, 3) = ar(i) + br(j) + mg;
            });
        Kokkos::fence();
        return;
      }
      m_dCand = Kokkos::View<size_t *, MemSpace>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             std::string("clash::cand")),
          size_t(2) * count + 64);
    }
  }

  void ensureIv(size_t pairs) {
    if (m_dIv.extent(0) < 2 * pairs)
      m_dIv = Kokkos::View<T *, MemSpace>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             std::string("clash::iv")),
          2 * pairs);
    if (!m_dIvCount.data())
      m_dIvCount = Kokkos::View<int, MemSpace>(std::string("clash::ivcount"));
  }

  std::vector<std::pair<T, T>> m_iv;
};

} // namespace kokkos_quickhull
