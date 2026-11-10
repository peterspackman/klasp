#pragma once
/*
//  * Convex hull of spheres (centers + radii): the Kokkos QuickHull port
//  * specialised to additively weighted point sets. A sphere (c_i, r_i)
//  * contributes iff some unit u has
//  *     c_i . u + r_i  >=  c_j . u + r_j   for all j.
//  * Point-above-plane tests become tangent distances,
//  *     dist = n . c + d + r          (n kept unit length),
//  * and planes through 3 points become outer tritangent planes of 3 spheres.
//  *
//  *   - Equal radii reduce to the point hull of the centers.
//  *   - Contained spheres need no pre-filtering.
//  *   - Output: triangles of sphere indices, each facet's unit-normal tangent
//  *     plane, and the contributing spheres.
//  *
//  * Degenerate (coplanar/collinear/coincident) centers use the point code's
//  * synthetic extra vertex, with radius 0.
//  *
//  * Requires kokkos_quickhull.hpp (Vec3, Plane, MeshBuilder are reused).
//  */

#include "kokkos_quickhull.hpp"
#include <algorithm>

namespace kokkos_quickhull {

template <typename T>
KOKKOS_INLINE_FUNCTION Vec3<T> scaled(const Vec3<T> &v, T s) {
  return {v.x * s, v.y * s, v.z * s};
}

struct SphereDiagnostics {
  size_t m_failedHorizonEdges = 0;
  // tritangent solves that had no real solution (near-engulfing sphere pairs
  // or collinear centers) and fell back to a best-effort supporting plane
  size_t m_degenerateTangentPlanes = 0;
};

// ---------------------------------------------------------------------------
// Result. Indices reference the original sphere array.
// ---------------------------------------------------------------------------
template <typename T> class SphereHull {
public:
  // facet f = spheres (m_indices[3f], m_indices[3f+1], m_indices[3f+2]),
  // tangent plane m_facePlanes[f] (unit outward normal; n.c + d + r == 0 for
  // the three supporting spheres, <= 0 for every sphere)
  std::vector<size_t> m_indices;
  std::vector<Plane<T>> m_facePlanes;
  std::vector<size_t> m_hullSphereSource; // unique contributing spheres
  std::vector<size_t> m_reducedIndices;   // m_indices remapped into the above

  const std::vector<size_t> &indices() const { return m_indices; }
  const std::vector<Plane<T>> &facePlanes() const { return m_facePlanes; }
  const std::vector<size_t> &sphereSourceIndices() const {
    return m_hullSphereSource;
  }
  const std::vector<size_t> &reducedIndices() const { return m_reducedIndices; }
};

// KokkosSphereHull: O(N) sphere work on the device, O(H) tangent-plane
// topology on the host.
template <typename T, typename IndexT = std::uint32_t,
          typename ExecSpace = Kokkos::DefaultExecutionSpace>
class KokkosSphereHull {
public:
  using MemSpace = typename ExecSpace::memory_space;
  using PointsView = Kokkos::View<T *[3], Kokkos::LayoutLeft, MemSpace>;
  using PointsHost = typename PointsView::host_mirror_type;
  using RadiiView = Kokkos::View<T *, MemSpace>;
  using RadiiHost = typename RadiiView::host_mirror_type;
  using Mesh = MeshBuilder<T, IndexT>;
  using Face = typename Mesh::Face;

public:
  // Public because nvcc rejects private nested types in __global__
  // instantiations. Normals are unit, so the epsilon test is dist > eps.
  struct DevPlane {
    T nx, ny, nz, d;
  };
  struct Segment {
    IndexT start;
    IndexT count;
  };
  struct MaxRes {
    T dist;
    IndexT idx;
  };

private:
  // --- persistent state ---
  PointsView m_points; // sphere centers (capacity N+1: synthetic vertex)
  PointsHost m_pointsHost;
  RadiiView m_radii;
  RadiiHost m_radiiHost;
  size_t m_numPoints = 0;

  Kokkos::View<IndexT *, MemSpace> m_arena;
  size_t m_arenaTail = 0;

  Kokkos::View<Segment *, MemSpace> m_dSegments;
  typename Kokkos::View<Segment *, MemSpace>::host_mirror_type m_hSegments;
  Kokkos::View<IndexT *, MemSpace> m_dSegPrefix;
  typename Kokkos::View<IndexT *, MemSpace>::host_mirror_type m_hSegPrefix;
  Kokkos::View<DevPlane *, MemSpace> m_dPlanes;
  typename Kokkos::View<DevPlane *, MemSpace>::host_mirror_type m_hPlanes;
  Kokkos::View<std::int32_t *, MemSpace> m_dAssign;
  Kokkos::View<IndexT *, MemSpace> m_dPointIdx;
  Kokkos::View<IndexT *, MemSpace> m_dCounts;
  typename Kokkos::View<IndexT *, MemSpace>::host_mirror_type m_hCounts;
  Kokkos::View<IndexT *, MemSpace> m_dCursors;
  Kokkos::View<IndexT *, MemSpace> m_dOffsets;
  typename Kokkos::View<IndexT *, MemSpace>::host_mirror_type m_hOffsets;
  Kokkos::View<MaxRes *, MemSpace> m_dMax;
  typename Kokkos::View<MaxRes *, MemSpace>::host_mirror_type m_hMax;

  Mesh m_mesh;
  T m_epsilon = 0, m_epsilonSquared = 0, m_scale = 0;
  bool m_planar = false;
  std::array<size_t, 6> m_extremeValues{};
  SphereDiagnostics m_diagnostics;

  std::vector<size_t> m_newFaceIndices, m_newHalfEdgeIndices, m_visibleFaces,
      m_horizonEdges;
  struct FaceData {
    size_t faceIndex;
    size_t enteredFromHalfEdge;
  };
  std::vector<FaceData> m_possiblyVisibleFaces;
  std::deque<size_t> m_faceList;
  std::vector<Segment> m_srcSegments;
  std::vector<std::vector<IndexT>> m_hostScratch;
  std::vector<Plane<T>> m_newPlanes;

  size_t m_hostPathThreshold = 4096;

public:
  const SphereDiagnostics &getDiagnostics() const { return m_diagnostics; }
  void setHostPathThreshold(size_t t) { m_hostPathThreshold = t; }

  // centers layout: x0,y0,z0,x1,y1,z1,...  radii layout: r0,r1,...
  SphereHull<T> getSphereHull(const T *centers, const T *radii,
                              size_t sphereCount, bool CCW,
                              T eps = defaultEps<T>()) {
    buildMesh(centers, radii, sphereCount, eps);
    return extractHull(CCW);
  }

  // Public because nvcc forbids KOKKOS_LAMBDA in private/protected member
  // functions. Treat everything below as implementation detail.
public:
  Vec3<T> pt(size_t i) const {
    return {m_pointsHost(i, 0), m_pointsHost(i, 1), m_pointsHost(i, 2)};
  }
  T rad(size_t i) const { return m_radiiHost(i); }

  template <typename V> static void ensure(V &v, size_t n, const char *name) {
    if (v.extent(0) < n)
      v = V(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(name)),
            std::max(n, size_t(2) * v.extent(0)));
  }
  template <typename V, typename H>
  static void ensurePair(V &v, H &h, size_t n, const char *name) {
    if (v.extent(0) < n) {
      v = V(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(name)),
            std::max(n, size_t(2) * v.extent(0)));
      h = Kokkos::create_mirror_view(v);
    }
  }

  // Outer tritangent plane of spheres (ia, ib, ic): unit n with
  //   n.(cb-ca) = ra-rb,  n.(cc-ca) = ra-rc,  |n| = 1,  d = -(n.ca + ra).
  // Of the two roots, pick the one whose normal points like refN.
  Plane<T> tritangentPlane(size_t ia, size_t ib, size_t ic,
                           const Vec3<T> &refN) {
    const Vec3<T> ca = pt(ia), cb = pt(ib), cc = pt(ic);
    const T ra = rad(ia), rb = rad(ib), rc = rad(ic);
    const Vec3<T> a = cb - ca, b = cc - ca;
    const T aa = a.dot(a), ab = a.dot(b), bb = b.dot(b);
    const T det = aa * bb - ab * ab;
    const Vec3<T> axb = a.cross(b);
    const T axb2 = axb.squaredNorm();

    if (det > T(0) && axb2 > T(0)) {
      const T r1 = ra - rb, r2 = ra - rc;
      const T alpha = (bb * r1 - ab * r2) / det;
      const T beta = (aa * r2 - ab * r1) / det;
      const Vec3<T> w = scaled(a, alpha) + scaled(b, beta); // in-plane part
      const T g2 = (T(1) - w.squaredNorm()) / axb2;
      if (g2 >= T(0)) {
        T g = std::sqrt(g2);
        if (axb.dot(refN) < T(0))
          g = -g; // root selection: top vs bottom tangent plane
        Vec3<T> n = w + scaled(axb, g);
        const T inv = T(1) / std::sqrt(n.squaredNorm());
        n = scaled(n, inv);
        Plane<T> P;
        P.n = n;
        P.d = -(n.dot(ca) + ra);
        P.sqrNLength = T(1);
        return P;
      }
    }
    // Degenerate (collinear centers / near-engulfing pair): best-effort
    // supporting plane along refN, tangent to the most protruding sphere.
    m_diagnostics.m_degenerateTangentPlanes++;
    const T len2 = refN.squaredNorm();
    Vec3<T> n = len2 > T(0) ? scaled(refN, T(1) / std::sqrt(len2))
                            : Vec3<T>(0, 0, 1);
    const T off = std::max(
        {n.dot(ca) + ra, n.dot(cb) + rb, n.dot(cc) + rc});
    Plane<T> P;
    P.n = n;
    P.d = -off;
    P.sqrNLength = T(1);
    return P;
  }

  // O(N) kernels: as the point port, with every distance gaining "+ r".

  // Extreme spheres: along +axis maximize c+r, along -axis minimize c-r.
  // (These are the spheres tangent to the axis-aligned supporting planes.)
  void getExtremeValues() {
    const auto points = m_points;
    const auto radii = m_radii;
    const size_t n = m_numPoints;
    for (int axis = 0; axis < 3; axis++) {
      using Reducer = Kokkos::MinMaxLoc<T, size_t>;
      typename Reducer::value_type r;
      Kokkos::parallel_reduce(
          "sqh::extremes", Kokkos::RangePolicy<ExecSpace>(0, n),
          KOKKOS_LAMBDA(const size_t i, typename Reducer::value_type &upd) {
            const T c = points(i, axis), rr = radii(i);
            const T vmax = c + rr, vmin = c - rr;
            if (vmax > upd.max_val) {
              upd.max_val = vmax;
              upd.max_loc = i;
            }
            if (vmin < upd.min_val) {
              upd.min_val = vmin;
              upd.min_loc = i;
            }
          },
          Reducer(r));
      m_extremeValues[2 * axis] = r.max_loc;
      m_extremeValues[2 * axis + 1] = r.min_loc;
      // scale covers center magnitude + radius -> sound epsilon scaling
      m_scale = std::max(
          m_scale, std::max(std::abs(r.max_val), std::abs(r.min_val)));
    }
  }

  // Seeding searches run on CENTERS (any nondegenerate tetrahedron of
  // centers is a valid seed; degeneracy decisions then mirror the point
  // port exactly).
  std::pair<T, size_t> farthestFromLine(const Vec3<T> &S, const Vec3<T> &V) {
    const auto points = m_points;
    const T invLen2 = T(1) / V.squaredNorm();
    const T sx = S.x, sy = S.y, sz = S.z, vx = V.x, vy = V.y, vz = V.z;
    using Reducer = Kokkos::MaxLoc<T, size_t>;
    typename Reducer::value_type r;
    Kokkos::parallel_reduce(
        "sqh::farthest_line", Kokkos::RangePolicy<ExecSpace>(0, m_numPoints),
        KOKKOS_LAMBDA(const size_t i, typename Reducer::value_type &upd) {
          const T dx = points(i, 0) - sx, dy = points(i, 1) - sy,
                  dz = points(i, 2) - sz;
          const T t = dx * vx + dy * vy + dz * vz;
          const T d2 = dx * dx + dy * dy + dz * dz - t * t * invLen2;
          if (d2 > upd.val) {
            upd.val = d2;
            upd.loc = i;
          }
        },
        Reducer(r));
    return {r.val, r.loc};
  }

  std::pair<T, size_t> farthestFromPlane(const Plane<T> &P) {
    const auto points = m_points;
    const T nx = P.n.x, ny = P.n.y, nz = P.n.z, d0 = P.d;
    using Reducer = Kokkos::MaxLoc<T, size_t>;
    typename Reducer::value_type r;
    Kokkos::parallel_reduce(
        "sqh::farthest_plane", Kokkos::RangePolicy<ExecSpace>(0, m_numPoints),
        KOKKOS_LAMBDA(const size_t i, typename Reducer::value_type &upd) {
          const T d = Kokkos::abs(points(i, 0) * nx + points(i, 1) * ny +
                                  points(i, 2) * nz + d0);
          if (d > upd.val) {
            upd.val = d;
            upd.loc = i;
          }
        },
        Reducer(r));
    return {r.val, r.loc};
  }

  // Compact live conflict segments to the front of a fresh arena.
  void compactArena(size_t extraNeeded) {
    std::vector<Segment> live;
    std::vector<size_t> liveFace;
    size_t total = 0;
    for (size_t fi = 0; fi < m_mesh.faces.size(); fi++) {
      Face &f = m_mesh.faces[fi];
      if (!f.isDisabled() && f.conflictCount > 0) {
        live.push_back({(IndexT)f.conflictOffset, (IndexT)f.conflictCount});
        liveFace.push_back(fi);
        total += f.conflictCount;
      }
    }
    const size_t newCap =
        std::max(m_arena.extent(0), size_t(2) * (total + extraNeeded) + 64);
    Kokkos::View<IndexT *, MemSpace> fresh(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("sqh::arena")),
        newCap);

    const size_t k = live.size();
    ensurePair(m_dSegments, m_hSegments, k, "sqh::segs");
    ensurePair(m_dSegPrefix, m_hSegPrefix, k + 1, "sqh::segprefix");
    size_t acc = 0;
    for (size_t s = 0; s < k; s++) {
      m_hSegments(s) = live[s];
      m_hSegPrefix(s) = (IndexT)acc;
      acc += live[s].count;
    }
    m_hSegPrefix(k) = (IndexT)acc;
    Kokkos::deep_copy(m_dSegments, m_hSegments);
    Kokkos::deep_copy(m_dSegPrefix, m_hSegPrefix);

    const auto segs = m_dSegments;
    const auto prefix = m_dSegPrefix;
    const auto src = m_arena;
    const size_t kk = k;
    Kokkos::parallel_for(
        "sqh::compact", Kokkos::RangePolicy<ExecSpace>(0, total),
        KOKKOS_LAMBDA(const size_t w) {
          size_t lo = 0, hi = kk;
          while (hi - lo > 1) {
            const size_t mid = (lo + hi) / 2;
            if ((size_t)prefix(mid) <= w)
              lo = mid;
            else
              hi = mid;
          }
          const size_t local = w - prefix(lo);
          fresh(w) = src(segs(lo).start + local);
        });
    Kokkos::fence();

    acc = 0;
    for (size_t s = 0; s < k; s++) {
      m_mesh.faces[liveFace[s]].conflictOffset = acc;
      acc += live[s].count;
    }
    m_arena = fresh;
    m_arenaTail = total;
  }

  size_t arenaReserve(size_t n) {
    if (m_arenaTail + n > m_arena.extent(0))
      compactArena(n);
    const size_t at = m_arenaTail;
    m_arenaTail += n;
    return at;
  }

  // Conflict repartition (3-kernel scheme); classification uses the tangent
  // distance n.c + d + r.
  template <bool Identity>
  void partitionPoints(const std::vector<Segment> &srcSegments,
                       size_t skipIndex, const std::vector<Plane<T>> &planes,
                       const std::vector<size_t> &faceIds) {
    const size_t k = planes.size();
    size_t totalWork = 0;
    if (Identity) {
      totalWork = m_numPoints;
    } else {
      for (const auto &s : srcSegments)
        totalWork += s.count;
    }
    for (size_t j = 0; j < k; j++) {
      Face &f = m_mesh.faces[faceIds[j]];
      f.conflictOffset = 0;
      f.conflictCount = 0;
      f.mostDistantPointDist = 0;
      f.mostDistantPoint = 0;
    }
    if (totalWork == 0 || k == 0)
      return;

    const auto srcArena = m_arena;
    const size_t base = arenaReserve(totalWork);
    const auto writeArena = m_arena;

    constexpr bool hostAccessible =
        Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemSpace>::accessible;
    if constexpr (hostAccessible) {
     if (totalWork < m_hostPathThreshold) {
      Kokkos::fence();
      m_hostScratch.resize(k);
      for (auto &s : m_hostScratch)
        s.clear();
      std::vector<std::pair<T, IndexT>> best(k, {T(0), IndexT(0)});
      const T eps = m_epsilon;
      auto classify = [&](IndexT p) {
        if (!Identity && p == (IndexT)skipIndex)
          return;
        const T px = m_pointsHost(p, 0), py = m_pointsHost(p, 1),
                pz = m_pointsHost(p, 2), pr = m_radiiHost(p);
        for (size_t j = 0; j < k; j++) {
          const Plane<T> &pl = planes[j];
          const T D = pl.n.x * px + pl.n.y * py + pl.n.z * pz + pl.d + pr;
          if (D > eps) {
            m_hostScratch[j].push_back(p);
            if (D > best[j].first)
              best[j] = {D, p};
            break;
          }
        }
      };
      if (Identity) {
        for (size_t i = 0; i < m_numPoints; i++)
          classify((IndexT)i);
      } else {
        for (const Segment &s : srcSegments)
          for (IndexT i = 0; i < s.count; i++)
            classify(srcArena(s.start + i));
      }
      size_t acc = base;
      for (size_t j = 0; j < k; j++) {
        Face &f = m_mesh.faces[faceIds[j]];
        f.conflictCount = m_hostScratch[j].size();
        if (f.conflictCount > 0) {
          f.conflictOffset = acc;
          for (size_t i = 0; i < f.conflictCount; i++)
            writeArena(acc + i) = m_hostScratch[j][i];
          acc += f.conflictCount;
          f.mostDistantPointDist = best[j].first;
          f.mostDistantPoint = best[j].second;
        }
      }
      m_arenaTail = acc;
      return;
     }
    }

    ensurePair(m_dPlanes, m_hPlanes, k, "sqh::planes");
    for (size_t j = 0; j < k; j++)
      m_hPlanes(j) = {planes[j].n.x, planes[j].n.y, planes[j].n.z,
                      planes[j].d};
    Kokkos::deep_copy(
        Kokkos::subview(m_dPlanes, std::make_pair(size_t(0), k)),
        Kokkos::subview(m_hPlanes, std::make_pair(size_t(0), k)));

    const size_t ns = Identity ? 0 : srcSegments.size();
    if (!Identity) {
      ensurePair(m_dSegments, m_hSegments, ns, "sqh::segs");
      ensurePair(m_dSegPrefix, m_hSegPrefix, ns + 1, "sqh::segprefix");
      size_t acc = 0;
      for (size_t s = 0; s < ns; s++) {
        m_hSegments(s) = srcSegments[s];
        m_hSegPrefix(s) = (IndexT)acc;
        acc += srcSegments[s].count;
      }
      m_hSegPrefix(ns) = (IndexT)acc;
      Kokkos::deep_copy(
          Kokkos::subview(m_dSegments, std::make_pair(size_t(0), ns)),
          Kokkos::subview(m_hSegments, std::make_pair(size_t(0), ns)));
      Kokkos::deep_copy(
          Kokkos::subview(m_dSegPrefix, std::make_pair(size_t(0), ns + 1)),
          Kokkos::subview(m_hSegPrefix, std::make_pair(size_t(0), ns + 1)));
    }

    ensure(m_dAssign, totalWork, "sqh::assign");
    ensure(m_dPointIdx, totalWork, "sqh::pidx");
    ensurePair(m_dCounts, m_hCounts, k, "sqh::counts");
    ensure(m_dCursors, k, "sqh::cursors");
    ensurePair(m_dOffsets, m_hOffsets, k, "sqh::offsets");
    ensurePair(m_dMax, m_hMax, k, "sqh::max");

    Kokkos::deep_copy(
        Kokkos::subview(m_dCounts, std::make_pair(size_t(0), k)), IndexT(0));

    const auto points = m_points;
    const auto radii = m_radii;
    const auto arena = srcArena;
    const auto segs = m_dSegments;
    const auto prefix = m_dSegPrefix;
    const auto dPlanes = m_dPlanes;
    const auto dAssign = m_dAssign;
    const auto dPointIdx = m_dPointIdx;
    const auto dCounts = m_dCounts;
    const T eps = m_epsilon;
    const IndexT skip = (IndexT)skipIndex;
    const size_t kk = k;
    const size_t numSegs = ns;

    Kokkos::parallel_for(
        "sqh::classify", Kokkos::RangePolicy<ExecSpace>(0, totalWork),
        KOKKOS_LAMBDA(const size_t w) {
          IndexT p;
          if (Identity) {
            p = (IndexT)w;
          } else {
            size_t lo = 0, hi = numSegs;
            while (hi - lo > 1) {
              const size_t mid = (lo + hi) / 2;
              if ((size_t)prefix(mid) <= w)
                lo = mid;
              else
                hi = mid;
            }
            p = arena(segs(lo).start + (w - prefix(lo)));
          }
          std::int32_t assign = -1;
          if (Identity || p != skip) {
            const T px = points(p, 0), py = points(p, 1), pz = points(p, 2);
            const T pr = radii(p);
            for (size_t j = 0; j < kk; j++) {
              const DevPlane pl = dPlanes(j);
              const T D = pl.nx * px + pl.ny * py + pl.nz * pz + pl.d + pr;
              if (D > eps) {
                assign = (std::int32_t)j;
                break;
              }
            }
          }
          dAssign(w) = assign;
          dPointIdx(w) = p;
          if (assign >= 0)
            Kokkos::atomic_inc(&dCounts(assign));
        });

    Kokkos::deep_copy(Kokkos::subview(m_hCounts, std::make_pair(size_t(0), k)),
                      Kokkos::subview(m_dCounts, std::make_pair(size_t(0), k)));
    size_t needed = 0;
    for (size_t j = 0; j < k; j++)
      needed += m_hCounts(j);
    m_arenaTail = base + needed;
    if (needed > 0) {
      size_t acc = base;
      for (size_t j = 0; j < k; j++) {
        m_hOffsets(j) = (IndexT)acc;
        acc += m_hCounts(j);
      }
      Kokkos::deep_copy(
          Kokkos::subview(m_dOffsets, std::make_pair(size_t(0), k)),
          Kokkos::subview(m_hOffsets, std::make_pair(size_t(0), k)));
      Kokkos::deep_copy(
          Kokkos::subview(m_dCursors, std::make_pair(size_t(0), k)),
          IndexT(0));

      const auto arenaNow = writeArena;
      const auto dCursors = m_dCursors;
      const auto dOffsets = m_dOffsets;
      const auto dPointIdx2 = m_dPointIdx;
      Kokkos::parallel_for(
          "sqh::scatter", Kokkos::RangePolicy<ExecSpace>(0, totalWork),
          KOKKOS_LAMBDA(const size_t w) {
            const std::int32_t a = dAssign(w);
            if (a < 0)
              return;
            const IndexT slot = Kokkos::atomic_fetch_add(&dCursors(a), IndexT(1));
            arenaNow(dOffsets(a) + slot) = dPointIdx2(w);
          });

      const auto dMax = m_dMax;
      const auto dCountsC = m_dCounts;
      const auto pointsC = m_points;
      const auto radiiC = m_radii;
      const auto dPlanesC = m_dPlanes;
      using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
      using member_type = typename TeamPolicy::member_type;
      Kokkos::parallel_for(
          "sqh::facemax", TeamPolicy(k, Kokkos::AUTO),
          KOKKOS_LAMBDA(const member_type &team) {
            const int f = team.league_rank();
            const IndexT cnt = dCountsC(f);
            const IndexT off = dOffsets(f);
            using Reducer = Kokkos::MaxLoc<T, IndexT>;
            typename Reducer::value_type r;
            const DevPlane pl = dPlanesC(f);
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, cnt),
                [&](const IndexT i, typename Reducer::value_type &upd) {
                  const IndexT p = arenaNow(off + i);
                  const T D = pl.nx * pointsC(p, 0) + pl.ny * pointsC(p, 1) +
                              pl.nz * pointsC(p, 2) + pl.d + radiiC(p);
                  if (D > upd.val) {
                    upd.val = D;
                    upd.loc = p;
                  }
                },
                Reducer(r));
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
              dMax(f) = {cnt ? r.val : T(0), cnt ? r.loc : IndexT(0)};
            });
          });
      Kokkos::deep_copy(Kokkos::subview(m_hMax, std::make_pair(size_t(0), k)),
                        Kokkos::subview(m_dMax, std::make_pair(size_t(0), k)));
    }
    Kokkos::fence();

    for (size_t j = 0; j < k; j++) {
      Face &f = m_mesh.faces[faceIds[j]];
      f.conflictCount = m_hCounts(j);
      if (f.conflictCount > 0) {
        f.conflictOffset = m_hOffsets(j);
        f.mostDistantPointDist = m_hMax(j).dist;
        f.mostDistantPoint = m_hMax(j).idx;
      }
    }
  }

  // Initial tetrahedron: seeded and checked for degeneracy on the centers; the
  // four faces then get tritangent planes.
  void setupInitialTetrahedron() {
    const size_t vertexCount = m_numPoints;

    if (vertexCount <= 4) {
      size_t v[4] = {0, std::min<size_t>(1, vertexCount - 1),
                     std::min<size_t>(2, vertexCount - 1),
                     std::min<size_t>(3, vertexCount - 1)};
      const Vec3<T> N = triangleNormal(pt(v[0]), pt(v[1]), pt(v[2]));
      const Plane<T> trianglePlane(N, pt(v[0]));
      if (trianglePlane.isPointOnPositiveSide(pt(v[3])))
        std::swap(v[0], v[1]);
      m_mesh.setup(v[0], v[1], v[2], v[3]);
      computeInitialPlanesAndPartition();
      return;
    }

    T maxD = m_epsilonSquared;
    std::pair<size_t, size_t> selectedPoints{0, 0};
    for (size_t i = 0; i < 6; i++)
      for (size_t j = i + 1; j < 6; j++) {
        const T d =
            (pt(m_extremeValues[i]) - pt(m_extremeValues[j])).squaredNorm();
        if (d > maxD) {
          maxD = d;
          selectedPoints = {m_extremeValues[i], m_extremeValues[j]};
        }
      }
    if (maxD == m_epsilonSquared) {
      // All centers (nearly) coincide: the hull is dominated by the largest
      // sphere; emit the same degenerate micro-mesh the point port does, but
      // seed it with the largest-radius sphere so the support is right.
      size_t big = 0;
      for (size_t i = 1; i < vertexCount; i++)
        if (rad(i) > rad(big))
          big = i;
      m_mesh.setup(big, big, big, big);
      computeInitialPlanesAndPartition();
      return;
    }

    const Vec3<T> S = pt(selectedPoints.first);
    const Vec3<T> V = pt(selectedPoints.second) - pt(selectedPoints.first);
    auto [lineD, lineI] = farthestFromLine(S, V);
    if (lineD <= m_epsilonSquared) {
      // collinear centers: thin-triangle fallback (host scan, rare path)
      size_t i = 0;
      for (; i < vertexCount; i++) {
        const Vec3<T> ve = pt(i);
        if (!(ve == pt(selectedPoints.first)) &&
            !(ve == pt(selectedPoints.second)))
          break;
      }
      const size_t thirdPoint = (i == vertexCount) ? selectedPoints.first : i;
      for (i = 0; i < vertexCount; i++) {
        const Vec3<T> ve = pt(i);
        if (!(ve == pt(selectedPoints.first)) &&
            !(ve == pt(selectedPoints.second)) && !(ve == pt(thirdPoint)))
          break;
      }
      const size_t fourthPoint = (i == vertexCount) ? selectedPoints.first : i;
      m_mesh.setup(selectedPoints.first, selectedPoints.second, thirdPoint,
                   fourthPoint);
      computeInitialPlanesAndPartition();
      return;
    }

    std::array<size_t, 3> baseTriangle{selectedPoints.first,
                                       selectedPoints.second, lineI};
    const Vec3<T> btv[3] = {pt(baseTriangle[0]), pt(baseTriangle[1]),
                            pt(baseTriangle[2])};

    const Vec3<T> N = triangleNormal(btv[0], btv[1], btv[2]);
    Plane<T> trianglePlane(N, btv[0]);
    auto [planeD, planeI] = farthestFromPlane(trianglePlane);
    size_t maxI = planeI;
    if (planeD <= m_epsilon) {
      // Coplanar centers (the balls may still be 3D): synthetic radius-0
      // vertex off the plane, remapped at the end, as in the point port.
      m_planar = true;
      const Vec3<T> N1 = triangleNormal(btv[1], btv[2], btv[0]);
      const Vec3<T> extraPoint = N1 + pt(0);
      m_pointsHost(m_numPoints, 0) = extraPoint.x;
      m_pointsHost(m_numPoints, 1) = extraPoint.y;
      m_pointsHost(m_numPoints, 2) = extraPoint.z;
      m_radiiHost(m_numPoints) = T(0);
      maxI = m_numPoints;
      m_numPoints += 1;
      Kokkos::deep_copy(m_points, m_pointsHost);
      Kokkos::deep_copy(m_radii, m_radiiHost);
    }

    const Plane<T> triPlane(N, btv[0]);
    if (triPlane.isPointOnPositiveSide(pt(maxI)))
      std::swap(baseTriangle[0], baseTriangle[1]);

    m_mesh.setup(baseTriangle[0], baseTriangle[1], baseTriangle[2], maxI);
    computeInitialPlanesAndPartition();
  }

  void computeInitialPlanesAndPartition() {
    m_newPlanes.clear();
    std::vector<size_t> faceIds;
    for (size_t fi = 0; fi < m_mesh.faces.size(); fi++) {
      Face &f = m_mesh.faces[fi];
      const auto v = m_mesh.getVertexIndicesOfFace(f);
      const Vec3<T> refN = triangleNormal(pt(v[0]), pt(v[1]), pt(v[2]));
      f.P = tritangentPlane(v[0], v[1], v[2], refN);
      m_newPlanes.push_back(f.P);
      faceIds.push_back(fi);
    }
    partitionPoints<true>({}, m_numPoints /*no skip*/, m_newPlanes, faceIds);
  }

  bool reorderHorizonEdges(std::vector<size_t> &horizonEdges) {
    const size_t horizonEdgeCount = horizonEdges.size();
    for (size_t i = 0; i + 1 < horizonEdgeCount; i++) {
      const size_t endVertex = m_mesh.halfEdges[horizonEdges[i]].endVertex;
      bool foundNext = false;
      for (size_t j = i + 1; j < horizonEdgeCount; j++) {
        const size_t beginVertex =
            m_mesh.halfEdges[m_mesh.halfEdges[horizonEdges[j]].opp].endVertex;
        if (beginVertex == endVertex) {
          std::swap(horizonEdges[i + 1], horizonEdges[j]);
          foundNext = true;
          break;
        }
      }
      if (!foundNext)
        return false;
    }
    return true;
  }

  void removePointFromFaceConflicts(Face &f, size_t pointIndex) {
    const auto arena = m_arena;
    const size_t off = f.conflictOffset, cnt = f.conflictCount;
    const IndexT target = (IndexT)pointIndex;
    Kokkos::View<IndexT, MemSpace> dPos(std::string("sqh::pos"));
    Kokkos::deep_copy(dPos, IndexT(cnt));
    Kokkos::parallel_for(
        "sqh::findpt", Kokkos::RangePolicy<ExecSpace>(0, cnt),
        KOKKOS_LAMBDA(const size_t i) {
          if (arena(off + i) == target)
            dPos() = (IndexT)i;
        });
    IndexT pos;
    Kokkos::deep_copy(pos, dPos);
    if (pos < cnt) {
      Kokkos::parallel_for(
          "sqh::swappt", Kokkos::RangePolicy<ExecSpace>(0, 1),
          KOKKOS_LAMBDA(const size_t) { arena(off + pos) = arena(off + cnt - 1); });
      f.conflictCount -= 1;
    }
    Kokkos::fence();
  }

  // Main loop. The apex is the sphere with the greatest tangent distance.
  void createConvexHalfEdgeMesh() {
    m_visibleFaces.clear();
    m_horizonEdges.clear();
    m_possiblyVisibleFaces.clear();

    setupInitialTetrahedron();
    assert(m_mesh.faces.size() == 4);

    m_faceList.clear();
    for (size_t i = 0; i < 4; i++) {
      Face &f = m_mesh.faces[i];
      if (f.conflictCount > 0) {
        m_faceList.push_back(i);
        f.inFaceStack = 1;
      }
    }

    size_t iter = 0;
    while (!m_faceList.empty()) {
      iter++;
      if (iter == std::numeric_limits<size_t>::max())
        iter = 0;

      const size_t topFaceIndex = m_faceList.front();
      m_faceList.pop_front();

      Face &tf = m_mesh.faces[topFaceIndex];
      tf.inFaceStack = 0;
      if (tf.conflictCount == 0 || tf.isDisabled())
        continue;

      const size_t activePointIndex = tf.mostDistantPoint;
      const Vec3<T> activePoint = pt(activePointIndex);
      const T activeRadius = rad(activePointIndex);

      m_horizonEdges.clear();
      m_possiblyVisibleFaces.clear();
      m_visibleFaces.clear();
      m_possiblyVisibleFaces.push_back(
          {topFaceIndex, std::numeric_limits<size_t>::max()});
      while (!m_possiblyVisibleFaces.empty()) {
        const FaceData faceData = m_possiblyVisibleFaces.back();
        m_possiblyVisibleFaces.pop_back();
        Face &pvf = m_mesh.faces[faceData.faceIndex];
        assert(!pvf.isDisabled());

        if (pvf.visibilityCheckedOnIteration == iter) {
          if (pvf.isVisibleFaceOnCurrentIteration)
            continue;
        } else {
          pvf.visibilityCheckedOnIteration = iter;
          // tangent distance of the apex sphere above this face
          const T d = pvf.P.signedDistance(activePoint) + activeRadius;
          if (d > 0) {
            pvf.isVisibleFaceOnCurrentIteration = 1;
            pvf.horizonEdgesOnCurrentIteration = 0;
            m_visibleFaces.push_back(faceData.faceIndex);
            for (size_t heIndex : m_mesh.getHalfEdgeIndicesOfFace(pvf)) {
              if (m_mesh.halfEdges[heIndex].opp != faceData.enteredFromHalfEdge)
                m_possiblyVisibleFaces.push_back(
                    {m_mesh.halfEdges[m_mesh.halfEdges[heIndex].opp].face,
                     heIndex});
            }
            continue;
          }
          assert(faceData.faceIndex != topFaceIndex);
        }

        pvf.isVisibleFaceOnCurrentIteration = 0;
        m_horizonEdges.push_back(faceData.enteredFromHalfEdge);
        const auto halfEdges = m_mesh.getHalfEdgeIndicesOfFace(
            m_mesh.faces[m_mesh.halfEdges[faceData.enteredFromHalfEdge].face]);
        const std::int8_t ind =
            (halfEdges[0] == faceData.enteredFromHalfEdge)
                ? 0
                : (halfEdges[1] == faceData.enteredFromHalfEdge ? 1 : 2);
        m_mesh.faces[m_mesh.halfEdges[faceData.enteredFromHalfEdge].face]
            .horizonEdgesOnCurrentIteration |= (1 << ind);
      }
      const size_t horizonEdgeCount = m_horizonEdges.size();

      if (!reorderHorizonEdges(m_horizonEdges)) {
        m_diagnostics.m_failedHorizonEdges++;
        removePointFromFaceConflicts(tf, activePointIndex);
        continue;
      }

      m_newFaceIndices.clear();
      m_newHalfEdgeIndices.clear();
      m_srcSegments.clear();
      size_t disableCounter = 0;
      for (size_t faceIndex : m_visibleFaces) {
        Face &disabledFace = m_mesh.faces[faceIndex];
        auto halfEdges = m_mesh.getHalfEdgeIndicesOfFace(disabledFace);
        for (size_t j = 0; j < 3; j++) {
          if ((disabledFace.horizonEdgesOnCurrentIteration & (1 << j)) == 0) {
            if (disableCounter < horizonEdgeCount * 2) {
              m_newHalfEdgeIndices.push_back(halfEdges[j]);
              disableCounter++;
            } else {
              m_mesh.disableHalfEdge(halfEdges[j]);
            }
          }
        }
        if (disabledFace.conflictCount > 0) {
          m_srcSegments.push_back({(IndexT)disabledFace.conflictOffset,
                                   (IndexT)disabledFace.conflictCount});
          disabledFace.conflictCount = 0;
        }
        m_mesh.disableFace(faceIndex);
      }
      if (disableCounter < horizonEdgeCount * 2) {
        const size_t needed = horizonEdgeCount * 2 - disableCounter;
        for (size_t i = 0; i < needed; i++)
          m_newHalfEdgeIndices.push_back(m_mesh.addHalfEdge());
      }

      m_newPlanes.clear();
      for (size_t i = 0; i < horizonEdgeCount; i++) {
        const size_t AB = m_horizonEdges[i];
        const auto hev =
            m_mesh.getVertexIndicesOfHalfEdge(m_mesh.halfEdges[AB]);
        const size_t A = hev[0];

        const size_t newFaceIndex = m_mesh.addFace();
        m_newFaceIndices.push_back(newFaceIndex);

        const size_t CA = m_newHalfEdgeIndices[2 * i + 0];
        const size_t BC = m_newHalfEdgeIndices[2 * i + 1];

        m_mesh.halfEdges[AB].next = BC;
        m_mesh.halfEdges[BC].next = CA;
        m_mesh.halfEdges[CA].next = AB;
        m_mesh.halfEdges[BC].face = newFaceIndex;
        m_mesh.halfEdges[CA].face = newFaceIndex;
        m_mesh.halfEdges[AB].face = newFaceIndex;
        m_mesh.halfEdges[CA].endVertex = A;
        m_mesh.halfEdges[BC].endVertex = activePointIndex;

        Face &newFace = m_mesh.faces[newFaceIndex];
        // reference normal: unweighted triangle normal in face winding (the
        // exact answer for equal radii; the root selector for unequal radii)
        const Vec3<T> refN = triangleNormal(pt(A), pt(hev[1]), activePoint);
        newFace.P = tritangentPlane(A, hev[1], activePointIndex, refN);
        newFace.he = AB;
        m_newPlanes.push_back(newFace.P);

        m_mesh.halfEdges[CA].opp =
            m_newHalfEdgeIndices[i > 0 ? i * 2 - 1 : 2 * horizonEdgeCount - 1];
        m_mesh.halfEdges[BC].opp =
            m_newHalfEdgeIndices[((i + 1) * 2) % (horizonEdgeCount * 2)];
      }

      partitionPoints<false>(m_srcSegments, activePointIndex, m_newPlanes,
                             m_newFaceIndices);

      for (size_t newFaceIndex : m_newFaceIndices) {
        Face &newFace = m_mesh.faces[newFaceIndex];
        if (newFace.conflictCount > 0 && !newFace.inFaceStack) {
          m_faceList.push_back(newFaceIndex);
          newFace.inFaceStack = 1;
        }
      }
    }
  }

  void buildMesh(const T *centers, const T *radii, size_t sphereCount,
                 T epsilon) {
    m_mesh = Mesh();
    m_planar = false;
    m_scale = 0;
    m_diagnostics = SphereDiagnostics();
    m_arenaTail = 0;
    if (sphereCount == 0) {
      m_numPoints = 0;
      return;
    }
    m_numPoints = sphereCount;

    // +1 spare slot for the coplanar-centers synthetic vertex
    m_points = PointsView(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("sqh::points")),
        sphereCount + 1);
    m_pointsHost = Kokkos::create_mirror_view(m_points);
    m_radii = RadiiView(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("sqh::radii")),
        sphereCount + 1);
    m_radiiHost = Kokkos::create_mirror_view(m_radii);
    for (size_t i = 0; i < sphereCount; i++) {
      m_pointsHost(i, 0) = centers[3 * i + 0];
      m_pointsHost(i, 1) = centers[3 * i + 1];
      m_pointsHost(i, 2) = centers[3 * i + 2];
      m_radiiHost(i) = radii[i];
    }
    Kokkos::deep_copy(m_points, m_pointsHost);
    Kokkos::deep_copy(m_radii, m_radiiHost);

    if (m_arena.extent(0) < 2 * sphereCount + 64)
      m_arena = Kokkos::View<IndexT *, MemSpace>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("sqh::arena")),
          2 * sphereCount + 64);

    getExtremeValues();
    m_epsilon = epsilon * m_scale;
    m_epsilonSquared = m_epsilon * m_epsilon;

    createConvexHalfEdgeMesh();

    if (m_planar) {
      const size_t extraPointIndex = m_numPoints - 1;
      for (auto &he : m_mesh.halfEdges)
        if (he.endVertex == extraPointIndex)
          he.endVertex = 0;
      m_numPoints -= 1;
    }
  }

  SphereHull<T> extractHull(bool CCW) {
    SphereHull<T> hull;
    std::vector<bool> faceProcessed(m_mesh.faces.size(), false);
    std::vector<size_t> faceStack;
    for (size_t i = 0; i < m_mesh.faces.size(); i++)
      if (!m_mesh.faces[i].isDisabled()) {
        faceStack.push_back(i);
        break;
      }
    if (faceStack.empty())
      return hull;

    const size_t iCCW = CCW ? 1 : 0;
    const size_t finalFaceCount =
        m_mesh.faces.size() - m_mesh.disabledFaces.size();
    hull.m_indices.reserve(finalFaceCount * 3);
    hull.m_facePlanes.reserve(finalFaceCount);

    while (!faceStack.empty()) {
      const size_t top = faceStack.back();
      faceStack.pop_back();
      assert(!m_mesh.faces[top].isDisabled());
      if (faceProcessed[top])
        continue;
      faceProcessed[top] = true;
      const auto halfEdges = m_mesh.getHalfEdgeIndicesOfFace(m_mesh.faces[top]);
      const size_t adjacent[] = {
          m_mesh.halfEdges[m_mesh.halfEdges[halfEdges[0]].opp].face,
          m_mesh.halfEdges[m_mesh.halfEdges[halfEdges[1]].opp].face,
          m_mesh.halfEdges[m_mesh.halfEdges[halfEdges[2]].opp].face};
      for (size_t a : adjacent)
        if (!faceProcessed[a] && !m_mesh.faces[a].isDisabled())
          faceStack.push_back(a);
      const auto v = m_mesh.getVertexIndicesOfFace(m_mesh.faces[top]);
      hull.m_indices.push_back(v[0]);
      hull.m_indices.push_back(v[1 + iCCW]);
      hull.m_indices.push_back(v[2 - iCCW]);
      hull.m_facePlanes.push_back(m_mesh.faces[top].P);
    }

    std::vector<long long> remap(m_numPoints + 1, -1);
    hull.m_reducedIndices.reserve(hull.m_indices.size());
    for (size_t idx : hull.m_indices) {
      if (remap[idx] < 0) {
        remap[idx] = (long long)hull.m_hullSphereSource.size();
        hull.m_hullSphereSource.push_back(idx);
      }
      hull.m_reducedIndices.push_back((size_t)remap[idx]);
    }
    return hull;
  }
};

} // namespace kokkos_quickhull
