#pragma once
/*
 * Kokkos port of the 3D QuickHull implementation by Antti Kuukka
 * (Eigen adaptation: peterspackman/quickhull). Public Domain, as the original.
 *
 * O(N) point work (extreme scans, farthest-point searches, conflict-point
 * partitioning) runs as Kokkos kernels; O(H) topology work (half-edge
 * bookkeeping, horizon, face creation) stays serial on the host.
 *
 * Per-face conflict lists are segments [offset, count) in one flat device
 * index arena, compacted only when the tail runs out of space. Points are
 * View<T*[3], LayoutLeft> so each axis is contiguous across points; a host
 * mirror of the coordinates serves the topology code.
 */

#include <Kokkos_Core.hpp>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <vector>

namespace kokkos_quickhull {

struct DiagnosticsData {
  size_t m_failedHorizonEdges = 0;
};

template <typename FloatType> constexpr FloatType defaultEps() {
  return std::is_same<FloatType, float>::value ? FloatType(0.0001)
                                               : FloatType(0.0000001);
}

// ---------------------------------------------------------------------------
// Small host-side math helpers (replaces Eigen usage of the original)
// ---------------------------------------------------------------------------
template <typename T> struct Vec3 {
  T x, y, z;
  KOKKOS_INLINE_FUNCTION Vec3() : x(0), y(0), z(0) {}
  KOKKOS_INLINE_FUNCTION Vec3(T a, T b, T c) : x(a), y(b), z(c) {}
  KOKKOS_INLINE_FUNCTION Vec3 operator-(const Vec3 &o) const {
    return {x - o.x, y - o.y, z - o.z};
  }
  KOKKOS_INLINE_FUNCTION Vec3 operator+(const Vec3 &o) const {
    return {x + o.x, y + o.y, z + o.z};
  }
  KOKKOS_INLINE_FUNCTION T dot(const Vec3 &o) const {
    return x * o.x + y * o.y + z * o.z;
  }
  KOKKOS_INLINE_FUNCTION Vec3 cross(const Vec3 &o) const {
    return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
  }
  KOKKOS_INLINE_FUNCTION T squaredNorm() const { return dot(*this); }
  bool operator==(const Vec3 &o) const {
    return x == o.x && y == o.y && z == o.z;
  }
};

template <typename T>
KOKKOS_INLINE_FUNCTION Vec3<T> triangleNormal(const Vec3<T> &a,
                                              const Vec3<T> &b,
                                              const Vec3<T> &c) {
  return (a - c).cross(b - c);
}

// Plane: n.dot(p) + d, plus |n|^2 cached for the epsilon test.
template <typename T> struct Plane {
  Vec3<T> n;
  T d = 0;
  T sqrNLength = 0;
  Plane() = default;
  Plane(const Vec3<T> &N, const Vec3<T> &P)
      : n(N), d(-N.dot(P)), sqrNLength(N.dot(N)) {}
  KOKKOS_INLINE_FUNCTION T signedDistance(const Vec3<T> &q) const {
    return n.dot(q) + d;
  }
  bool isPointOnPositiveSide(const Vec3<T> &q) const {
    return signedDistance(q) >= 0;
  }
};

// ---------------------------------------------------------------------------
// Host-side half-edge mesh (ported from mesh.h, Eigen removed,
// per-face conflict lists replaced by arena segments)
// ---------------------------------------------------------------------------
template <typename T, typename IndexT> struct MeshBuilder {
  static constexpr size_t kInvalid = std::numeric_limits<size_t>::max();

  struct HalfEdge {
    size_t endVertex = kInvalid;
    size_t opp = kInvalid;
    size_t face = kInvalid;
    size_t next = kInvalid;
    void disable() { endVertex = kInvalid; }
    bool isDisabled() const { return endVertex == kInvalid; }
  };

  struct Face {
    size_t he = kInvalid;
    Plane<T> P{};
    T mostDistantPointDist = 0;
    size_t mostDistantPoint = 0;
    size_t visibilityCheckedOnIteration = 0;
    std::uint8_t isVisibleFaceOnCurrentIteration : 1;
    std::uint8_t inFaceStack : 1;
    std::uint8_t horizonEdgesOnCurrentIteration : 3;
    // Conflict list: segment in the device index arena. count==0 <=> none.
    size_t conflictOffset = 0;
    size_t conflictCount = 0;

    Face()
        : isVisibleFaceOnCurrentIteration(0), inFaceStack(0),
          horizonEdgesOnCurrentIteration(0) {}
    void disable() { he = kInvalid; }
    bool isDisabled() const { return he == kInvalid; }
  };

  std::vector<Face> faces;
  std::vector<HalfEdge> halfEdges;
  std::vector<size_t> disabledFaces, disabledHalfEdges;

  size_t addFace() {
    if (!disabledFaces.empty()) {
      size_t index = disabledFaces.back();
      Face &f = faces[index];
      assert(f.isDisabled());
      assert(f.conflictCount == 0);
      f.mostDistantPointDist = 0;
      disabledFaces.pop_back();
      return index;
    }
    faces.emplace_back();
    return faces.size() - 1;
  }

  size_t addHalfEdge() {
    if (!disabledHalfEdges.empty()) {
      const size_t index = disabledHalfEdges.back();
      disabledHalfEdges.pop_back();
      return index;
    }
    halfEdges.emplace_back();
    return halfEdges.size() - 1;
  }

  // Disable a face; caller is responsible for harvesting its conflict segment.
  void disableFace(size_t faceIndex) {
    faces[faceIndex].disable();
    disabledFaces.push_back(faceIndex);
  }

  void disableHalfEdge(size_t heIndex) {
    halfEdges[heIndex].disable();
    disabledHalfEdges.push_back(heIndex);
  }

  // Initial tetrahedron ABCD (verbatim topology from the original).
  void setup(size_t a, size_t b, size_t c, size_t d) {
    faces.clear();
    halfEdges.clear();
    disabledFaces.clear();
    disabledHalfEdges.clear();
    faces.reserve(4);
    halfEdges.reserve(12);

    auto he = [&](size_t end, size_t opp, size_t face, size_t next) {
      HalfEdge e;
      e.endVertex = end;
      e.opp = opp;
      e.face = face;
      e.next = next;
      halfEdges.push_back(e);
    };
    he(b, 6, 0, 1);   // AB
    he(c, 9, 0, 2);   // BC
    he(a, 3, 0, 0);   // CA
    he(c, 2, 1, 4);   // AC
    he(d, 11, 1, 5);  // CD
    he(a, 7, 1, 3);   // DA
    he(a, 0, 2, 7);   // BA
    he(d, 5, 2, 8);   // AD
    he(b, 10, 2, 6);  // DB
    he(b, 1, 3, 10);  // CB
    he(d, 8, 3, 11);  // BD
    he(c, 4, 3, 9);   // DC

    for (size_t i = 0; i < 4; i++) {
      Face f;
      f.he = 3 * i;
      faces.push_back(f);
    }
  }

  std::array<size_t, 3> getVertexIndicesOfFace(const Face &f) const {
    std::array<size_t, 3> v;
    const HalfEdge *e = &halfEdges[f.he];
    v[0] = e->endVertex;
    e = &halfEdges[e->next];
    v[1] = e->endVertex;
    e = &halfEdges[e->next];
    v[2] = e->endVertex;
    return v;
  }

  std::array<size_t, 2> getVertexIndicesOfHalfEdge(const HalfEdge &e) const {
    return {halfEdges[e.opp].endVertex, e.endVertex};
  }

  std::array<size_t, 3> getHalfEdgeIndicesOfFace(const Face &f) const {
    return {f.he, halfEdges[f.he].next, halfEdges[halfEdges[f.he].next].next};
  }
};

// ---------------------------------------------------------------------------
// Result container (host). Indices reference the *original* point cloud.
// ---------------------------------------------------------------------------
template <typename T> class ConvexHull {
public:
  std::vector<size_t> m_indices;          // triangle list, 3 per face
  std::vector<Vec3<T>> m_hullVertices;    // unique hull vertices (reduced)
  std::vector<size_t> m_hullVertexSource; // original index of each hull vertex
  std::vector<size_t> m_reducedIndices;   // triangles re-indexed into above

  const std::vector<size_t> &indices() const { return m_indices; }
  const std::vector<Vec3<T>> &vertices() const { return m_hullVertices; }
  const std::vector<size_t> &reducedIndices() const { return m_reducedIndices; }
  const std::vector<size_t> &vertexSourceIndices() const {
    return m_hullVertexSource;
  }
};

} // namespace kokkos_quickhull

namespace kokkos_quickhull {

// ===========================================================================
// KokkosQuickHull
//   T        : float or double
//   IndexT   : index type stored in the device arena (uint32_t default;
//              use uint64_t / size_t for clouds beyond 2^32 points)
//   ExecSpace: Kokkos execution space for the point kernels
// ===========================================================================
template <typename T, typename IndexT = std::uint32_t,
          typename ExecSpace = Kokkos::DefaultExecutionSpace>
class KokkosQuickHull {
public:
  using MemSpace = typename ExecSpace::memory_space;
  using PointsView = Kokkos::View<T *[3], Kokkos::LayoutLeft, MemSpace>;
  using PointsHost = typename PointsView::host_mirror_type;
  using Mesh = MeshBuilder<T, IndexT>;
  using Face = typename Mesh::Face;

public:
  // Public because nvcc does not allow private nested types in __global__
  // instantiations.
  struct DevPlane {
    T nx, ny, nz, d, sqrN;
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
  // --- persistent state -----------------------------------------------------
  PointsView m_points;      // device coordinates (capacity may be N+1: planar)
  PointsHost m_pointsHost;  // read-only mirror for O(H) topology code
  size_t m_numPoints = 0;

  // conflict index arena
  Kokkos::View<IndexT *, MemSpace> m_arena;
  size_t m_arenaTail = 0;
  size_t m_liveCount = 0; // total points currently in live segments

  // scratch (grown on demand, never shrunk during a run)
  Kokkos::View<Segment *, MemSpace> m_dSegments;
  typename Kokkos::View<Segment *, MemSpace>::host_mirror_type m_hSegments;
  Kokkos::View<IndexT *, MemSpace> m_dSegPrefix; // exclusive prefix, size k+1
  typename Kokkos::View<IndexT *, MemSpace>::host_mirror_type m_hSegPrefix;
  Kokkos::View<DevPlane *, MemSpace> m_dPlanes;
  typename Kokkos::View<DevPlane *, MemSpace>::host_mirror_type m_hPlanes;
  Kokkos::View<std::int32_t *, MemSpace> m_dAssign;  // per work item
  Kokkos::View<IndexT *, MemSpace> m_dPointIdx;       // resolved point per work item
  Kokkos::View<IndexT *, MemSpace> m_dCounts;  // per new face
  typename Kokkos::View<IndexT *, MemSpace>::host_mirror_type m_hCounts;
  Kokkos::View<IndexT *, MemSpace> m_dCursors;
  Kokkos::View<IndexT *, MemSpace> m_dOffsets; // arena offsets per new face
  typename Kokkos::View<IndexT *, MemSpace>::host_mirror_type m_hOffsets;
  Kokkos::View<MaxRes *, MemSpace> m_dMax;
  typename Kokkos::View<MaxRes *, MemSpace>::host_mirror_type m_hMax;

  Mesh m_mesh;
  T m_epsilon = 0, m_epsilonSquared = 0, m_scale = 0;
  bool m_planar = false;
  std::array<size_t, 6> m_extremeValues{};
  DiagnosticsData m_diagnostics;

  // iteration scratch (host)
  std::vector<size_t> m_newFaceIndices, m_newHalfEdgeIndices, m_visibleFaces,
      m_horizonEdges;
  struct FaceData {
    size_t faceIndex;
    size_t enteredFromHalfEdge;
  };
  std::vector<FaceData> m_possiblyVisibleFaces;
  std::deque<size_t> m_faceList;
  std::vector<Segment> m_srcSegments;   // disabled faces' conflict segments
  std::vector<std::vector<IndexT>> m_hostScratch; // small-work fast path
  std::vector<Plane<T>> m_newPlanes;    // planes of new faces this iteration

  size_t m_hostPathThreshold = 4096;

public:
  const DiagnosticsData &getDiagnostics() const { return m_diagnostics; }
  // For host-accessible memory spaces, conflict repartitions smaller than
  // this run as a serial host loop instead of device kernels. Set to 0 to
  // force kernels always.
  void setHostPathThreshold(size_t t) { m_hostPathThreshold = t; }

  // Compute the hull. vertexData layout: x0,y0,z0,x1,y1,z1,...
  ConvexHull<T> getConvexHull(const T *vertexData, size_t vertexCount,
                              bool CCW, T eps = defaultEps<T>()) {
    buildMesh(vertexData, vertexCount, eps);
    return extractHull(CCW, vertexCount);
  }

  // Public because nvcc forbids KOKKOS_LAMBDA in private member functions.
  // Treat everything from here on as implementation detail.
public:
  Vec3<T> pt(size_t i) const {
    return {m_pointsHost(i, 0), m_pointsHost(i, 1), m_pointsHost(i, 2)};
  }

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

  // -------------------------------------------------------------------------
  // O(N) kernels
  // -------------------------------------------------------------------------
  void getExtremeValues() {
    const auto points = m_points;
    const size_t n = m_numPoints;
    for (int axis = 0; axis < 3; axis++) {
      using Reducer = Kokkos::MinMaxLoc<T, size_t>;
      typename Reducer::value_type r;
      Kokkos::parallel_reduce(
          "qh::extremes", Kokkos::RangePolicy<ExecSpace>(0, n),
          KOKKOS_LAMBDA(const size_t i, typename Reducer::value_type &upd) {
            const T v = points(i, axis);
            if (v > upd.max_val) {
              upd.max_val = v;
              upd.max_loc = i;
            }
            if (v < upd.min_val) {
              upd.min_val = v;
              upd.min_loc = i;
            }
          },
          Reducer(r));
      m_extremeValues[2 * axis] = r.max_loc;
      m_extremeValues[2 * axis + 1] = r.min_loc;
      // scale = max |extreme coordinate| (the original intends this; its
      // Eigen adaptation indexes data()[i] instead of data()[3*i] there)
      m_scale = std::max(m_scale, std::max(std::abs(r.max_val),
                                           std::abs(r.min_val)));
    }
  }

  // farthest point (squared distance) from the ray S + t*V
  std::pair<T, size_t> farthestFromLine(const Vec3<T> &S, const Vec3<T> &V) {
    const auto points = m_points;
    const T invLen2 = T(1) / V.squaredNorm();
    const T sx = S.x, sy = S.y, sz = S.z, vx = V.x, vy = V.y, vz = V.z;
    using Reducer = Kokkos::MaxLoc<T, size_t>;
    typename Reducer::value_type r;
    Kokkos::parallel_reduce(
        "qh::farthest_line", Kokkos::RangePolicy<ExecSpace>(0, m_numPoints),
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

  // farthest point (|signed distance|) from a plane
  std::pair<T, size_t> farthestFromPlane(const Plane<T> &P) {
    const auto points = m_points;
    const T nx = P.n.x, ny = P.n.y, nz = P.n.z, d0 = P.d;
    using Reducer = Kokkos::MaxLoc<T, size_t>;
    typename Reducer::value_type r;
    Kokkos::parallel_reduce(
        "qh::farthest_plane", Kokkos::RangePolicy<ExecSpace>(0, m_numPoints),
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
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "qh::arena"), newCap);

    const size_t k = live.size();
    ensurePair(m_dSegments, m_hSegments, k, "qh::segs");
    ensurePair(m_dSegPrefix, m_hSegPrefix, k + 1, "qh::segprefix");
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
    Kokkos::parallel_for(
        "qh::compact", Kokkos::RangePolicy<ExecSpace>(0, total),
        KOKKOS_LAMBDA(const size_t w) {
          // binary search: which segment does work item w belong to
          size_t lo = 0, hi = k;
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

    // rewrite face offsets
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

  // -------------------------------------------------------------------------
  // The workhorse: classify source points against k new planes, append the
  // per-face conflict segments to the arena, and find per-face farthest
  // points. Source points are either the whole cloud (Identity=true, used
  // once after the initial tetrahedron) or the union of the given arena
  // segments (Identity=false), skipping `skipIndex` (the apex).
  // Writes conflictOffset/Count/mostDistant* of the k faces in `faceIds`.
  // -------------------------------------------------------------------------
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
    for (size_t j = 0; j < k; j++) { // default: empty conflict lists
      Face &f = m_mesh.faces[faceIds[j]];
      f.conflictOffset = 0;
      f.conflictCount = 0;
      f.mostDistantPointDist = 0;
      f.mostDistantPoint = 0;
    }
    if (totalWork == 0 || k == 0)
      return;

    // Keep a reference to the current arena (source segments live there), then
    // pre-reserve worst-case output space so any compaction happens now.
    const auto srcArena = m_arena;
    const size_t base = arenaReserve(totalWork);
    const auto writeArena = m_arena;

    // Fast path: for host-accessible memory and tiny work, a serial loop beats
    // the kernel launches; this covers the long tail of iterations.
    constexpr bool hostAccessible =
        Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemSpace>::accessible;
    if constexpr (hostAccessible) {
     if (totalWork < m_hostPathThreshold) {
      Kokkos::fence(); // ensure prior device writes to the arena are visible
      m_hostScratch.resize(k);
      for (auto &s : m_hostScratch)
        s.clear();
      std::vector<std::pair<T, IndexT>> best(k, {T(0), IndexT(0)});
      auto classify = [&](IndexT p) {
        if (!Identity && p == (IndexT)skipIndex)
          return;
        const T px = m_pointsHost(p, 0), py = m_pointsHost(p, 1),
                pz = m_pointsHost(p, 2);
        for (size_t j = 0; j < k; j++) {
          const Plane<T> &pl = planes[j];
          const T D = pl.n.x * px + pl.n.y * py + pl.n.z * pz + pl.d;
          if (D > 0 && D * D > m_epsilonSquared * pl.sqrNLength) {
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
      m_arenaTail = acc; // roll back unused reservation
      return;
     }
    }

    // upload planes
    ensurePair(m_dPlanes, m_hPlanes, k, "qh::planes");
    for (size_t j = 0; j < k; j++)
      m_hPlanes(j) = {planes[j].n.x, planes[j].n.y, planes[j].n.z, planes[j].d,
                      planes[j].sqrNLength};
    Kokkos::deep_copy(
        Kokkos::subview(m_dPlanes, std::make_pair(size_t(0), k)),
        Kokkos::subview(m_hPlanes, std::make_pair(size_t(0), k)));

    // upload segment table
    const size_t ns = Identity ? 0 : srcSegments.size();
    if (!Identity) {
      ensurePair(m_dSegments, m_hSegments, ns, "qh::segs");
      ensurePair(m_dSegPrefix, m_hSegPrefix, ns + 1, "qh::segprefix");
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

    ensure(m_dAssign, totalWork, "qh::assign");
    ensure(m_dPointIdx, totalWork, "qh::pidx");
    ensurePair(m_dCounts, m_hCounts, k, "qh::counts");
    ensure(m_dCursors, k, "qh::cursors");
    ensurePair(m_dOffsets, m_hOffsets, k, "qh::offsets");
    ensurePair(m_dMax, m_hMax, k, "qh::max");

    Kokkos::deep_copy(
        Kokkos::subview(m_dCounts, std::make_pair(size_t(0), k)), IndexT(0));

    const auto points = m_points;
    const auto arena = srcArena;
    const auto segs = m_dSegments;
    const auto prefix = m_dSegPrefix;
    const auto dPlanes = m_dPlanes;
    const auto dAssign = m_dAssign;
    const auto dPointIdx = m_dPointIdx;
    const auto dCounts = m_dCounts;
    const T eps2 = m_epsilonSquared;
    const IndexT skip = (IndexT)skipIndex;
    const size_t kk = k;
    const size_t numSegs = ns;

    // Pass 1: classify (first plane with positive eps-distance wins, exactly
    // the original addPointToFace loop ordering).
    Kokkos::parallel_for(
        "qh::classify", Kokkos::RangePolicy<ExecSpace>(0, totalWork),
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
            for (size_t j = 0; j < kk; j++) {
              const DevPlane pl = dPlanes(j);
              const T D = pl.nx * px + pl.ny * py + pl.nz * pz + pl.d;
              if (D > 0 && D * D > eps2 * pl.sqrN) {
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

    // counts -> host, allocate arena segments
    Kokkos::deep_copy(Kokkos::subview(m_hCounts, std::make_pair(size_t(0), k)),
                      Kokkos::subview(m_dCounts, std::make_pair(size_t(0), k)));
    size_t needed = 0;
    for (size_t j = 0; j < k; j++)
      needed += m_hCounts(j);
    // Roll back the unused part of the pre-reserved block.
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

      // Pass 2: scatter point indices into the new segments.
      const auto arenaNow = writeArena;
      const auto dCursors = m_dCursors;
      const auto dOffsets = m_dOffsets;
      const auto dPointIdx2 = m_dPointIdx;
      Kokkos::parallel_for(
          "qh::scatter", Kokkos::RangePolicy<ExecSpace>(0, totalWork),
          KOKKOS_LAMBDA(const size_t w) {
            const std::int32_t a = dAssign(w);
            if (a < 0)
              return;
            const IndexT slot = Kokkos::atomic_fetch_add(&dCursors(a), IndexT(1));
            arenaNow(dOffsets(a) + slot) = dPointIdx2(w);
          });

      // Pass 3: segmented MaxLoc per new face (most distant conflict point)
      const auto dMax = m_dMax;
      const auto dCountsC = m_dCounts;
      const auto pointsC = m_points;
      const auto dPlanesC = m_dPlanes;
      using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
      using member_type = typename TeamPolicy::member_type;
      Kokkos::parallel_for(
          "qh::facemax", TeamPolicy(k, Kokkos::AUTO),
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
                              pl.nz * pointsC(p, 2) + pl.d;
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

    // write back to host faces
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

  // -------------------------------------------------------------------------
  // Initial tetrahedron (ported from the original; the two O(N) searches are
  // the kernels above, all small-data decisions use the host mirror)
  // -------------------------------------------------------------------------
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

    // Two most distant extreme points
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
      // Degenerate: all points coincide
      m_mesh.setup(0, std::min<size_t>(1, vertexCount - 1),
                   std::min<size_t>(2, vertexCount - 1),
                   std::min<size_t>(3, vertexCount - 1));
      computeInitialPlanesAndPartition();
      return;
    }

    // Most distant point from the line through the two chosen points (kernel)
    const Vec3<T> S = pt(selectedPoints.first);
    const Vec3<T> V = pt(selectedPoints.second) - pt(selectedPoints.first);
    auto [lineD, lineI] = farthestFromLine(S, V);
    if (lineD <= m_epsilonSquared) {
      // 1D subspace: thin triangle fallback (host scan, rare path)
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

    // 4th vertex: farthest from the base triangle plane (kernel)
    const Vec3<T> N = triangleNormal(btv[0], btv[1], btv[2]);
    Plane<T> trianglePlane(N, btv[0]);
    auto [planeD, planeI] = farthestFromPlane(trianglePlane);
    size_t maxI = planeI;
    if (planeD <= m_epsilon) {
      // Planar cloud: append one synthetic point off the plane, remap later
      m_planar = true;
      const Vec3<T> N1 = triangleNormal(btv[1], btv[2], btv[0]);
      const Vec3<T> extraPoint = N1 + pt(0);
      // points view was allocated with one spare row for exactly this case
      m_pointsHost(m_numPoints, 0) = extraPoint.x;
      m_pointsHost(m_numPoints, 1) = extraPoint.y;
      m_pointsHost(m_numPoints, 2) = extraPoint.z;
      maxI = m_numPoints;
      m_numPoints += 1;
      Kokkos::deep_copy(m_points, m_pointsHost);
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
      const Vec3<T> N1 = triangleNormal(pt(v[0]), pt(v[1]), pt(v[2]));
      f.P = Plane<T>(N1, pt(v[0]));
      m_newPlanes.push_back(f.P);
      faceIds.push_back(fi);
    }
    // Assign every point to (at most) the first face it is outside of
    partitionPoints<true>({}, m_numPoints /*no skip*/, m_newPlanes, faceIds);
  }

  // -------------------------------------------------------------------------
  // Horizon edge reordering (verbatim port)
  // -------------------------------------------------------------------------
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

  // Remove the apex point from a face's conflict segment (rare failure path).
  void removePointFromFaceConflicts(Face &f, size_t pointIndex) {
    const auto arena = m_arena;
    const size_t off = f.conflictOffset, cnt = f.conflictCount;
    const IndexT target = (IndexT)pointIndex;
    Kokkos::View<IndexT, MemSpace> dPos("qh::pos");
    Kokkos::deep_copy(dPos, IndexT(cnt));
    Kokkos::parallel_for(
        "qh::findpt", Kokkos::RangePolicy<ExecSpace>(0, cnt),
        KOKKOS_LAMBDA(const size_t i) {
          if (arena(off + i) == target)
            dPos() = (IndexT)i;
        });
    IndexT pos;
    Kokkos::deep_copy(pos, dPos);
    if (pos < cnt) {
      Kokkos::parallel_for(
          "qh::swappt", Kokkos::RangePolicy<ExecSpace>(0, 1),
          KOKKOS_LAMBDA(const size_t) { arena(off + pos) = arena(off + cnt - 1); });
      f.conflictCount -= 1;
    }
    Kokkos::fence();
  }

  // -------------------------------------------------------------------------
  // Main loop (host topology + device partitioning)
  // -------------------------------------------------------------------------
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

      // Visibility DFS over face adjacency, collecting the horizon edges
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
          const T d = pvf.P.signedDistance(activePoint);
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

        // Not visible: the half edge we came through is a horizon edge
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

      // Disable visible faces and their interior half edges; collect their
      // conflict segments as the source set for the repartition kernel.
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

      // Create new faces around the horizon loop (verbatim topology)
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
        const Vec3<T> planeNormal =
            triangleNormal(pt(A), pt(hev[1]), activePoint);
        newFace.P = Plane<T>(planeNormal, activePoint);
        newFace.he = AB;
        m_newPlanes.push_back(newFace.P);

        m_mesh.halfEdges[CA].opp =
            m_newHalfEdgeIndices[i > 0 ? i * 2 - 1 : 2 * horizonEdgeCount - 1];
        m_mesh.halfEdges[BC].opp =
            m_newHalfEdgeIndices[((i + 1) * 2) % (horizonEdgeCount * 2)];
      }

      // Reassign conflict points of the disabled faces to the new faces.
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

  void buildMesh(const T *vertexData, size_t vertexCount, T epsilon) {
    m_mesh = Mesh();
    m_planar = false;
    m_scale = 0;
    m_diagnostics = DiagnosticsData();
    m_arenaTail = 0;
    if (vertexCount == 0) {
      m_numPoints = 0;
      return;
    }
    m_numPoints = vertexCount;

    // +1 spare slot for the planar-case synthetic point
    m_points = PointsView(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "qh::points"),
        vertexCount + 1);
    m_pointsHost = Kokkos::create_mirror_view(m_points);
    for (size_t i = 0; i < vertexCount; i++) {
      m_pointsHost(i, 0) = vertexData[3 * i + 0];
      m_pointsHost(i, 1) = vertexData[3 * i + 1];
      m_pointsHost(i, 2) = vertexData[3 * i + 2];
    }
    Kokkos::deep_copy(m_points, m_pointsHost);

    if (m_arena.extent(0) < 2 * vertexCount + 64)
      m_arena = Kokkos::View<IndexT *, MemSpace>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "qh::arena"),
          2 * vertexCount + 64);

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

  // Extract triangle indices by flood fill over enabled faces (as original)
  ConvexHull<T> extractHull(bool CCW, size_t /*originalCount*/) {
    ConvexHull<T> hull;
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
    }

    // reduced representation
    std::vector<long long> remap(m_numPoints + 1, -1);
    hull.m_reducedIndices.reserve(hull.m_indices.size());
    for (size_t idx : hull.m_indices) {
      if (remap[idx] < 0) {
        remap[idx] = (long long)hull.m_hullVertices.size();
        hull.m_hullVertices.push_back(pt(idx));
        hull.m_hullVertexSource.push_back(idx);
      }
      hull.m_reducedIndices.push_back((size_t)remap[idx]);
    }
    return hull;
  }
};

} // namespace kokkos_quickhull
