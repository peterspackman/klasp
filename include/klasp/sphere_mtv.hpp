#pragma once
// Host/device convex minimum-translation-vector (MTV) between two sets of
// spheres, callable inside the generate kernel. The exact host reference is
// KokkosSphereClash::hullMTV.
//
// Both sets are treated as the convex hull of their spheres. The signed depth is
//
//     s* = min_{|u|=1} h(u),   h(u) = max_i (cA_i.u + rA_i)
//                                   + max_j (-cB_j.u + rB_j),
//
// the support function of A - B minimised over the unit sphere. s* > 0 means
// the hulls penetrate by s*; s* < 0 means they are separated by -s*. Shifting B
// by s*.u* makes them touch. h is minimised from a set of seed directions by
// projected gradient descent; callers re-verify against the exact atom detector.

#include <Kokkos_Core.hpp>
#include "klasp/types.hpp"

namespace klasp {

struct MTVResult {
    RealType depth;    // >0: convex hulls penetrate by this much; <0: -clearance
    RealType dir[3];   // unit; shift set B by depth*dir to make the hulls touch
};

namespace detail {

// h(u) and its gradient witness w = cA[a*] - cB[b*] (a*, b* the active support
// spheres). Centers are interleaved x,y,z; radii are one per sphere.
KOKKOS_INLINE_FUNCTION
void mtv_support_witness(const RealType* cA, const RealType* rA, int nA,
                         const RealType* cB, const RealType* rB, int nB,
                         const RealType u[3], RealType& h, RealType w[3]) {
    RealType ha = RealType(-1e30); int ai = 0;
    for (int i = 0; i < nA; ++i) {
        const RealType d = cA[3*i]*u[0] + cA[3*i+1]*u[1] + cA[3*i+2]*u[2] + rA[i];
        if (d > ha) { ha = d; ai = i; }
    }
    RealType hb = RealType(-1e30); int bi = 0;
    for (int j = 0; j < nB; ++j) {
        const RealType d = -(cB[3*j]*u[0] + cB[3*j+1]*u[1] + cB[3*j+2]*u[2]) + rB[j];
        if (d > hb) { hb = d; bi = j; }
    }
    h = ha + hb;
    w[0] = cA[3*ai+0] - cB[3*bi+0];
    w[1] = cA[3*ai+1] - cB[3*bi+1];
    w[2] = cA[3*ai+2] - cB[3*bi+2];
}

}  // namespace detail

// Convex MTV between sphere sets A and B, from `n_seed` spherical-Fibonacci seed
// directions. Returns depth = min_u h(u) and the unit minimiser dir.
KOKKOS_INLINE_FUNCTION
MTVResult sphere_set_mtv(const RealType* cA, const RealType* rA, int nA,
                         const RealType* cB, const RealType* rB, int nB,
                         int n_seed = 48) {
    MTVResult out;
    out.depth = RealType(-1e30);
    out.dir[0] = 0; out.dir[1] = 0; out.dir[2] = 1;
    if (nA <= 0 || nB <= 0) return out;

    RealType best_h = RealType(1e30);
    RealType best_u[3] = {0, 0, 1};

    // Seed 0: centroid difference (points from A's centroid toward B's), the
    // near-optimal direction for the common shallow-penetration case.
    {
        RealType ca[3] = {0, 0, 0}, cb[3] = {0, 0, 0};
        for (int i = 0; i < nA; ++i) { ca[0]+=cA[3*i]; ca[1]+=cA[3*i+1]; ca[2]+=cA[3*i+2]; }
        for (int j = 0; j < nB; ++j) { cb[0]+=cB[3*j]; cb[1]+=cB[3*j+1]; cb[2]+=cB[3*j+2]; }
        const RealType ia = RealType(1)/nA, ib = RealType(1)/nB;
        RealType u[3] = {cb[0]*ib - ca[0]*ia, cb[1]*ib - ca[1]*ia, cb[2]*ib - ca[2]*ia};
        RealType n2 = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
        if (n2 > 0) {
            const RealType inv = RealType(1)/Kokkos::sqrt(n2);
            u[0]*=inv; u[1]*=inv; u[2]*=inv;
            RealType h, w[3];
            detail::mtv_support_witness(cA,rA,nA, cB,rB,nB, u, h, w);
            if (h < best_h) { best_h = h; best_u[0]=u[0]; best_u[1]=u[1]; best_u[2]=u[2]; }
        }
    }

    // Spherical-Fibonacci seed spread.
    const RealType GA = RealType(M_PI) * (RealType(3) - Kokkos::sqrt(RealType(5)));
    for (int k = 0; k < n_seed; ++k) {
        const RealType z = RealType(1) - RealType(2)*(RealType(k)+RealType(0.5))/RealType(n_seed);
        const RealType rr = Kokkos::sqrt(Kokkos::fmax(RealType(0), RealType(1) - z*z));
        RealType u[3] = {rr*Kokkos::cos(GA*k), rr*Kokkos::sin(GA*k), z};
        RealType h, w[3];
        detail::mtv_support_witness(cA,rA,nA, cB,rB,nB, u, h, w);
        if (h < best_h) { best_h = h; best_u[0]=u[0]; best_u[1]=u[1]; best_u[2]=u[2]; }
    }

    // Projected gradient descent on the unit sphere from the best seed, with a
    // backtracking line search on h. The Riemannian gradient is w projected off
    // u; |w_perp| -> 0 means u is aligned with a face normal (a local min of h).
    RealType u[3] = {best_u[0], best_u[1], best_u[2]};
    RealType h, w[3];
    detail::mtv_support_witness(cA,rA,nA, cB,rB,nB, u, h, w);
    RealType step = RealType(0.5);
    for (int it = 0; it < 32; ++it) {
        const RealType wu = w[0]*u[0] + w[1]*u[1] + w[2]*u[2];
        RealType wp[3] = {w[0]-wu*u[0], w[1]-wu*u[1], w[2]-wu*u[2]};
        const RealType wn = Kokkos::sqrt(wp[0]*wp[0] + wp[1]*wp[1] + wp[2]*wp[2]);
        if (wn < RealType(1e-12)) break;
        const RealType ig = RealType(1)/wn;
        wp[0]*=ig; wp[1]*=ig; wp[2]*=ig;
        bool improved = false;
        for (int ls = 0; ls < 8; ++ls) {
            RealType un[3] = {u[0]-step*wp[0], u[1]-step*wp[1], u[2]-step*wp[2]};
            const RealType inv = RealType(1)/Kokkos::sqrt(un[0]*un[0]+un[1]*un[1]+un[2]*un[2]);
            un[0]*=inv; un[1]*=inv; un[2]*=inv;
            RealType h2, w2[3];
            detail::mtv_support_witness(cA,rA,nA, cB,rB,nB, un, h2, w2);
            if (h2 < h - RealType(1e-15)) {
                u[0]=un[0]; u[1]=un[1]; u[2]=un[2];
                h = h2; w[0]=w2[0]; w[1]=w2[1]; w[2]=w2[2];
                improved = true;
                break;
            }
            step *= RealType(0.5);
        }
        if (!improved) { step *= RealType(0.5); if (step < RealType(1e-6)) break; }
    }
    if (h < best_h) { best_h = h; best_u[0]=u[0]; best_u[1]=u[1]; best_u[2]=u[2]; }

    out.depth = best_h;
    out.dir[0] = best_u[0]; out.dir[1] = best_u[1]; out.dir[2] = best_u[2];
    return out;
}

}  // namespace klasp
