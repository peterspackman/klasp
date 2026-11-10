#pragma once
// The prior a CSP landscape is drawn from: how one quasi-random point in
// [0,1)^D becomes a pose for every independent molecule and a unit cell.
// Chain samplers must propose from this same distribution.
//
// Kept header-only and inlined so the arithmetic (and FMA contraction) matches
// the generator kernel bit for bit.
//
// The length prior depends on the rotated molecule, so the draw is two steps:
//     PriorDraw dof;
//     kgf_point(seed, prior.ndim(), qr);
//     if (!prior_draw_pose(prior, qr, dof)) return;     // rejected angles
//     ... rotate the basis by dof.rot, measure s_mins ...
//     if (!prior_accepts_volume(prior, s_mins, dof.sqrt_x)) return;
//     prior_draw_lengths(prior, qr, seed, s_mins, dof);

#include "klasp/generate_kernel.hpp"   // gpu_map_angle
#include "klasp/generator.hpp"         // MAX_INDEP
#include "klasp/kgf.hpp"
#include "klasp/rotation.hpp"
#include "klasp/sgdata.hpp"
#include "klasp/types.hpp"

namespace klasp {

// Largest quasi-random point the prior ever asks for: 6 DOFs per independent
// molecule plus at most 6 free cell parameters, rounded up.
static constexpr int PRIOR_MAX_DIM = 56;

// The prior for one space group and one Z'; independent of the seed.
struct SamplingPrior {
    LatticeType lattice = TRICLINIC;
    int n_indep = 1;
    int n_free_angles = 0;
    int n_free_lengths = 0;
    // The cell-angle window, radians.
    RealType amin_rad = static_cast<RealType>(45.0 * M_PI / 180.0);
    RealType amax_rad = static_cast<RealType>(135.0 * M_PI / 180.0);
    // Volume budget in Angstrom^3: molecular volume x n_symops x tvp, floored
    // for lattices whose density comes from the contraction pass.
    RealType max_volume = 0;

    KOKKOS_INLINE_FUNCTION
    int ndim() const { return 6 * n_indep + n_free_angles + n_free_lengths; }
};

// Fill in lattice and the free-parameter counts from a space group number. Host only.
inline SamplingPrior make_sampling_prior(int space_group, int n_indep,
                                         double max_volume) {
    SamplingPrior p;
    p.lattice = get_space_group(space_group).lattice;
    p.n_indep = n_indep;
    lattice_free_params(p.lattice, p.n_free_lengths, p.n_free_angles);
    p.max_volume = static_cast<RealType>(max_volume);
    return p;
}

// One draw: the DOFs of one structure before clash resolution, which mutates
// `trans`, `a`, `b` and `c` in place.
struct PriorDraw {
    RealType trans[MAX_INDEP][3];    // fractional COM per independent molecule
    RealType rot[MAX_INDEP][3][3];   // body -> cell rotation
    RealType rot_vec[MAX_INDEP][3];  // the same rotation as angle-axis
    RealType alpha, beta, gamma;     // cell angles, radians
    // Cached trig on the angles; a * b * c * sqrt_x is the volume.
    RealType ca, cb, cg, sg_val, sqrt_x;
    RealType a, b, c;                // cell lengths, Angstrom
};

// Pose and cell angles. Translations are the quasi-random coordinates; rotations
// use vec2rot (Shoemake, uniform on SO(3)). Returns false when the angles close
// on no real volume (a rejected seed).
KOKKOS_INLINE_FUNCTION
bool prior_draw_pose(const SamplingPrior& p, const RealType* qr, PriorDraw& d) {
    for (int m = 0; m < p.n_indep; ++m) {
        d.trans[m][0] = qr[6*m + 0];
        d.trans[m][1] = qr[6*m + 1];
        d.trans[m][2] = qr[6*m + 2];
        RealType rv[3] = {qr[6*m + 3], qr[6*m + 4], qr[6*m + 5]};
        vec2rot(rv, d.rot[m]);
        rotation_matrix_to_angle_axis(d.rot[m], d.rot_vec[m]);
    }

    // Shared cell angles; dims start at offset 6*n_indep.
    const RealType* qa = &qr[6 * p.n_indep];
    RealType alpha, beta, gamma;
    switch (p.lattice) {
        case TRICLINIC:
            alpha = gpu_map_angle(qa[0], p.amin_rad, p.amax_rad);
            beta  = gpu_map_angle(qa[1], p.amin_rad, p.amax_rad);
            gamma = gpu_map_angle(qa[2], p.amin_rad, p.amax_rad);
            break;
        case MONOCLINIC:
            alpha = M_PI/2; beta = gpu_map_angle(qa[0], p.amin_rad, p.amax_rad); gamma = M_PI/2;
            break;
        case RHOMBOHEDRAL:
            alpha = beta = gamma = gpu_map_angle(qa[0], p.amin_rad, p.amax_rad);
            break;
        case HEXAGONAL:
            alpha = M_PI/2; beta = M_PI/2; gamma = 2.0*M_PI/3.0;
            break;
        default:
            alpha = beta = gamma = M_PI/2;
            break;
    }

    RealType ca = Kokkos::cos(alpha), cb = Kokkos::cos(beta), cg = Kokkos::cos(gamma);
    RealType sg_val = Kokkos::sin(gamma);
    RealType x_val = 1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg;
    if (x_val <= 0) return false;
    RealType sqrt_x = Kokkos::sqrt(x_val);

    d.alpha = alpha; d.beta = beta; d.gamma = gamma;
    d.ca = ca; d.cb = cb; d.cg = cg;
    d.sg_val = sg_val; d.sqrt_x = sqrt_x;
    return true;
}

// Whether the tightest cell this pose can occupy fits in the volume budget.
// `s_mins` is the per-axis minimum cell length in Angstrom.
KOKKOS_INLINE_FUNCTION
bool prior_accepts_volume(const SamplingPrior& p, const RealType s_mins[3],
                          RealType sqrt_x) {
    return !(s_mins[0] * s_mins[1] * s_mins[2] * sqrt_x > p.max_volume);
}

// Whether the prior could have produced this cell: all three angles in the
// window and the volume within budget. Lengths in Angstrom, angles in radians.
KOKKOS_INLINE_FUNCTION
bool prior_accepts_cell(const SamplingPrior& p, RealType a, RealType b, RealType c,
                        RealType alpha, RealType beta, RealType gamma) {
    if (alpha < p.amin_rad || alpha > p.amax_rad) return false;
    if (beta  < p.amin_rad || beta  > p.amax_rad) return false;
    if (gamma < p.amin_rad || gamma > p.amax_rad) return false;
    const RealType ca = Kokkos::cos(alpha), cb = Kokkos::cos(beta), cg = Kokkos::cos(gamma);
    const RealType x = 1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg;
    if (x <= 0) return false;   // the angles close on no real volume
    const RealType lens[3] = {a, b, c};
    return prior_accepts_volume(p, lens, Kokkos::sqrt(x));
}

// Cell lengths, conditioned on the pose through `s_mins` and the angles through
// `d.sqrt_x`. Each free length is uniform between its floor and the ceiling the
// remaining volume budget allows; `seed` picks the axis order to avoid biasing
// the first axis long, so the lengths depend on the seed value as well as `qr`.
KOKKOS_INLINE_FUNCTION
void prior_draw_lengths(const SamplingPrior& p, const RealType* qr, int seed,
                        const RealType s_mins[3], PriorDraw& d) {
    const RealType max_vol = p.max_volume;
    const RealType sqrt_x = d.sqrt_x;
    const RealType* lp = &qr[6 * p.n_indep + p.n_free_angles];
    RealType a, b, c;
    int perms[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    if (p.lattice == TRICLINIC || p.lattice == MONOCLINIC || p.lattice == ORTHORHOMBIC) {
        int px = perms[seed % 6][0], py = perms[seed % 6][1], pz = perms[seed % 6][2];
        RealType v_len[3];
        RealType s_max = max_vol / (sqrt_x * s_mins[py] * s_mins[pz]);
        v_len[px] = s_mins[px] + lp[px] * (s_max - s_mins[px]);
        s_max = max_vol / (sqrt_x * v_len[px] * s_mins[pz]);
        v_len[py] = s_mins[py] + lp[py] * (s_max - s_mins[py]);
        s_max = max_vol / (sqrt_x * v_len[px] * v_len[py]);
        v_len[pz] = s_mins[pz] + lp[pz] * (s_max - s_mins[pz]);
        a = v_len[0]; b = v_len[1]; c = v_len[2];
    } else if (p.lattice == TETRAGONAL || p.lattice == HEXAGONAL) {
        RealType s_min_a = Kokkos::fmax(s_mins[0], s_mins[1]);
        RealType s_min_c = s_mins[2];
        if (seed % 2 == 0) {
            RealType s_max = Kokkos::sqrt(max_vol / (sqrt_x * s_min_c));
            a = s_min_a + lp[0] * (s_max - s_min_a);
            s_max = max_vol / (sqrt_x * a * a);
            c = s_min_c + lp[1] * (s_max - s_min_c);
        } else {
            RealType s_max = max_vol / (sqrt_x * s_min_a * s_min_a);
            c = s_min_c + lp[1] * (s_max - s_min_c);
            s_max = Kokkos::sqrt(max_vol / (sqrt_x * c));
            a = s_min_a + lp[0] * (s_max - s_min_a);
        }
        b = a;
    } else {
        RealType s_max = Kokkos::cbrt(max_vol / sqrt_x);
        RealType s_min = Kokkos::fmax(Kokkos::fmax(s_mins[0], s_mins[1]), s_mins[2]);
        a = s_min + lp[0] * (s_max - s_min);
        b = c = a;
    }
    d.a = a; d.b = b; d.c = c;
}

// The whole draw in one call. Returns false for a seed the prior rejects.
template <typename MeasureFn>
KOKKOS_INLINE_FUNCTION
bool prior_draw(const SamplingPrior& p, int seed, RealType* qr,
                const MeasureFn& measure_s_mins, PriorDraw& d) {
    kgf_point(seed, p.ndim(), qr);
    if (!prior_draw_pose(p, qr, d)) return false;
    RealType s_mins[3];
    measure_s_mins(d, s_mins);
    if (!prior_accepts_volume(p, s_mins, d.sqrt_x)) return false;
    prior_draw_lengths(p, qr, seed, s_mins, d);
    return true;
}

}  // namespace klasp
