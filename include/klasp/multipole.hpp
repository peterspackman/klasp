#pragma once

#include <Kokkos_Core.hpp>
#include "klasp/types.hpp"
#include "klasp/multipole_contraction.hpp"

namespace klasp {

// ---------------------------------------------------------------------------
// Constexpr Hermite indexing (matches occ::ints::hermite_index)
// ---------------------------------------------------------------------------

// Number of Hermite components at rank L: (L+1)(L+2)/2
constexpr int nherm(int L) {
    return (L + 1) * (L + 2) / 2;
}

// Total components from rank 0..L: (L+1)(L+2)(L+3)/6
constexpr int nhermsum(int L) {
    return (L + 1) * (L + 2) * (L + 3) / 6;
}

// Linear index for Hermite Gaussian (t,u,v)
KOKKOS_INLINE_FUNCTION
constexpr int hermite_index(int t, int u, int v) {
    int L = t + u + v;
    int offset = (L > 0) ? nhermsum(L - 1) : 0;
    int uv = u + v;
    return offset + uv * (uv + 1) / 2 + v;
}

// Constexpr factorial (up to 8! = 40320)
constexpr double factorial(int n) {
    double r = 1.0;
    for (int i = 2; i <= n; ++i) r *= i;
    return r;
}

// ---------------------------------------------------------------------------
// TUV entry: maps linear index -> (t, u, v)
// ---------------------------------------------------------------------------

struct TUV {
    int t, u, v;
};

// Build a single TUV entry for a given linear index, searching through ranks.
constexpr TUV make_tuv(int idx) {
    // Iterate over ranks and (t,u,v) within each rank
    for (int L = 0; L <= 8; ++L) {
        for (int t = L; t >= 0; --t) {
            for (int uv = L - t; uv >= 0; --uv) {
                int u = L - t - uv;
                int v = uv;
                if (hermite_index(t, u, v) == idx)
                    return {t, u, v};
            }
        }
    }
    return {0, 0, 0}; // unreachable
}

// ---------------------------------------------------------------------------
// MultipoleConstants<MaxRank>: all compile-time tables
// ---------------------------------------------------------------------------

template <int MaxRank>
struct MultipoleConstants {
    static constexpr int N_CART = nhermsum(MaxRank);

    TUV tuv_table[N_CART];

    int rank_table[N_CART];

    double sign_inv_fact[N_CART]; // (-1)^l / (t!u!v!)
    double inv_fact[N_CART];      // 1 / (t!u!v!)

    constexpr MultipoleConstants() : tuv_table{}, rank_table{},
                                      sign_inv_fact{}, inv_fact{} {
        for (int i = 0; i < N_CART; ++i) {
            tuv_table[i] = make_tuv(i);
            int L = tuv_table[i].t + tuv_table[i].u + tuv_table[i].v;
            rank_table[i] = L;
            double ifact = 1.0 / (factorial(tuv_table[i].t)
                                 * factorial(tuv_table[i].u)
                                 * factorial(tuv_table[i].v));
            double sign = (L % 2 == 0) ? 1.0 : -1.0;
            sign_inv_fact[i] = sign * ifact;
            inv_fact[i] = ifact;
        }
    }
};

// ---------------------------------------------------------------------------
// InteractionConstants<MaxOrder>: tables for T-tensor contraction
// ---------------------------------------------------------------------------

template <int MaxOrder>
struct InteractionConstants {
    // MaxOrder = max(rankA + rankB) to include in contraction
    static constexpr int N_CART = nhermsum(4); // always 35 for MaxRank=4
    static constexpr int N_T_ENERGY = nhermsum(MaxOrder);
    static constexpr int N_T_FORCE = nhermsum(MaxOrder + 1);

    // Byte-wide indices keep this struct small for the constant cache; see the
    // static constexpr note below.
    static_assert(N_T_FORCE <= 256,
                  "T-tensor index no longer fits a byte; widen the tables");

    // For each (i, j) pair: ADD_TABLE[i][j] = hermite_index(ta+tb, ua+ub, va+vb)
    uint8_t add_table[N_CART][N_CART];

    // Force tables: shifted by +1 in each direction
    uint8_t force_x_table[N_CART][N_CART];
    uint8_t force_y_table[N_CART][N_CART];
    uint8_t force_z_table[N_CART][N_CART];

    // Active mask: rank[i] + rank[j] <= MaxOrder, and callers pass
    // MaxOrder = MaxRank. A "rank 4" run keeps interactions to 1/R^5, not the
    // full lA + lB <= 8 between two rank-4 sites.
    bool active[N_CART][N_CART];

    constexpr InteractionConstants() : add_table{}, force_x_table{},
                                        force_y_table{}, force_z_table{},
                                        active{} {
        // Not static: a static local in a constexpr function is C++23.
        constexpr MultipoleConstants<4> mc{};
        for (int i = 0; i < N_CART; ++i) {
            auto [ta, ua, va] = mc.tuv_table[i];
            int la = ta + ua + va;
            for (int j = 0; j < N_CART; ++j) {
                auto [tb, ub, vb] = mc.tuv_table[j];
                int lb = tb + ub + vb;
                active[i][j] = (la + lb <= MaxOrder);
                if (active[i][j]) {
                    add_table[i][j] = hermite_index(ta+tb, ua+ub, va+vb);
                    force_x_table[i][j] = hermite_index(ta+tb+1, ua+ub, va+vb);
                    force_y_table[i][j] = hermite_index(ta+tb, ua+ub+1, va+vb);
                    force_z_table[i][j] = hermite_index(ta+tb, ua+ub, va+vb+1);
                } else {
                    add_table[i][j] = 0;
                    force_x_table[i][j] = 0;
                    force_y_table[i][j] = 0;
                    force_z_table[i][j] = 0;
                }
            }
        }
    }
};

// Device-side uses of these tables must declare the instance `static constexpr`,
// not plain `constexpr`: nvcc gives a plain constexpr local a per-thread stack
// copy (about 21 KB for InteractionConstants), while static storage is shared
// through the constant cache.
inline constexpr MultipoleConstants<4> MC4{};
inline constexpr InteractionConstants<4> IC4{};

// ---------------------------------------------------------------------------
// T-tensor computation via Obara-Saika recurrence
// ---------------------------------------------------------------------------

// Two forms of each: one takes 1/R^2 and 1/R from the caller, the other
// computes them from R and delegates.
template <int MaxOrder>
KOKKOS_INLINE_FUNCTION
void compute_t_tensor(RealType Rx, RealType Ry, RealType Rz,
                      RealType invR2, RealType invR,
                      RealType T[nhermsum(MaxOrder)]) {
    constexpr int nsz = nhermsum(MaxOrder);
    constexpr int num_aux = MaxOrder + 1;

    // Auxiliary integrals: R[m * nsz + idx]
    RealType R_arr[num_aux * nsz];
    for (int i = 0; i < num_aux * nsz; ++i) R_arr[i] = 0.0;

    // Base case: T^(m)_{000}
    constexpr int hi000 = hermite_index(0, 0, 0);
    R_arr[0 * nsz + hi000] = invR;
    for (int m = 1; m < num_aux; ++m) {
        R_arr[m * nsz + hi000] = -(2 * m - 1) * invR2 * R_arr[(m - 1) * nsz + hi000];
    }

    // Build up t (x-direction)
    for (int t = 0; t < MaxOrder; ++t) {
        for (int m = 0; m < MaxOrder - t; ++m) {
            RealType val = Rx * R_arr[(m + 1) * nsz + hermite_index(t, 0, 0)];
            if (t > 0)
                val += t * R_arr[(m + 1) * nsz + hermite_index(t - 1, 0, 0)];
            R_arr[m * nsz + hermite_index(t + 1, 0, 0)] = val;
        }
    }

    // Build up u (y-direction)
    for (int t = 0; t <= MaxOrder; ++t) {
        for (int u = 0; u < MaxOrder - t; ++u) {
            for (int m = 0; m < MaxOrder - t - u; ++m) {
                RealType val = Ry * R_arr[(m + 1) * nsz + hermite_index(t, u, 0)];
                if (u > 0)
                    val += u * R_arr[(m + 1) * nsz + hermite_index(t, u - 1, 0)];
                R_arr[m * nsz + hermite_index(t, u + 1, 0)] = val;
            }
        }
    }

    // Build up v (z-direction)
    for (int t = 0; t <= MaxOrder; ++t) {
        for (int u = 0; u <= MaxOrder - t; ++u) {
            for (int v = 0; v < MaxOrder - t - u; ++v) {
                for (int m = 0; m < MaxOrder - t - u - v; ++m) {
                    RealType val = Rz * R_arr[(m + 1) * nsz + hermite_index(t, u, v)];
                    if (v > 0)
                        val += v * R_arr[(m + 1) * nsz + hermite_index(t, u, v - 1)];
                    R_arr[m * nsz + hermite_index(t, u, v + 1)] = val;
                }
            }
        }
    }

    // Extract m=0 layer
    for (int i = 0; i < nsz; ++i)
        T[i] = R_arr[i];
}

template <int MaxOrder>
KOKKOS_INLINE_FUNCTION
void compute_t_tensor(RealType Rx, RealType Ry, RealType Rz,
                      RealType T[nhermsum(MaxOrder)]) {
    const RealType R2 = Rx * Rx + Ry * Ry + Rz * Rz;
    const RealType invR2 = 1.0 / R2;
    compute_t_tensor<MaxOrder>(Rx, Ry, Rz, invR2, Kokkos::sqrt(invR2), T);
}

// ---------------------------------------------------------------------------
// Hardcoded T-tensor specializations (generated, branch-free, no array overhead)
// ---------------------------------------------------------------------------

template <>
KOKKOS_INLINE_FUNCTION
void compute_t_tensor<2>(RealType Rx, RealType Ry, RealType Rz,
                      RealType invR2, RealType invR,
                      RealType T[10]) {
    RealType R0_000 = invR;
    RealType R1_000 = -1.0 * invR2 * R0_000;
    RealType R2_000 = -3.0 * invR2 * R1_000;

    RealType R0_100 = Rx * R1_000;
    RealType R1_100 = Rx * R2_000;
    RealType R0_200 = Rx * R1_100 + 1.0 * R1_000;

    RealType R0_010 = Ry * R1_000;
    RealType R1_010 = Ry * R2_000;
    RealType R0_020 = Ry * R1_010 + 1.0 * R1_000;
    RealType R0_110 = Ry * R1_100;

    RealType R0_001 = Rz * R1_000;
    RealType R1_001 = Rz * R2_000;
    RealType R0_002 = Rz * R1_001 + 1.0 * R1_000;
    RealType R0_011 = Rz * R1_010;
    RealType R0_101 = Rz * R1_100;

    T[0] = R0_000; T[1] = R0_100; T[2] = R0_010; T[3] = R0_001;
    T[4] = R0_200; T[5] = R0_110; T[6] = R0_101;
    T[7] = R0_020; T[8] = R0_011; T[9] = R0_002;
}

template <>
KOKKOS_INLINE_FUNCTION
void compute_t_tensor<5>(RealType Rx, RealType Ry, RealType Rz,
                      RealType invR2, RealType invR,
                      RealType T[56]) {
    // Base case: R^(m)_{000}
    RealType R0_000 = invR;
    RealType R1_000 = -invR2 * R0_000;
    RealType R2_000 = -3.0 * invR2 * R1_000;
    RealType R3_000 = -5.0 * invR2 * R2_000;
    RealType R4_000 = -7.0 * invR2 * R3_000;
    RealType R5_000 = -9.0 * invR2 * R4_000;

    // X-direction
    RealType R0_100 = Rx * R1_000;
    RealType R1_100 = Rx * R2_000;
    RealType R2_100 = Rx * R3_000;
    RealType R3_100 = Rx * R4_000;
    RealType R4_100 = Rx * R5_000;
    RealType R0_200 = Rx * R1_100 + R1_000;
    RealType R1_200 = Rx * R2_100 + R2_000;
    RealType R2_200 = Rx * R3_100 + R3_000;
    RealType R3_200 = Rx * R4_100 + R4_000;
    RealType R0_300 = Rx * R1_200 + 2.0 * R1_100;
    RealType R1_300 = Rx * R2_200 + 2.0 * R2_100;
    RealType R2_300 = Rx * R3_200 + 2.0 * R3_100;
    RealType R0_400 = Rx * R1_300 + 3.0 * R1_200;
    RealType R1_400 = Rx * R2_300 + 3.0 * R2_200;
    RealType R0_500 = Rx * R1_400 + 4.0 * R1_300;

    // Y-direction
    RealType R0_010 = Ry * R1_000;
    RealType R1_010 = Ry * R2_000;
    RealType R2_010 = Ry * R3_000;
    RealType R3_010 = Ry * R4_000;
    RealType R4_010 = Ry * R5_000;
    RealType R0_020 = Ry * R1_010 + R1_000;
    RealType R1_020 = Ry * R2_010 + R2_000;
    RealType R2_020 = Ry * R3_010 + R3_000;
    RealType R3_020 = Ry * R4_010 + R4_000;
    RealType R0_030 = Ry * R1_020 + 2.0 * R1_010;
    RealType R1_030 = Ry * R2_020 + 2.0 * R2_010;
    RealType R2_030 = Ry * R3_020 + 2.0 * R3_010;
    RealType R0_040 = Ry * R1_030 + 3.0 * R1_020;
    RealType R1_040 = Ry * R2_030 + 3.0 * R2_020;
    RealType R0_050 = Ry * R1_040 + 4.0 * R1_030;
    RealType R0_110 = Ry * R1_100;
    RealType R1_110 = Ry * R2_100;
    RealType R2_110 = Ry * R3_100;
    RealType R3_110 = Ry * R4_100;
    RealType R0_120 = Ry * R1_110 + R1_100;
    RealType R1_120 = Ry * R2_110 + R2_100;
    RealType R2_120 = Ry * R3_110 + R3_100;
    RealType R0_130 = Ry * R1_120 + 2.0 * R1_110;
    RealType R1_130 = Ry * R2_120 + 2.0 * R2_110;
    RealType R0_140 = Ry * R1_130 + 3.0 * R1_120;
    RealType R0_210 = Ry * R1_200;
    RealType R1_210 = Ry * R2_200;
    RealType R2_210 = Ry * R3_200;
    RealType R0_220 = Ry * R1_210 + R1_200;
    RealType R1_220 = Ry * R2_210 + R2_200;
    RealType R0_230 = Ry * R1_220 + 2.0 * R1_210;
    RealType R0_310 = Ry * R1_300;
    RealType R1_310 = Ry * R2_300;
    RealType R0_320 = Ry * R1_310 + R1_300;
    RealType R0_410 = Ry * R1_400;

    // Z-direction
    RealType R0_001 = Rz * R1_000;
    RealType R1_001 = Rz * R2_000;
    RealType R2_001 = Rz * R3_000;
    RealType R3_001 = Rz * R4_000;
    RealType R4_001 = Rz * R5_000;
    RealType R0_002 = Rz * R1_001 + R1_000;
    RealType R1_002 = Rz * R2_001 + R2_000;
    RealType R2_002 = Rz * R3_001 + R3_000;
    RealType R3_002 = Rz * R4_001 + R4_000;
    RealType R0_003 = Rz * R1_002 + 2.0 * R1_001;
    RealType R1_003 = Rz * R2_002 + 2.0 * R2_001;
    RealType R2_003 = Rz * R3_002 + 2.0 * R3_001;
    RealType R0_004 = Rz * R1_003 + 3.0 * R1_002;
    RealType R1_004 = Rz * R2_003 + 3.0 * R2_002;
    RealType R0_005 = Rz * R1_004 + 4.0 * R1_003;
    RealType R0_011 = Rz * R1_010;
    RealType R1_011 = Rz * R2_010;
    RealType R2_011 = Rz * R3_010;
    RealType R3_011 = Rz * R4_010;
    RealType R0_012 = Rz * R1_011 + R1_010;
    RealType R1_012 = Rz * R2_011 + R2_010;
    RealType R2_012 = Rz * R3_011 + R3_010;
    RealType R0_013 = Rz * R1_012 + 2.0 * R1_011;
    RealType R1_013 = Rz * R2_012 + 2.0 * R2_011;
    RealType R0_014 = Rz * R1_013 + 3.0 * R1_012;
    RealType R0_021 = Rz * R1_020;
    RealType R1_021 = Rz * R2_020;
    RealType R2_021 = Rz * R3_020;
    RealType R0_022 = Rz * R1_021 + R1_020;
    RealType R1_022 = Rz * R2_021 + R2_020;
    RealType R0_023 = Rz * R1_022 + 2.0 * R1_021;
    RealType R0_031 = Rz * R1_030;
    RealType R1_031 = Rz * R2_030;
    RealType R0_032 = Rz * R1_031 + R1_030;
    RealType R0_041 = Rz * R1_040;
    RealType R0_101 = Rz * R1_100;
    RealType R1_101 = Rz * R2_100;
    RealType R2_101 = Rz * R3_100;
    RealType R3_101 = Rz * R4_100;
    RealType R0_102 = Rz * R1_101 + R1_100;
    RealType R1_102 = Rz * R2_101 + R2_100;
    RealType R2_102 = Rz * R3_101 + R3_100;
    RealType R0_103 = Rz * R1_102 + 2.0 * R1_101;
    RealType R1_103 = Rz * R2_102 + 2.0 * R2_101;
    RealType R0_104 = Rz * R1_103 + 3.0 * R1_102;
    RealType R0_111 = Rz * R1_110;
    RealType R1_111 = Rz * R2_110;
    RealType R2_111 = Rz * R3_110;
    RealType R0_112 = Rz * R1_111 + R1_110;
    RealType R1_112 = Rz * R2_111 + R2_110;
    RealType R0_113 = Rz * R1_112 + 2.0 * R1_111;
    RealType R0_121 = Rz * R1_120;
    RealType R1_121 = Rz * R2_120;
    RealType R0_122 = Rz * R1_121 + R1_120;
    RealType R0_131 = Rz * R1_130;
    RealType R0_201 = Rz * R1_200;
    RealType R1_201 = Rz * R2_200;
    RealType R2_201 = Rz * R3_200;
    RealType R0_202 = Rz * R1_201 + R1_200;
    RealType R1_202 = Rz * R2_201 + R2_200;
    RealType R0_203 = Rz * R1_202 + 2.0 * R1_201;
    RealType R0_211 = Rz * R1_210;
    RealType R1_211 = Rz * R2_210;
    RealType R0_212 = Rz * R1_211 + R1_210;
    RealType R0_221 = Rz * R1_220;
    RealType R0_301 = Rz * R1_300;
    RealType R1_301 = Rz * R2_300;
    RealType R0_302 = Rz * R1_301 + R1_300;
    RealType R0_311 = Rz * R1_310;
    RealType R0_401 = Rz * R1_400;

    T[0] = R0_000; T[1] = R0_100; T[2] = R0_010; T[3] = R0_001;
    T[4] = R0_200; T[5] = R0_110; T[6] = R0_101;
    T[7] = R0_020; T[8] = R0_011; T[9] = R0_002;
    T[10] = R0_300; T[11] = R0_210; T[12] = R0_201;
    T[13] = R0_120; T[14] = R0_111; T[15] = R0_102;
    T[16] = R0_030; T[17] = R0_021; T[18] = R0_012; T[19] = R0_003;
    T[20] = R0_400; T[21] = R0_310; T[22] = R0_301;
    T[23] = R0_220; T[24] = R0_211; T[25] = R0_202;
    T[26] = R0_130; T[27] = R0_121; T[28] = R0_112; T[29] = R0_103;
    T[30] = R0_040; T[31] = R0_031; T[32] = R0_022;
    T[33] = R0_013; T[34] = R0_004;
    T[35] = R0_500; T[36] = R0_410; T[37] = R0_401;
    T[38] = R0_320; T[39] = R0_311; T[40] = R0_302;
    T[41] = R0_230; T[42] = R0_221; T[43] = R0_212; T[44] = R0_203;
    T[45] = R0_140; T[46] = R0_131; T[47] = R0_122;
    T[48] = R0_113; T[49] = R0_104;
    T[50] = R0_050; T[51] = R0_041; T[52] = R0_032;
    T[53] = R0_023; T[54] = R0_014; T[55] = R0_005;
}

// ---------------------------------------------------------------------------
// Multipole pair energy (energy only)
// ---------------------------------------------------------------------------

template <int MaxOrder>
KOKKOS_INLINE_FUNCTION
RealType multipole_pair_energy(const RealType cart_a[nhermsum(4)],
                               const RealType cart_b[nhermsum(4)],
                               RealType Rx, RealType Ry, RealType Rz) {
    static constexpr InteractionConstants<MaxOrder> ic{};
    static constexpr MultipoleConstants<4> mc{};
    constexpr int N = nhermsum(4);

    // Compute T-tensor at MaxOrder
    RealType T[nhermsum(MaxOrder)];
    compute_t_tensor<MaxOrder>(Rx, Ry, Rz, T);

    // Contract: E = sum_{i,j active} wA[i] * T[add[i][j]] * wB[j]
    RealType E = 0.0;
    for (int i = 0; i < N; ++i) {
        RealType wA_i = mc.sign_inv_fact[i] * cart_a[i];
        if (wA_i == 0.0) continue;
        for (int j = 0; j < N; ++j) {
            if (!ic.active[i][j]) continue;
            RealType wB_j = mc.inv_fact[j] * cart_b[j];
            E += wA_i * T[ic.add_table[i][j]] * wB_j;
        }
    }
    return E;
}

// ---------------------------------------------------------------------------
// Multipole pair energy + forces (w.r.t. displacement vector)
// ---------------------------------------------------------------------------

template <int MaxOrder>
KOKKOS_INLINE_FUNCTION
RealType multipole_pair_energy_forces(const RealType cart_a[nhermsum(4)],
                                      const RealType cart_b[nhermsum(4)],
                                      RealType Rx, RealType Ry, RealType Rz,
                                      RealType force[3]) {
    static constexpr InteractionConstants<MaxOrder> ic{};
    static constexpr MultipoleConstants<4> mc{};
    constexpr int N = nhermsum(4);

    // Compute T-tensor at MaxOrder+1 (one higher for gradient)
    RealType T[nhermsum(MaxOrder + 1)];
    compute_t_tensor<MaxOrder + 1>(Rx, Ry, Rz, T);

    RealType E = 0.0;
    force[0] = force[1] = force[2] = 0.0;

    for (int i = 0; i < N; ++i) {
        RealType wA_i = mc.sign_inv_fact[i] * cart_a[i];
        if (wA_i == 0.0) continue;
        for (int j = 0; j < N; ++j) {
            if (!ic.active[i][j]) continue;
            RealType wB_j = mc.inv_fact[j] * cart_b[j];
            RealType wAB = wA_i * wB_j;

            E += wAB * T[ic.add_table[i][j]];
            force[0] += wAB * T[ic.force_x_table[i][j]];
            force[1] += wAB * T[ic.force_y_table[i][j]];
            force[2] += wAB * T[ic.force_z_table[i][j]];
        }
    }
    return E;
}

// ---------------------------------------------------------------------------
// Multipole pair energy + forces + multipole gradients
// ---------------------------------------------------------------------------

// NeedB: whether dE/d(cart_b) is wanted. With NeedB false dE_dcart_b is not
// touched, and E, force and dE_dcart_a are bitwise identical to NeedB true.
template <int MaxOrder, bool NeedB = true>
KOKKOS_INLINE_FUNCTION
RealType multipole_pair_energy_forces_mp_grad(
    const RealType cart_a[nhermsum(4)],
    const RealType cart_b[nhermsum(4)],
    RealType Rx, RealType Ry, RealType Rz,
    RealType invR2, RealType invR,
    RealType force[3],
    RealType dE_dcart_a[nhermsum(4)],
    RealType dE_dcart_b[nhermsum(4)]) {

    RealType T[nhermsum(MaxOrder + 1)];
    compute_t_tensor<MaxOrder + 1>(Rx, Ry, Rz, invR2, invR, T);

    // The contraction is generated straight-line code: see
    // multipole_contraction.hpp and scripts/gen_multipole_contraction.py.
    return MultipoleContraction<MaxOrder, NeedB>::run(cart_a, cart_b, T, force,
                                                      dE_dcart_a, dE_dcart_b);
}

// The same from R alone, for a caller with no reciprocal to share.
template <int MaxOrder, bool NeedB = true>
KOKKOS_INLINE_FUNCTION
RealType multipole_pair_energy_forces_mp_grad(
    const RealType cart_a[nhermsum(4)],
    const RealType cart_b[nhermsum(4)],
    RealType Rx, RealType Ry, RealType Rz,
    RealType force[3],
    RealType dE_dcart_a[nhermsum(4)],
    RealType dE_dcart_b[nhermsum(4)]) {
    const RealType R2 = Rx * Rx + Ry * Ry + Rz * Rz;
    const RealType invR2 = 1.0 / R2;
    return multipole_pair_energy_forces_mp_grad<MaxOrder, NeedB>(
        cart_a, cart_b, Rx, Ry, Rz, invR2, Kokkos::sqrt(invR2),
        force, dE_dcart_a, dE_dcart_b);
}

// ---------------------------------------------------------------------------
// Multipole pair: dE/d(cart_a) only (for split kernel to reduce register pressure).
// Recomputes T-tensor independently.
// ---------------------------------------------------------------------------

template <int MaxOrder>
KOKKOS_INLINE_FUNCTION
void multipole_pair_grad_a(const RealType cart_a[nhermsum(4)],
                            const RealType cart_b[nhermsum(4)],
                            RealType Rx, RealType Ry, RealType Rz,
                            RealType dE_dcart_a[nhermsum(4)]) {
    static constexpr InteractionConstants<MaxOrder> ic{};
    static constexpr MultipoleConstants<4> mc{};
    constexpr int N = nhermsum(4);

    RealType T[nhermsum(MaxOrder)];
    compute_t_tensor<MaxOrder>(Rx, Ry, Rz, T);

    for (int c = 0; c < N; ++c) dE_dcart_a[c] = 0.0;

    for (int i = 0; i < N; ++i) {
        RealType acc = 0.0;
        for (int j = 0; j < N; ++j) {
            if (!ic.active[i][j]) continue;
            acc += T[ic.add_table[i][j]] * mc.inv_fact[j] * cart_b[j];
        }
        dE_dcart_a[i] = mc.sign_inv_fact[i] * acc;
    }
}

// ---------------------------------------------------------------------------
// Multipole pair: dE/d(cart_b) only (for split kernel to reduce register pressure).
// Recomputes T-tensor independently.
// ---------------------------------------------------------------------------

template <int MaxOrder>
KOKKOS_INLINE_FUNCTION
void multipole_pair_grad_b(const RealType cart_a[nhermsum(4)],
                            const RealType cart_b[nhermsum(4)],
                            RealType Rx, RealType Ry, RealType Rz,
                            RealType dE_dcart_b[nhermsum(4)]) {
    static constexpr InteractionConstants<MaxOrder> ic{};
    static constexpr MultipoleConstants<4> mc{};
    constexpr int N = nhermsum(4);

    RealType T[nhermsum(MaxOrder)];
    compute_t_tensor<MaxOrder>(Rx, Ry, Rz, T);

    for (int c = 0; c < N; ++c) dE_dcart_b[c] = 0.0;

    for (int j = 0; j < N; ++j) {
        RealType acc = 0.0;
        for (int i = 0; i < N; ++i) {
            if (!ic.active[i][j]) continue;
            acc += T[ic.add_table[i][j]] * mc.sign_inv_fact[i] * cart_a[i];
        }
        dE_dcart_b[j] = mc.inv_fact[j] * acc;
    }
}

// ---------------------------------------------------------------------------
// Spherical-to-Cartesian conversion (Stone convention)
// ---------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
void spherical_to_cartesian(const RealType* mp_sph, int n_components,
                            int max_rank, RealType cart[nhermsum(4)]) {
    constexpr int N = nhermsum(4);
    for (int i = 0; i < N; ++i) cart[i] = 0.0;

    // Rank 0: Q00
    cart[hermite_index(0, 0, 0)] = mp_sph[0];

    if (max_rank >= 1 && n_components >= 4) {
        RealType Q10 = mp_sph[1];
        RealType Q11c = mp_sph[2];
        RealType Q11s = mp_sph[3];
        cart[hermite_index(1, 0, 0)] = Q11c;  // x
        cart[hermite_index(0, 1, 0)] = Q11s;  // y
        cart[hermite_index(0, 0, 1)] = Q10;   // z
    }

    if (max_rank >= 2 && n_components >= 9) {
        RealType Q20 = mp_sph[4];
        RealType Q21c = mp_sph[5];
        RealType Q21s = mp_sph[6];
        RealType Q22c = mp_sph[7];
        RealType Q22s = mp_sph[8];
        RealType inv_sqrt3 = 1.0 / Kokkos::sqrt(3.0);

        cart[hermite_index(0, 0, 2)] = 2.0 * Q20 / 3.0;
        cart[hermite_index(2, 0, 0)] = -Q20 / 3.0 + Q22c * inv_sqrt3;
        cart[hermite_index(0, 2, 0)] = -Q20 / 3.0 - Q22c * inv_sqrt3;
        cart[hermite_index(1, 0, 1)] = Q21c * inv_sqrt3;
        cart[hermite_index(0, 1, 1)] = Q21s * inv_sqrt3;
        cart[hermite_index(1, 1, 0)] = Q22s * inv_sqrt3;
    }

    if (max_rank >= 3 && n_components >= 16) {
        RealType Q30 = mp_sph[9];
        RealType Q31c = mp_sph[10];
        RealType Q31s = mp_sph[11];
        RealType Q32c = mp_sph[12];
        RealType Q32s = mp_sph[13];
        RealType Q33c = mp_sph[14];
        RealType Q33s = mp_sph[15];
        RealType sqrt6 = Kokkos::sqrt(6.0);
        RealType sqrt10 = Kokkos::sqrt(10.0);
        RealType sqrt15 = Kokkos::sqrt(15.0);
        RealType c31c = Q31c / sqrt6;
        RealType c31s = Q31s / sqrt6;

        cart[hermite_index(0, 0, 3)] = 2.0 * Q30 / 5.0;
        cart[hermite_index(2, 0, 1)] = -Q30 / 5.0 + Q32c / sqrt15;
        cart[hermite_index(0, 2, 1)] = -Q30 / 5.0 - Q32c / sqrt15;
        cart[hermite_index(1, 0, 2)] = 4.0 * c31c / 5.0;
        cart[hermite_index(0, 1, 2)] = 4.0 * c31s / 5.0;
        cart[hermite_index(1, 1, 1)] = Q32s / sqrt15;
        cart[hermite_index(3, 0, 0)] = -3.0 * c31c / 5.0 + Q33c / sqrt10;
        cart[hermite_index(1, 2, 0)] = -c31c / 5.0 - Q33c / sqrt10;
        cart[hermite_index(2, 1, 0)] = -c31s / 5.0 + Q33s / sqrt10;
        cart[hermite_index(0, 3, 0)] = -3.0 * c31s / 5.0 - Q33s / sqrt10;
    }

    if (max_rank >= 4 && n_components >= 25) {
        RealType Q40 = mp_sph[16];
        RealType Q41c = mp_sph[17];
        RealType Q41s = mp_sph[18];
        RealType Q42c = mp_sph[19];
        RealType Q42s = mp_sph[20];
        RealType Q43c = mp_sph[21];
        RealType Q43s = mp_sph[22];
        RealType Q44c = mp_sph[23];
        RealType Q44s = mp_sph[24];
        RealType sqrt5 = Kokkos::sqrt(5.0);
        RealType sqrt10 = Kokkos::sqrt(10.0);
        RealType sqrt35 = Kokkos::sqrt(35.0);
        RealType sqrt70 = Kokkos::sqrt(70.0);
        RealType inv_7sqrt5 = 1.0 / (7.0 * sqrt5);
        RealType inv_7sqrt10 = 1.0 / (7.0 * sqrt10);

        cart[hermite_index(0, 0, 4)] = 8.0 * Q40 / 35.0;
        cart[hermite_index(4, 0, 0)] = 3.0 * Q40 / 35.0 + Q44c / sqrt35 - 2.0 * Q42c * inv_7sqrt5;
        cart[hermite_index(0, 4, 0)] = 3.0 * Q40 / 35.0 + Q44c / sqrt35 + 2.0 * Q42c * inv_7sqrt5;

        cart[hermite_index(2, 2, 0)] = Q40 / 35.0 - Q44c / sqrt35;
        cart[hermite_index(2, 0, 2)] = -4.0 * Q40 / 35.0 + 2.0 * Q42c * inv_7sqrt5;
        cart[hermite_index(0, 2, 2)] = -4.0 * Q40 / 35.0 - 2.0 * Q42c * inv_7sqrt5;

        cart[hermite_index(3, 1, 0)] = Q44s / sqrt35 - Q42s * inv_7sqrt5;
        cart[hermite_index(1, 3, 0)] = -Q42s * inv_7sqrt5 - Q44s / sqrt35;
        cart[hermite_index(1, 1, 2)] = 2.0 * Q42s * inv_7sqrt5;

        cart[hermite_index(3, 0, 1)] = -3.0 * Q41c * inv_7sqrt10 + Q43c / sqrt70;
        cart[hermite_index(1, 2, 1)] = -Q41c * inv_7sqrt10 - Q43c / sqrt70;
        cart[hermite_index(1, 0, 3)] = 4.0 * Q41c * inv_7sqrt10;

        cart[hermite_index(2, 1, 1)] = Q43s / sqrt70 - Q41s * inv_7sqrt10;
        cart[hermite_index(0, 3, 1)] = -3.0 * Q41s * inv_7sqrt10 - Q43s / sqrt70;
        cart[hermite_index(0, 1, 3)] = 4.0 * Q41s * inv_7sqrt10;
    }
}

// ---------------------------------------------------------------------------
// Cartesian multipole rotation (matches occ rotate_cartesian_multipole)
// ---------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
void rotate_cartesian_multipole(const RealType* body,
                                const RealType R[3][3], int max_rank,
                                RealType* lab) {
    int N = nhermsum(max_rank);
    constexpr double fact_arr[] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320};

    // Rank 0 is invariant
    lab[0] = body[0];
    for (int i = 1; i < N; ++i) lab[i] = 0.0;

    // For each rank l from 1 to max_rank
    for (int l = 1; l <= max_rank; ++l) {
        // Iterate over target (t,u,v) with t+u+v = l
        for (int t = l; t >= 0; --t) {
            for (int uv = l - t; uv >= 0; --uv) {
                int u = l - t - uv;
                int v = uv;
                int idx_out = hermite_index(t, u, v);

                RealType val = 0.0;

                // Iterate over source (a,b,c) with a+b+c = l
                for (int a = l; a >= 0; --a) {
                    for (int bc = l - a; bc >= 0; --bc) {
                        int b = l - a - bc;
                        int c = bc;
                        int idx_in = hermite_index(a, b, c);

                        RealType body_val = body[idx_in];
                        if (body_val == 0.0) continue;

                        // Sum over transportation matrices P (3x3 nonneg int)
                        // Row sums = (t,u,v), col sums = (a,b,c)
                        RealType coeff = 0.0;
                        double ft = fact_arr[t];
                        double fu = fact_arr[u];
                        double fv = fact_arr[v];

                        for (int pxx = 0; pxx <= (t < a ? t : a); ++pxx) {
                            for (int pxy = 0; pxy <= ((t-pxx) < b ? (t-pxx) : b); ++pxy) {
                                int pxz = t - pxx - pxy;
                                if (pxz > c || pxz < 0) continue;

                                for (int pyx = 0; pyx <= (u < (a-pxx) ? u : (a-pxx)); ++pyx) {
                                    int pzx = a - pxx - pyx;
                                    if (pzx > v || pzx < 0) continue;

                                    for (int pyy = 0; pyy <= ((u-pyx) < (b-pxy) ? (u-pyx) : (b-pxy)); ++pyy) {
                                        int pyz = u - pyx - pyy;
                                        int pzy = b - pxy - pyy;
                                        int pzz = v - pzx - pzy;
                                        if (pyz < 0 || pzy < 0 || pzz < 0) continue;

                                        double mc = (ft / (fact_arr[pxx] * fact_arr[pxy] * fact_arr[pxz]))
                                                  * (fu / (fact_arr[pyx] * fact_arr[pyy] * fact_arr[pyz]))
                                                  * (fv / (fact_arr[pzx] * fact_arr[pzy] * fact_arr[pzz]));

                                        RealType mp = 1.0;
                                        // R[row][col]^power
                                        for (int p = 0; p < pxx; ++p) mp *= R[0][0];
                                        for (int p = 0; p < pxy; ++p) mp *= R[0][1];
                                        for (int p = 0; p < pxz; ++p) mp *= R[0][2];
                                        for (int p = 0; p < pyx; ++p) mp *= R[1][0];
                                        for (int p = 0; p < pyy; ++p) mp *= R[1][1];
                                        for (int p = 0; p < pyz; ++p) mp *= R[1][2];
                                        for (int p = 0; p < pzx; ++p) mp *= R[2][0];
                                        for (int p = 0; p < pzy; ++p) mp *= R[2][1];
                                        for (int p = 0; p < pzz; ++p) mp *= R[2][2];

                                        coeff += mc * mp;
                                    }
                                }
                            }
                        }

                        val += coeff * body_val;
                    }
                }

                lab[idx_out] = val;
            }
        }
    }
}

// Rotate a dual multipole array (dE/d(cart[c])) by the same rotation.
//
// Component (t,u,v) stands for l!/(t!u!v!) tensor entries. With
// G = diag(l!/(t!u!v!)) and M the rotation of rotate_cartesian_multipole, a
// gradient transforms by the adjoint,
//
//     g' = M^-T g = G M G^-1 g,
//
// using M^T G M = G. This differs from applying M directly unless M is a signed
// permutation or rank <= 1.
KOKKOS_INLINE_FUNCTION
void rotate_cartesian_multipole_dual(const RealType* g_body,
                                     const RealType R[3][3], int max_rank,
                                     RealType* g_lab) {
    constexpr MultipoleConstants<4> mc{};
    constexpr double fact_arr[] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320};
    const int N = nhermsum(max_rank);

    RealType scaled[nhermsum(4)];
    for (int c = 0; c < N; ++c)                       // G^-1
        scaled[c] = g_body[c] / static_cast<RealType>(fact_arr[mc.rank_table[c]]
                                                      * mc.inv_fact[c]);
    rotate_cartesian_multipole(scaled, R, max_rank, g_lab);
    for (int c = 0; c < N; ++c)                       // G
        g_lab[c] *= static_cast<RealType>(fact_arr[mc.rank_table[c]] * mc.inv_fact[c]);
}

// ---------------------------------------------------------------------------
// The same derivative, from the rotated multipoles and a generator.
//
// Differentiating Q' = M...M Q gives one term per tensor slot, each
// (dM M^-1) acting on one index of Q' itself:
//
//     dQ'_{i1..il} = sum_k sum_j Omega_{ik j} Q'_{i1..j..il},   Omega = dM M^-1.
//
// On unique components a slot with t x's contributes t * Omega_{xj} times the
// component with one x replaced by j. M^-1 (not M^T) and the diagonal of Omega
// are kept so this holds for a non-orthogonal M.
// ---------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
void rotate_cartesian_multipole_generator(const RealType* lab,
                                          const RealType Omega[3][3], int max_rank,
                                          RealType* dlab) {
    const int N = nhermsum(max_rank);
    dlab[0] = 0.0;          // rank 0 is invariant
    for (int i = 1; i < N; ++i) dlab[i] = 0.0;
    for (int l = 1; l <= max_rank; ++l) {
        for (int t = l; t >= 0; --t) {
            for (int uv = l - t; uv >= 0; --uv) {
                const int u = l - t - uv;
                const int v = uv;
                const RealType self = lab[hermite_index(t, u, v)];
                RealType val = RealType(t) * Omega[0][0] * self
                    + RealType(u) * Omega[1][1] * self
                    + RealType(v) * Omega[2][2] * self;
                if (t > 0) {
                    val += RealType(t) * (Omega[0][1] * lab[hermite_index(t-1, u+1, v)]
                                        + Omega[0][2] * lab[hermite_index(t-1, u, v+1)]);
                }
                if (u > 0) {
                    val += RealType(u) * (Omega[1][0] * lab[hermite_index(t+1, u-1, v)]
                                        + Omega[1][2] * lab[hermite_index(t, u-1, v+1)]);
                }
                if (v > 0) {
                    val += RealType(v) * (Omega[2][0] * lab[hermite_index(t+1, u, v-1)]
                                        + Omega[2][1] * lab[hermite_index(t, u+1, v-1)]);
                }
                dlab[hermite_index(t, u, v)] = val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Derivative of Cartesian multipole rotation w.r.t. rotation matrix direction
// ---------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
void rotate_cartesian_multipole_deriv(const RealType* body,
                                      const RealType R[3][3],
                                      const RealType dR[3][3],
                                      int max_rank,
                                      RealType* dlab) {
    int N = nhermsum(max_rank);
    constexpr double fact_arr[] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320};
    for (int i = 0; i < N; ++i) dlab[i] = 0.0;

    // Rank 0: invariant, derivative = 0
    // For each rank l from 1 to max_rank
    for (int l = 1; l <= max_rank; ++l) {
        for (int t = l; t >= 0; --t) {
            for (int uv = l - t; uv >= 0; --uv) {
                int u = l - t - uv;
                int v = uv;
                int idx_out = hermite_index(t, u, v);

                RealType val = 0.0;

                for (int a = l; a >= 0; --a) {
                    for (int bc = l - a; bc >= 0; --bc) {
                        int b = l - a - bc;
                        int c = bc;
                        int idx_in = hermite_index(a, b, c);

                        RealType body_val = body[idx_in];
                        if (body_val == 0.0) continue;

                        // Product rule: d/d(param) of product of R^P terms
                        // = sum over each (i,j) factor, replace one R[i][j]^p with p*R[i][j]^(p-1)*dR[i][j]
                        RealType dcoeff = 0.0;
                        double ft = fact_arr[t];
                        double fu = fact_arr[u];
                        double fv = fact_arr[v];

                        for (int pxx = 0; pxx <= (t < a ? t : a); ++pxx) {
                            for (int pxy = 0; pxy <= ((t-pxx) < b ? (t-pxx) : b); ++pxy) {
                                int pxz = t - pxx - pxy;
                                if (pxz > c || pxz < 0) continue;

                                for (int pyx = 0; pyx <= (u < (a-pxx) ? u : (a-pxx)); ++pyx) {
                                    int pzx = a - pxx - pyx;
                                    if (pzx > v || pzx < 0) continue;

                                    for (int pyy = 0; pyy <= ((u-pyx) < (b-pxy) ? (u-pyx) : (b-pxy)); ++pyy) {
                                        int pyz = u - pyx - pyy;
                                        int pzy = b - pxy - pyy;
                                        int pzz = v - pzx - pzy;
                                        if (pyz < 0 || pzy < 0 || pzz < 0) continue;

                                        double mc = (ft / (fact_arr[pxx] * fact_arr[pxy] * fact_arr[pxz]))
                                                  * (fu / (fact_arr[pyx] * fact_arr[pyy] * fact_arr[pyz]))
                                                  * (fv / (fact_arr[pzx] * fact_arr[pzy] * fact_arr[pzz]));

                                        // The full monomial: prod R[i][j]^P[i][j]
                                        int P[3][3] = {{pxx,pxy,pxz},{pyx,pyy,pyz},{pzx,pzy,pzz}};

                                        // Product rule: for each (i,j) with P[i][j] > 0,
                                        // swap one R for its derivative. Rebuilt rather
                                        // than divided because R can be zero.
                                        //
                                        RealType dterm = 0.0;
                                        for (int ii = 0; ii < 3; ++ii) {
                                            for (int jj = 0; jj < 3; ++jj) {
                                                if (P[ii][jj] == 0) continue;
                                                // Replace one R[ii][jj] with dR[ii][jj]
                                                // = P[ii][jj] * dR[ii][jj] * R[ii][jj]^(P[ii][jj]-1) * rest
                                                RealType sub_prod = 1.0;
                                                for (int i2 = 0; i2 < 3; ++i2)
                                                    for (int j2 = 0; j2 < 3; ++j2) {
                                                        int pw = P[i2][j2];
                                                        if (i2 == ii && j2 == jj) pw -= 1;
                                                        for (int p = 0; p < pw; ++p)
                                                            sub_prod *= R[i2][j2];
                                                    }
                                                dterm += P[ii][jj] * dR[ii][jj] * sub_prod;
                                            }
                                        }

                                        dcoeff += mc * dterm;
                                    }
                                }
                            }
                        }

                        val += dcoeff * body_val;
                    }
                }

                dlab[idx_out] = val;
            }
        }
    }
}

}  // namespace klasp
