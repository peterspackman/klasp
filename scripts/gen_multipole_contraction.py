#!/usr/bin/env python3
"""Generate include/klasp/multipole_contraction.hpp: the straight-line T-tensor
contraction for each multipole order.

multipole_pair_energy_forces_mp_grad used to loop over all 35 x 35 component
pairs, testing an `active` mask and reading four index tables per pair. The
indices are compile-time constants, so the loop is written out instead: no
tables, no branches, and the T tensor can live in registers rather than in
per-thread local memory (which is where a runtime-indexed array goes on a GPU).

The arithmetic is also arranged to do less. With wA_i = s_i a_i and
wB_j = f_j b_j (s_i = (-1)^l/(t!u!v!), f_j = 1/(t!u!v!)):

    E        = sum_ij wA_i wB_j T[add_ij]
    force    = sum_ij wA_i wB_j T[shift_ij]
    dE/da_i  = s_i * sum_j T[add_ij] wB_j
    dE/db_j  = f_j * sum_i T[add_ij] wA_i

so the weights are applied once per component, not once per pair, and the
per-pair work is one product wA_i wB_j and five multiply-adds.

Run from the repository root:  python3 scripts/gen_multipole_contraction.py
"""
from fractions import Fraction
from math import factorial
from pathlib import Path


def nhermsum(L):
    return (L + 1) * (L + 2) * (L + 3) // 6


def hermite_index(t, u, v):
    L = t + u + v
    offset = nhermsum(L - 1) if L > 0 else 0
    uv = u + v
    return offset + uv * (uv + 1) // 2 + v


def tuv_table(max_l=4):
    n = nhermsum(max_l)
    out = [None] * n
    for L in range(max_l + 1):
        for t in range(L, -1, -1):
            for uv in range(L - t, -1, -1):
                u, v = L - t - uv, uv
                out[hermite_index(t, u, v)] = (t, u, v)
    return out


TUV = tuv_table(4)
N = len(TUV)  # 35
TUV_ALL = tuv_table(6)


def lit(x: Fraction) -> str:
    """A double literal with the full precision of the fraction."""
    return repr(float(x))


def gen_order(order, need_b):
    lines = []
    w = lines.append
    nT = nhermsum(order + 1)
    active = [(i, j) for i in range(N) for j in range(N)
              if sum(TUV[i]) + sum(TUV[j]) <= order]
    rows = sorted({i for i, _ in active})
    cols = sorted({j for _, j in active})
    w(f"template <>")
    w(f"struct MultipoleContraction<{order}, {'true' if need_b else 'false'}> {{")
    w(f"    // {len(active)} active component pairs, T tensor of {nT} entries.")
    w(f"    KOKKOS_INLINE_FUNCTION")
    w(f"    static RealType run(const RealType* a, const RealType* b, const RealType* T,")
    w(f"                        RealType force[3], RealType* dEa, RealType* dEb) {{")
    if not need_b:
        w(f"        (void)dEb;")
    # weighted multipoles, hoisted
    for j in cols:
        t, u, v = TUV[j]
        f = Fraction(1, factorial(t) * factorial(u) * factorial(v))
        w(f"        const RealType wb{j} = {lit(f)} * b[{j}];")
    for i in rows:
        t, u, v = TUV[i]
        s = Fraction((-1) ** (t + u + v), factorial(t) * factorial(u) * factorial(v))
        w(f"        const RealType wa{i} = {lit(s)} * a[{i}];")
    w(f"        RealType E = 0, fx = 0, fy = 0, fz = 0;")
    if need_b:
        w("        " + " ".join(f"RealType db{j} = 0;" for j in cols))
    for i in range(N):
        js = [j for ii, j in active if ii == i]
        if not js:
            w(f"        dEa[{i}] = 0;")
            continue
        t, u, v = TUV[i]
        s = Fraction((-1) ** (t + u + v), factorial(t) * factorial(u) * factorial(v))
        w(f"        {{")
        w(f"            RealType acc = 0;")
        for j in js:
            ta, ua, va = TUV[i]
            tb, ub, vb = TUV[j]
            add = hermite_index(ta + tb, ua + ub, va + vb)
            gx = hermite_index(ta + tb + 1, ua + ub, va + vb)
            gy = hermite_index(ta + tb, ua + ub + 1, va + vb)
            gz = hermite_index(ta + tb, ua + ub, va + vb + 1)
            w(f"            {{ const RealType w = wa{i} * wb{j}; const RealType t = T[{add}];")
            w(f"              E += w * t; fx += w * T[{gx}]; fy += w * T[{gy}]; fz += w * T[{gz}];")
            if need_b:
                w(f"              acc += t * wb{j}; db{j} += t * wa{i}; }}")
            else:
                w(f"              acc += t * wb{j}; }}")
        w(f"            dEa[{i}] = {lit(s)} * acc;")
        w(f"        }}")
    if need_b:
        for j in range(N):
            if j in cols:
                t, u, v = TUV[j]
                f = Fraction(1, factorial(t) * factorial(u) * factorial(v))
                w(f"        dEb[{j}] = {lit(f)} * db{j};")
            else:
                w(f"        dEb[{j}] = 0;")
    w(f"        force[0] = fx; force[1] = fy; force[2] = fz;")
    w(f"        return E;")
    w(f"    }}")
    w(f"}};")
    w("")
    return lines


def poly_mul(p, q):
    """Multiply two polynomials {(xpow, ypow, zpow, Rmonomial): coeff}."""
    out = {}
    for (xa, ya, za, ma), ca in p.items():
        for (xb, yb, zb, mb), cb in q.items():
            key = (xa + xb, ya + yb, za + zb, tuple(sorted(ma + mb)))
            out[key] = out.get(key, 0) + ca * cb
    return out


def poly_pow(p, n):
    out = {(0, 0, 0, ()): 1}
    for _ in range(n):
        out = poly_mul(out, p)
    return out


def gen_rotation(rank):
    """lab_{tuv} = sum_{abc} M_{tuv,abc}(R) body_{abc}, M the coefficient of
    x^a y^b z^c in (R_x . xyz)^t (R_y . xyz)^u (R_z . xyz)^v -- the tensor
    rotation Q' = R (x) ... (x) R Q on unique components, exactly what
    rotate_cartesian_multipole computes by enumerating transport matrices.
    Every monomial in R is a product of the 45 pairwise products p_ab = R_a R_b
    (degree 4: p * p; 3: p * R; 2: p), which the compiler shares across entries.
    """
    lines = []
    w = lines.append
    # linear forms: row i of R dotted with (x, y, z); R entries indexed 0..8
    rows = [{(1, 0, 0, (3 * i + 0,)): 1, (0, 1, 0, (3 * i + 1,)): 1, (0, 0, 1, (3 * i + 2,)): 1}
            for i in range(3)]
    w("template <>")
    w(f"KOKKOS_INLINE_FUNCTION")
    w(f"void rotate_cartesian_multipole_gen<{rank}>(const RealType* body, const RealType R[3][3],")
    w(f"                                            RealType* lab) {{")
    w("    lab[0] = body[0];")
    if rank >= 2:
        w("    // pairwise products of the entries of R, shared by every monomial below")
        for a in range(9):
            for b in range(a, 9):
                w(f"    const RealType p{a}{b} = R[{a // 3}][{a % 3}] * R[{b // 3}][{b % 3}];")

    def mono(m):
        """Code for the product of R entries listed (sorted) in m."""
        if len(m) == 0:
            return "1.0"
        if len(m) == 1:
            return f"R[{m[0] // 3}][{m[0] % 3}]"
        if len(m) == 2:
            return f"p{m[0]}{m[1]}"
        if len(m) == 3:
            return f"p{m[0]}{m[1]} * R[{m[2] // 3}][{m[2] % 3}]"
        return f"p{m[0]}{m[1]} * p{m[2]}{m[3]}"

    for L in range(1, rank + 1):
        for t in range(L, -1, -1):
            for uv in range(L - t, -1, -1):
                u, v = L - t - uv, uv
                poly = poly_mul(poly_mul(poly_pow(rows[0], t), poly_pow(rows[1], u)),
                                poly_pow(rows[2], v))
                # group by input component (a, b, c)
                by_in = {}
                for (a, b, c, m), coef in poly.items():
                    by_in.setdefault((a, b, c), []).append((coef, m))
                terms = []
                for (a, b, c), ms in sorted(by_in.items()):
                    parts = " + ".join(f"{coef}.0 * {mono(m)}" if coef != 1 else mono(m)
                                       for coef, m in sorted(ms, key=lambda cm: cm[1]))
                    terms.append(f"({parts}) * body[{hermite_index(a, b, c)}]")
                w(f"    lab[{hermite_index(t, u, v)}] = " + "\n        + ".join(terms) + ";")
    w("}")
    w("")
    return lines


def gen_t_tensor(order):
    """The T tensor to `order` by the Obara-Saika recurrence, straight-line, in
    any scalar. Mirrors compute_t_tensor's generic loops statement for statement,
    naming the auxiliaries R{m}_{t}{u}{v}; the ones no later line reads are dead
    code the compiler drops."""
    lines = []
    w = lines.append
    nsz = nhermsum(order)
    w("template <typename Real>")
    w(f"struct TTensor<{order}, Real> {{")
    w("    KOKKOS_INLINE_FUNCTION")
    w("    static void run(Real Rx, Real Ry, Real Rz, Real invR2, Real invR, Real* T) {")
    w("        const Real R0_000 = invR;")
    for m in range(1, order + 1):
        w(f"        const Real R{m}_000 = Real(-{2 * m - 1}) * invR2 * R{m - 1}_000;")
    for t in range(order):
        for m in range(order - t):
            term = f"Rx * R{m + 1}_{t}00"
            if t > 0:
                term += f" + Real({t}) * R{m + 1}_{t - 1}00"
            w(f"        const Real R{m}_{t + 1}00 = {term};")
    for t in range(order + 1):
        for u in range(order - t):
            for m in range(order - t - u):
                term = f"Ry * R{m + 1}_{t}{u}0"
                if u > 0:
                    term += f" + Real({u}) * R{m + 1}_{t}{u - 1}0"
                w(f"        const Real R{m}_{t}{u + 1}0 = {term};")
    for t in range(order + 1):
        for u in range(order - t + 1):
            for v in range(order - t - u):
                for m in range(order - t - u - v):
                    term = f"Rz * R{m + 1}_{t}{u}{v}"
                    if v > 0:
                        term += f" + Real({v}) * R{m + 1}_{t}{u}{v - 1}"
                    w(f"        const Real R{m}_{t}{u}{v + 1} = {term};")
    for i in range(nsz):
        t, u, v = TUV_ALL[i]
        w(f"        T[{i}] = R0_{t}{u}{v};")
    w("    }")
    w("};")
    w("")
    return lines


def gen_higher(order, need_b):
    """The contraction without the (charge, dipole) x (charge, dipole) pairs:
    the rank >= 2 multipole energy, less the bare charge-charge, charge-dipole
    and dipole-dipole terms those pairs of the full contraction are. Those
    Ewald handles. Subtracting them after the fact --
    what pair_multipole_higher did -- cancels the largest term in the sum
    against itself, which single precision cannot afford and double precision
    did not need. Scalar-templated so the same code runs in float."""
    lines = []
    w = lines.append
    nT = nhermsum(order + 1)
    # Everything the mask keeps except the pairs Ewald owns: both sites of rank
    # <= 1 (charge-charge, charge-dipole, dipole-dipole). Not "la + lb <= 1":
    # charge-quadrupole belongs here.
    active = [(i, j) for i in range(N) for j in range(N)
              if sum(TUV[i]) + sum(TUV[j]) <= order
              and not (sum(TUV[i]) <= 1 and sum(TUV[j]) <= 1)]
    rows = sorted({i for i, _ in active})
    cols = sorted({j for _, j in active})
    w("template <typename Real>")
    w(f"struct MultipoleHigherContraction<{order}, {'true' if need_b else 'false'}, Real> {{")
    w(f"    // {len(active)} active component pairs, T tensor of {nT} entries.")
    w("    KOKKOS_INLINE_FUNCTION")
    w("    static Real run(const RealType* a, const RealType* b, const Real* T,")
    w("                    Real force[3], Real* dEa, Real* dEb) {")
    if not need_b:
        w("        (void)dEb;")
    if not active:
        w("        (void)a; (void)b; (void)T;")
        w("        force[0] = force[1] = force[2] = Real(0);")
        w(f"        for (int c = 0; c < {N}; ++c) {{ dEa[c] = Real(0);"
          + (" dEb[c] = Real(0);" if need_b else "") + " }")
        w("        return Real(0);")
        w("    }")
        w("};")
        w("")
        return lines
    for j in cols:
        t, u, v = TUV[j]
        f = Fraction(1, factorial(t) * factorial(u) * factorial(v))
        w(f"        const Real wb{j} = Real({lit(f)}) * Real(b[{j}]);")
    for i in rows:
        t, u, v = TUV[i]
        s = Fraction((-1) ** (t + u + v), factorial(t) * factorial(u) * factorial(v))
        w(f"        const Real wa{i} = Real({lit(s)}) * Real(a[{i}]);")
    w("        Real E = Real(0), fx = Real(0), fy = Real(0), fz = Real(0);")
    if need_b:
        w("        " + " ".join(f"Real db{j} = Real(0);" for j in cols))
    for i in range(N):
        js = [j for ii, j in active if ii == i]
        if not js:
            w(f"        dEa[{i}] = Real(0);")
            continue
        t, u, v = TUV[i]
        s = Fraction((-1) ** (t + u + v), factorial(t) * factorial(u) * factorial(v))
        w("        {")
        w("            Real acc = Real(0);")
        for j in js:
            ta, ua, va = TUV[i]
            tb, ub, vb = TUV[j]
            add = hermite_index(ta + tb, ua + ub, va + vb)
            gx = hermite_index(ta + tb + 1, ua + ub, va + vb)
            gy = hermite_index(ta + tb, ua + ub + 1, va + vb)
            gz = hermite_index(ta + tb, ua + ub, va + vb + 1)
            w(f"            {{ const Real w = wa{i} * wb{j}; const Real t = T[{add}];")
            w(f"              E += w * t; fx += w * T[{gx}]; fy += w * T[{gy}]; fz += w * T[{gz}];")
            if need_b:
                w(f"              acc += t * wb{j}; db{j} += t * wa{i}; }}")
            else:
                w(f"              acc += t * wb{j}; }}")
        w(f"            dEa[{i}] = Real({lit(s)}) * acc;")
        w("        }")
    if need_b:
        for j in range(N):
            if j in cols:
                t, u, v = TUV[j]
                f = Fraction(1, factorial(t) * factorial(u) * factorial(v))
                w(f"        dEb[{j}] = Real({lit(f)}) * db{j};")
            else:
                w(f"        dEb[{j}] = Real(0);")
    w("        force[0] = fx; force[1] = fy; force[2] = fz;")
    w("        return E;")
    w("    }")
    w("};")
    w("")
    return lines


def gen_pair_moments(order):
    """The pair-moment builders for `order`.

    E = sum_ij wA_i wB_j T[add_ij] over the higher pairs is sum_k s_k T[k] with
        s_k = sum_{add_ij = k} wA_i wB_j,
    a vector over the T-tensor entries of total order <= `order` that depends
    on the two sites' multipoles alone. Every lattice image of a site pair
    shares it, so it is built once per site pair per iteration and each image
    is a dot product with T (energy) and with the shifted T (force).

    The rotational gradient folds the same way. With dE/da_c = s_c sum_d
    T[add_cd] wB_d and the atom's rotation Jacobian J[m][c] = d(lab_c)/dtheta_m,
        g_m += sum_c J[m][c] dE/da_c = sum_k T[k] v_{m,k},
        v_{m,k} = sum_{add_cd = k} (s_c J[m][c]) wB_d,
    so the per-image torque is three more dot products and the 35-component
    adjoint and its Jacobian contraction never exist. The partner's torque (only
    for its own identity copy) is w_{m,k} = sum_{add_cd = k} wA_c (f_d J_b[m][d]).
    """
    lines = []
    w = lines.append
    nT = nhermsum(order)
    active = [(i, j) for i in range(N) for j in range(N)
              if sum(TUV[i]) + sum(TUV[j]) <= order
              and not (sum(TUV[i]) <= 1 and sum(TUV[j]) <= 1)]
    rows = sorted({i for i, _ in active})
    cols = sorted({j for _, j in active})
    by_k = {}
    for i, j in active:
        ta, ua, va = TUV[i]
        tb, ub, vb = TUV[j]
        by_k.setdefault(hermite_index(ta + tb, ua + ub, va + vb), []).append((i, j))
    w("template <>")
    w(f"struct PairMoments<{order}> {{")
    w(f"    static constexpr int NT = {nT};   // T-tensor entries of total order <= {order}")
    w(f"    static constexpr bool empty = {'true' if not active else 'false'};")
    w("    // s and the three torque moments of the a side, from the two sites' lab")
    w("    // multipoles and a's rotation Jacobian (3 x 35, component-major).")
    w("    KOKKOS_INLINE_FUNCTION")
    w("    static void build(const RealType* a, const RealType* b, const RealType* J,")
    w("                      RealType* s, RealType* v) {")
    if not active:
        w("        (void)a; (void)b; (void)J; (void)s; (void)v;")
        w("    }")
    else:
        for j in cols:
            t, u, vv = TUV[j]
            f = Fraction(1, factorial(t) * factorial(u) * factorial(vv))
            w(f"        const RealType wb{j} = {lit(f)} * b[{j}];")
        for i in rows:
            t, u, vv = TUV[i]
            sg = Fraction((-1) ** (t + u + vv), factorial(t) * factorial(u) * factorial(vv))
            w(f"        const RealType wa{i} = {lit(sg)} * a[{i}];")
            w(f"        const RealType ja{i}_0 = {lit(sg)} * J[{i}], ja{i}_1 = {lit(sg)} * J[{35 + i}], ja{i}_2 = {lit(sg)} * J[{70 + i}];")
        for k in range(nT):
            pairs = by_k.get(k, [])
            if not pairs:
                w(f"        s[{k}] = 0; v[{k}] = 0; v[{nT + k}] = 0; v[{2 * nT + k}] = 0;")
                continue
            w(f"        s[{k}] = " + " + ".join(f"wa{i} * wb{j}" for i, j in pairs) + ";")
            for m in range(3):
                w(f"        v[{m * nT + k}] = " + " + ".join(f"ja{i}_{m} * wb{j}" for i, j in pairs) + ";")
        w("    }")
    w("    // The b side's torque moments, for a partner whose rotation matters.")
    w("    KOKKOS_INLINE_FUNCTION")
    w("    static void build_b(const RealType* a, const RealType* b, const RealType* Jb,")
    w("                        RealType* wm) {")
    if not active:
        w("        (void)a; (void)b; (void)Jb; (void)wm;")
        w("    }")
    else:
        w("        (void)b;")
        for i in rows:
            t, u, vv = TUV[i]
            sg = Fraction((-1) ** (t + u + vv), factorial(t) * factorial(u) * factorial(vv))
            w(f"        const RealType wa{i} = {lit(sg)} * a[{i}];")
        for j in cols:
            t, u, vv = TUV[j]
            f = Fraction(1, factorial(t) * factorial(u) * factorial(vv))
            w(f"        const RealType jb{j}_0 = {lit(f)} * Jb[{j}], jb{j}_1 = {lit(f)} * Jb[{35 + j}], jb{j}_2 = {lit(f)} * Jb[{70 + j}];")
        for k in range(nT):
            pairs = by_k.get(k, [])
            if not pairs:
                w(f"        wm[{k}] = 0; wm[{nT + k}] = 0; wm[{2 * nT + k}] = 0;")
                continue
            for m in range(3):
                w(f"        wm[{m * nT + k}] = " + " + ".join(f"wa{i} * jb{j}_{m}" for i, j in pairs) + ";")
        w("    }")
    # One entry at a time, so a team can split a chunk of pairs across threads.
    w("    // Entry k of the moments of one pair: s_k, the three v_{m,k}, and with a")
    w("    // partner Jacobian the three w_{m,k}, into a row laid out s | v_0 v_1 v_2 |")
    w("    // w_0 w_1 w_2, NT entries each. For a team building a chunk of pairs")
    w("    // over (pair, k) items; keep the items k-major so a warp's lanes share")
    w("    // the case.")
    w("    KOKKOS_INLINE_FUNCTION")
    w("    static void build_entry(const int k, const RealType* a, const RealType* b,")
    w("                            const RealType* J, const RealType* Jb, RealType* row) {")
    if not active:
        w("        (void)k; (void)a; (void)b; (void)J; (void)Jb; (void)row;")
        w("    }")
    else:
        w("        switch (k) {")
        for k in range(nT):
            pairs = by_k.get(k, [])
            if not pairs:
                continue
            w(f"        case {k}: {{")
            for i in sorted({i for i, _ in pairs}):
                t, u, vv = TUV[i]
                sg = Fraction((-1) ** (t + u + vv), factorial(t) * factorial(u) * factorial(vv))
                w(f"            const RealType wa{i} = {lit(sg)} * a[{i}];")
                w(f"            const RealType ja{i}_0 = {lit(sg)} * J[{i}], ja{i}_1 = {lit(sg)} * J[{35 + i}], ja{i}_2 = {lit(sg)} * J[{70 + i}];")
            for j in sorted({j for _, j in pairs}):
                t, u, vv = TUV[j]
                f = Fraction(1, factorial(t) * factorial(u) * factorial(vv))
                w(f"            const RealType wb{j} = {lit(f)} * b[{j}];")
            w(f"            row[{k}] = " + " + ".join(f"wa{i} * wb{j}" for i, j in pairs) + ";")
            for m in range(3):
                w(f"            row[{(m + 1) * nT + k}] = " + " + ".join(f"ja{i}_{m} * wb{j}" for i, j in pairs) + ";")
            w("            if (Jb) {")
            for j in sorted({j for _, j in pairs}):
                t, u, vv = TUV[j]
                f = Fraction(1, factorial(t) * factorial(u) * factorial(vv))
                w(f"                const RealType jb{j}_0 = {lit(f)} * Jb[{j}], jb{j}_1 = {lit(f)} * Jb[{35 + j}], jb{j}_2 = {lit(f)} * Jb[{70 + j}];")
            for m in range(3):
                w(f"                row[{(m + 4) * nT + k}] = " + " + ".join(f"wa{i} * jb{j}_{m}" for i, j in pairs) + ";")
            w("            }")
            w("            break; }")
        w("        default:")
        w(f"            row[k] = 0; row[{nT} + k] = 0; row[{2 * nT} + k] = 0; row[{3 * nT} + k] = 0;")
        w(f"            if (Jb) {{ row[{4 * nT} + k] = 0; row[{5 * nT} + k] = 0; row[{6 * nT} + k] = 0; }}")
        w("            break;")
        w("        }")
        w("    }")
    # per image: dot products with T (order <= `order` + 1 for the force shifts)
    ks = sorted(by_k)

    def shifted(k, d):
        t, u, vv = TUV_ALL[k]
        return hermite_index(t + (d == 0), u + (d == 1), vv + (d == 2))

    w("    // One image: the untapered energy, force on a and torque on a from the")
    w("    // moments and this image's T tensor (nhermsum(order + 1) entries).")
    w("    KOKKOS_INLINE_FUNCTION")
    w("    static RealType contract(const RealType* s, const RealType* v, const RealType* T,")
    w("                             RealType f[3], RealType tq[3]) {")
    if not ks:
        w("        (void)s; (void)v; (void)T;")
        w("        f[0] = f[1] = f[2] = 0; tq[0] = tq[1] = tq[2] = 0;")
        w("        return 0;")
    else:
        w("        RealType E = 0, fx = 0, fy = 0, fz = 0, t0 = 0, t1 = 0, t2 = 0;")
        for k in ks:
            w(f"        {{ const RealType sk = s[{k}], tk = T[{k}];")
            w(f"          E += sk * tk; fx += sk * T[{shifted(k,0)}]; fy += sk * T[{shifted(k,1)}]; fz += sk * T[{shifted(k,2)}];")
            w(f"          t0 += v[{k}] * tk; t1 += v[{nT + k}] * tk; t2 += v[{2 * nT + k}] * tk; }}")
        w("        f[0] = fx; f[1] = fy; f[2] = fz; tq[0] = t0; tq[1] = t1; tq[2] = t2;")
        w("        return E;")
    w("    }")
    w("    // The partner's torque for one image.")
    w("    KOKKOS_INLINE_FUNCTION")
    w("    static void contract_b(const RealType* wm, const RealType* T, RealType tq[3]) {")
    if not ks:
        w("        (void)wm; (void)T;")
        w("        tq[0] = tq[1] = tq[2] = 0;")
    else:
        w("        RealType t0 = 0, t1 = 0, t2 = 0;")
        for k in ks:
            w(f"        {{ const RealType tk = T[{k}]; t0 += wm[{k}] * tk; t1 += wm[{nT + k}] * tk; t2 += wm[{2 * nT + k}] * tk; }}")
        w("        tq[0] = t0; tq[1] = t1; tq[2] = t2;")
    w("    }")
    w("};")
    w("")
    return lines


def main():
    out = []
    w = out.append
    w("#pragma once")
    w("// GENERATED by scripts/gen_multipole_contraction.py -- do not edit by hand.")
    w("//")
    w("// The straight-line T-tensor contraction of multipole_pair_energy_forces_mp_grad,")
    w("// one specialisation per multipole order and per whether the partner's adjoint")
    w("// is wanted, and the straight-line Cartesian multipole rotation per rank. See")
    w("// the generator for the arithmetic and why it is written out.")
    w("")
    w('#include "klasp/types.hpp"')
    w("#include <Kokkos_Core.hpp>")
    w("")
    w("namespace klasp {")
    w("")
    w("// E = sum_ij wA_i wB_j T[add_ij] over the active pairs of order MaxOrder,")
    w("// with the force from the shifted T entries and dE/d(cart) for both sites.")
    w("// `a` and `b` are the 35 Cartesian components, `T` has nhermsum(MaxOrder + 1)")
    w("// entries, dEa and dEb are written in full (35 entries; NeedB false leaves dEb")
    w("// untouched).")
    w("template <int MaxOrder, bool NeedB>")
    w("struct MultipoleContraction;")
    w("")
    for order in range(5):
        for need_b in (True, False):
            out.extend(gen_order(order, need_b))
    w("// The T tensor to MaxOrder (nhermsum(MaxOrder) entries) from R and 1/R^2, 1/R,")
    w("// straight-line in any scalar: the same recurrence compute_t_tensor runs in")
    w("// loops over a local array.")
    w("template <int MaxOrder, typename Real>")
    w("struct TTensor;")
    w("")
    for order in range(1, 6):
        out.extend(gen_t_tensor(order))
    w("// The rank >= 2 part of the contraction alone -- see gen_higher.")
    w("template <int MaxOrder, bool NeedB, typename Real>")
    w("struct MultipoleHigherContraction;")
    w("")
    for order in range(5):
        for need_b in (True, False):
            out.extend(gen_higher(order, need_b))
    w("// Pair moments: the contraction's dependence on the two sites' multipoles,")
    w("// separated from its dependence on their separation -- see gen_pair_moments.")
    w("template <int MaxOrder>")
    w("struct PairMoments;")
    w("")
    for order in range(5):
        out.extend(gen_pair_moments(order))
    w("// lab = R (x) ... (x) R body on unique Cartesian components, ranks 0..MaxRank,")
    w("// as rotate_cartesian_multipole computes it by enumerating transport matrices --")
    w("// written out as the polynomial in R it is. Rank 0 is copied, ranks above")
    w("// MaxRank are not touched.")
    w("template <int MaxRank>")
    w("KOKKOS_INLINE_FUNCTION")
    w("void rotate_cartesian_multipole_gen(const RealType* body, const RealType R[3][3],")
    w("                                    RealType* lab);")
    w("")
    w("template <>")
    w("KOKKOS_INLINE_FUNCTION")
    w("void rotate_cartesian_multipole_gen<0>(const RealType* body, const RealType R[3][3],")
    w("                                       RealType* lab) {")
    w("    (void)R;")
    w("    lab[0] = body[0];")
    w("}")
    w("")
    for rank in range(1, 5):
        out.extend(gen_rotation(rank))
    w("}  // namespace klasp")
    path = Path(__file__).resolve().parent.parent / "include" / "klasp" / "multipole_contraction.hpp"
    path.write_text("\n".join(out) + "\n")
    print(f"wrote {path} ({len(out)} lines)")


if __name__ == "__main__":
    main()
