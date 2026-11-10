#pragma once
// Detecting slabs of vacuum: cells packed into part of the cell with an empty
// layer across the rest. These are stationary points past the potential cutoff,
// and density does not identify them.
//
// A vacuum slab in a periodic crystal must lie parallel to a lattice plane, so
// scanning low-index (hkl) and projecting onto each plane normal reduces the
// question to the largest gap in a circular 1-D arrangement.

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace klasp {

struct VacuumSlab {
    double gap = 0.0;        // largest centre-to-centre gap, Angstrom
    int h = 0, k = 0, l = 0; // the plane it was found on
};

// |G_hkl| from the metric tensor. Lengths in any unit; the returned spacing is
// in that unit. Returns 0 for a degenerate cell.
inline double interplanar_spacing(const double cp[6], int h, int k, int l) {
    constexpr double DEG2RAD = 3.14159265358979323846 / 180.0;
    const double a = cp[0], b = cp[1], c = cp[2];
    if (a <= 0 || b <= 0 || c <= 0) return 0.0;
    const double al = cp[3]*DEG2RAD, be = cp[4]*DEG2RAD, ga = cp[5]*DEG2RAD;
    const double ca = std::cos(al), cb = std::cos(be), cg = std::cos(ga);
    const double sa = std::sin(al), sb = std::sin(be), sg = std::sin(ga);
    const double v2 = 1.0 - ca*ca - cb*cb - cg*cg + 2.0*ca*cb*cg;
    if (v2 <= 1e-12) return 0.0;
    const double inv_d2 = ((h*h)/(a*a)*sa*sa + (k*k)/(b*b)*sb*sb + (l*l)/(c*c)*sg*sg
                          + 2.0*h*k/(a*b)*(ca*cb - cg)
                          + 2.0*k*l/(b*c)*(cb*cg - ca)
                          + 2.0*h*l/(a*c)*(ca*cg - cb)) / v2;
    return inv_d2 > 0.0 ? 1.0 / std::sqrt(inv_d2) : 0.0;
}

// The (hkl) scanned: coprime triples with |index| <= 2, one per family, since
// (hkl) and (-h-k-l) describe the same planes.
inline const std::vector<std::array<int, 3>>& vacuum_planes() {
    static const std::vector<std::array<int, 3>> planes = [] {
        auto gcd = [](int x, int y) { while (y) { int t = x % y; x = y; y = t; } return x < 0 ? -x : x; };
        std::vector<std::array<int, 3>> p;
        for (int h = -2; h <= 2; ++h)
            for (int k = -2; k <= 2; ++k)
                for (int l = 0; l <= 2; ++l) {
                    if (!h && !k && !l) continue;
                    if (gcd(gcd(h, k), l) != 1) continue;
                    // First non-zero index positive: one representative per family.
                    if (l == 0 && (h < 0 || (h == 0 && k < 0))) continue;
                    p.push_back({h, k, l});
                }
        return p;
    }();
    return planes;
}

// Largest vacuum gap over the scanned planes. `frac` is 3*n_atoms fractional
// coordinates of every atom in the whole cell, not the asymmetric unit.
// Projection onto the (hkl) normal is h*x + k*y + l*z wrapped into [0,1) and
// scaled by the interplanar spacing.
inline VacuumSlab max_vacuum_gap(const double cell_params[6],
                                 const double* frac, int n_atoms) {
    VacuumSlab best;
    if (n_atoms < 2 || !frac) return best;

    std::vector<double> t(static_cast<std::size_t>(n_atoms));
    for (const auto& p : vacuum_planes()) {
        const double d = interplanar_spacing(cell_params, p[0], p[1], p[2]);
        if (d <= 0.0) continue;
        for (int i = 0; i < n_atoms; ++i) {
            double u = p[0]*frac[3*i] + p[1]*frac[3*i+1] + p[2]*frac[3*i+2];
            u -= std::floor(u);
            t[static_cast<std::size_t>(i)] = u;
        }
        std::sort(t.begin(), t.end());
        double widest = t.front() + 1.0 - t.back();   // the gap across the wrap
        for (int i = 0; i + 1 < n_atoms; ++i)
            widest = std::max(widest, t[i + 1] - t[i]);
        const double gap = widest * d;
        if (gap > best.gap) { best.gap = gap; best.h = p[0]; best.k = p[1]; best.l = p[2]; }
    }
    return best;
}

}  // namespace klasp
