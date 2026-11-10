#pragma once
// Per-structure warning codes: why a stored structure should not be trusted.

#include <cmath>

namespace klasp {

// Persisted in the `warnings` column: append-only, never renumber.
enum StructureWarning : int {
    WARN_NONE            = 0,
    WARN_DEGENERATE_CELL = 1 << 0,  // lattice vectors nearly coplanar
    WARN_DENSITY         = 1 << 1,  // outside anything a molecular crystal reaches
    WARN_VACUUM_SLAB     = 1 << 2,  // an empty layer across the cell (see vacuum.hpp)
};

inline constexpr double WARN_SHAPE_FACTOR_MIN = 0.15;
inline constexpr double WARN_DENSITY_MIN_G_CM3 = 0.30;
inline constexpr double WARN_DENSITY_MAX_G_CM3 = 4.00;

// Largest gap between adjacent atomic planes, Angstrom, above which the cell
// holds a slab of vacuum rather than a packing.
inline constexpr double WARN_VACUUM_GAP_A = 6.0;

// The drop threshold, when derived, as a fraction of the potential cutoff:
// beyond the cutoff slabs no longer interact and every structure has the same
// energy.
inline constexpr double VACUUM_GAP_CUTOFF_FRACTION = 0.5;

// The threshold a run uses. `configured` is output.max_vacuum: negative derives
// it from `cutoff_angstrom` (the real-space cutoff), 0 keeps everything,
// positive is that many Angstrom.
inline double resolve_max_vacuum(double configured, double cutoff_angstrom) {
    if (configured >= 0.0) return configured;
    return VACUUM_GAP_CUTOFF_FRACTION * cutoff_angstrom;
}

// Volume over a*b*c: 1 for an orthogonal cell, 0 as the vectors become coplanar.
// Scale-free, so lengths may be in any unit; angles in degrees.
inline double cell_shape_factor(const double cell_params[6]) {
    constexpr double DEG2RAD = 3.14159265358979323846 / 180.0;
    const double ca = std::cos(cell_params[3] * DEG2RAD);
    const double cb = std::cos(cell_params[4] * DEG2RAD);
    const double cg = std::cos(cell_params[5] * DEG2RAD);
    const double arg = 1.0 - ca*ca - cb*cb - cg*cg + 2.0*ca*cb*cg;
    return arg > 0.0 ? std::sqrt(arg) : 0.0;
}

// density in g/cm^3, vacuum_gap in Angstrom; non-positive density or negative
// vacuum_gap means not computed.
inline int classify_warnings(const double cell_params[6], double density,
                             double vacuum_gap = -1.0) {
    int w = WARN_NONE;
    if (vacuum_gap > WARN_VACUUM_GAP_A) w |= WARN_VACUUM_SLAB;
    if (cell_shape_factor(cell_params) < WARN_SHAPE_FACTOR_MIN)
        w |= WARN_DEGENERATE_CELL;
    if (density > 0.0 &&
        (density < WARN_DENSITY_MIN_G_CM3 || density > WARN_DENSITY_MAX_G_CM3))
        w |= WARN_DENSITY;
    return w;
}

inline const char* warning_name(StructureWarning w) {
    switch (w) {
        case WARN_DEGENERATE_CELL: return "degenerate-cell";
        case WARN_DENSITY:         return "density";
        case WARN_VACUUM_SLAB:     return "vacuum-slab";
        default:                   return "";
    }
}

}  // namespace klasp
