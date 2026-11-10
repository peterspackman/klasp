#pragma once

// Ensure M_PI is available (needed for MSVC and some CUDA toolchains)
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include "klasp/types.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace klasp {

// Unit conversion constants (matching dmapy/occ)
inline constexpr RealType BOHR_TO_ANGSTROM = 0.529177210903;
inline constexpr RealType ANGSTROM_TO_BOHR = RealType(1.0) / BOHR_TO_ANGSTROM;
inline constexpr RealType AU_TO_KJ_PER_MOL = 2625.4996394799;
inline constexpr RealType EV_TO_KJ = 96.4853321233100184;
// Grams per amu. Use this everywhere so CIF and database densities agree.
inline constexpr double AMU_TO_GRAM = 1.66053906660e-24;

inline constexpr RealType EV_TO_AU = EV_TO_KJ / AU_TO_KJ_PER_MOL;  // 1 eV in Hartree

}  // namespace klasp
