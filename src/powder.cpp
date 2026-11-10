#include "klasp/powder.hpp"
#include <cstring>
#include <cstdint>
#include "klasp/constants.hpp"

#include <gemmi/it92.hpp>
#include <gemmi/elem.hpp>

#include <cmath>

namespace klasp {


namespace {

// 'PWS1'. Read as a float this is about 1.4e10, so it cannot head a dense
// (area-normalised) pattern.
constexpr std::uint32_t kPowderSparseMagic = 0x50575331u;

}  // namespace

std::vector<char> encode_powder(const float* pattern, int n_bins) {
    int n_nz = 0;
    for (int i = 0; i < n_bins; ++i) if (pattern[i] != 0.0f) ++n_nz;

    const std::size_t dense = static_cast<std::size_t>(n_bins) * sizeof(float);
    const std::size_t sparse = 2 * sizeof(std::uint32_t)
                             + static_cast<std::size_t>(n_nz) * (sizeof(std::uint16_t) + sizeof(float));
    // Stored dense when sparse would be larger or n_bins exceeds a 16-bit index.
    if (sparse >= dense || n_bins > 0xFFFF) {
        std::vector<char> out(dense);
        std::memcpy(out.data(), pattern, dense);
        return out;
    }

    std::vector<char> out(sparse);
    char* p = out.data();
    const std::uint32_t magic = kPowderSparseMagic;
    const std::uint32_t count = static_cast<std::uint32_t>(n_nz);
    std::memcpy(p, &magic, sizeof(magic)); p += sizeof(magic);
    std::memcpy(p, &count, sizeof(count)); p += sizeof(count);
    char* vals = p + static_cast<std::size_t>(n_nz) * sizeof(std::uint16_t);
    for (int i = 0; i < n_bins; ++i) {
        if (pattern[i] == 0.0f) continue;
        const std::uint16_t idx = static_cast<std::uint16_t>(i);
        std::memcpy(p, &idx, sizeof(idx)); p += sizeof(idx);
        std::memcpy(vals, &pattern[i], sizeof(float)); vals += sizeof(float);
    }
    return out;
}

bool decode_powder(const void* blob, std::size_t n_bytes, float* out, int n_bins) {
    if (!blob || !out || n_bins <= 0) return false;
    const char* p = static_cast<const char*>(blob);

    if (n_bytes == static_cast<std::size_t>(n_bins) * sizeof(float)) {
        std::memcpy(out, p, n_bytes);          // dense
        return true;
    }
    if (n_bytes < 2 * sizeof(std::uint32_t)) return false;
    std::uint32_t magic = 0, count = 0;
    std::memcpy(&magic, p, sizeof(magic));
    if (magic != kPowderSparseMagic) return false;
    std::memcpy(&count, p + sizeof(magic), sizeof(count));

    const std::size_t need = 2 * sizeof(std::uint32_t)
                           + static_cast<std::size_t>(count) * (sizeof(std::uint16_t) + sizeof(float));
    if (need != n_bytes) return false;         // truncated: refuse rather than half-fill

    std::fill(out, out + n_bins, 0.0f);
    const char* idxp = p + 2 * sizeof(std::uint32_t);
    const char* valp = idxp + static_cast<std::size_t>(count) * sizeof(std::uint16_t);
    for (std::uint32_t j = 0; j < count; ++j) {
        std::uint16_t idx = 0;
        std::memcpy(&idx, idxp + j * sizeof(std::uint16_t), sizeof(idx));
        if (idx >= n_bins) return false;
        std::memcpy(&out[idx], valp + j * sizeof(float), sizeof(float));
    }
    return true;
}

PowderFormFactors build_powder_form_factors(
    const std::vector<std::string>& elements)
{
    const int n_atoms = static_cast<int>(elements.size());
    PowderFormFactors ff;
    ff.n_atoms = n_atoms;
    ff.cm = View2D<RealType>("powder_cm", n_atoms, 9);

    auto h_cm = Kokkos::create_mirror_view(ff.cm);
    for (int i = 0; i < n_atoms; ++i) {
        for (int j = 0; j < 9; ++j) h_cm(i, j) = 0;

        auto el = gemmi::find_element(elements[i].c_str());
        // El::X is gemmi's "unknown" (padded ghost slots). has() does not filter
        // it: X is ordinal 0, which IT92 maps to oxygen's row.
        if (el == gemmi::El::X || !gemmi::IT92<double>::has(el))
            continue;  // ghost / unknown → zero scatterer
        const auto& coef = gemmi::IT92<double>::get(el, 0);
        for (int j = 0; j < 4; ++j) {
            h_cm(i, j)     = static_cast<RealType>(coef.a(j));
            h_cm(i, 4 + j) = static_cast<RealType>(coef.b(j));
        }
        h_cm(i, 8) = static_cast<RealType>(coef.c());
    }
    Kokkos::deep_copy(ff.cm, h_cm);
    return ff;
}

PowderResult compute_powder_pattern(const BatchGeometry& geom,
                                    const PowderFormFactors& ff,
                                    const PowderSettings& settings)
{
    const int n_batch = geom.n_batch;
    const int n_atoms = geom.n_atoms;
    const int n_bins  = settings.n_bins;

    const RealType lambda   = static_cast<RealType>(settings.wavelength);
    const RealType tt_min   = static_cast<RealType>(settings.two_theta_min);
    const RealType tt_max   = static_cast<RealType>(settings.two_theta_max);
    const RealType bin_step = (tt_max - tt_min) / static_cast<RealType>(n_bins);
    const RealType sigma    = static_cast<RealType>(settings.peak_fwhm) /
                              static_cast<RealType>(2.35482);  // FWHM → σ
    const int kernel_half   = std::max(1, static_cast<int>(
                                 std::ceil(3.0 * settings.peak_fwhm / 2.35482 / bin_step)));
    // g range in cycles/Å: |g| = 2 sin(θ)/λ.
    const RealType sin_half_max = static_cast<RealType>(
        std::sin(0.5 * settings.two_theta_max * M_PI / 180.0));
    const RealType sin_half_min = static_cast<RealType>(
        std::sin(0.5 * settings.two_theta_min * M_PI / 180.0));
    const RealType g_max = 2 * sin_half_max / lambda;
    const RealType g_min = 2 * sin_half_min / lambda;
    const RealType g_max_sq = g_max * g_max;
    const RealType g_min_sq = g_min * g_min;
    const RealType inv_bohr_to_A = RealType(1) / BOHR_TO_ANGSTROM;
    const RealType bohr_to_A = static_cast<RealType>(BOHR_TO_ANGSTROM);
    const bool do_lp = settings.lorentz_polarization;

    // hkl range from the cells: |h| <= |a| g_max, so the longest axis in the
    // batch sets it. Surplus hkl fail the g_sq window test in the kernel.
    int H = 1;
    {
        auto cell = geom.cell;
        int h_needed = 1;
        Kokkos::parallel_reduce("powder_hkl_range",
            Kokkos::RangePolicy<ExecSpace>(0, n_batch * 3),
            KOKKOS_LAMBDA(const int flat, int& lmax) {
                const int b = flat / 3;
                const int j = flat % 3;          // lattice vector = column j
                RealType len_sq = 0;
                for (int d = 0; d < 3; ++d) {
                    const RealType c = cell(b, d, j);
                    len_sq += c * c;
                }
                const RealType len_A = Kokkos::sqrt(len_sq) * bohr_to_A;
                const int need = static_cast<int>(Kokkos::ceil(len_A * g_max));
                if (need > lmax) lmax = need;
            }, Kokkos::Max<int>(h_needed));
        H = std::min(std::max(1, h_needed), settings.max_hkl);
    }
    const int H2    = 2 * H + 1;
    const int n_hkl = H2 * H2 * H2;

    // Fractional positions r_frac = cell_inv @ r_cart (both sides in Bohr).
    auto positions = geom.positions;
    auto cell_inv  = geom.cell_inv;
    View3D<RealType> r_frac("powder_rfrac", n_batch, n_atoms, 3);
    Kokkos::parallel_for("powder_rfrac",
        Kokkos::RangePolicy<ExecSpace>(0, n_batch * n_atoms),
        KOKKOS_LAMBDA(const int flat) {
            const int b = flat / n_atoms;
            const int i = flat % n_atoms;
            const RealType rx = positions(b, i, 0);
            const RealType ry = positions(b, i, 1);
            const RealType rz = positions(b, i, 2);
            for (int d = 0; d < 3; ++d) {
                r_frac(b, i, d) = cell_inv(b, d, 0) * rx
                                + cell_inv(b, d, 1) * ry
                                + cell_inv(b, d, 2) * rz;
            }
        });

    View2D<RealType> d_pat("powder_pat", n_batch, n_bins);
    Kokkos::deep_copy(d_pat, RealType(0));

    auto cm = ff.cm;
    ExecSpace().fence();

    // Main kernel: one thread per (batch, h, k, l). Computes F(hkl) by
    // summing over atoms, then Gaussian-smears |F|² into the 2θ histogram.
    Kokkos::parallel_for("powder_sf_bin",
        Kokkos::RangePolicy<ExecSpace>(0, n_batch * n_hkl),
        KOKKOS_LAMBDA(const int flat) {
            const int b = flat / n_hkl;
            const int s = flat % n_hkl;
            const int h = s / (H2 * H2) - H;
            const int rem = s % (H2 * H2);
            const int k = rem / H2 - H;
            const int l = rem % H2 - H;
            if (h == 0 && k == 0 && l == 0) return;

            // g_cart in 1/Å: rows of cell_inv are reciprocal vectors in 1/Bohr.
            RealType gx = (h * cell_inv(b, 0, 0) + k * cell_inv(b, 1, 0) + l * cell_inv(b, 2, 0)) * inv_bohr_to_A;
            RealType gy = (h * cell_inv(b, 0, 1) + k * cell_inv(b, 1, 1) + l * cell_inv(b, 2, 1)) * inv_bohr_to_A;
            RealType gz = (h * cell_inv(b, 0, 2) + k * cell_inv(b, 1, 2) + l * cell_inv(b, 2, 2)) * inv_bohr_to_A;
            const RealType g_sq = gx * gx + gy * gy + gz * gz;
            if (g_sq < g_min_sq || g_sq > g_max_sq) return;

            const RealType g_mag   = Kokkos::sqrt(g_sq);
            const RealType sin_th  = RealType(0.5) * lambda * g_mag;
            if (sin_th >= RealType(1)) return;
            const RealType theta   = Kokkos::asin(sin_th);
            const RealType two_theta_deg = RealType(2) * theta *
                                           RealType(180.0 / M_PI);

            // Structure factor Σ_j f_j(stol²) exp(2πi (h x_j + k y_j + l z_j)).
            const RealType stol2 = RealType(0.25) * g_sq;  // (sin θ/λ)²
            RealType f_re = 0, f_im = 0;
            for (int j = 0; j < n_atoms; ++j) {
                RealType fj = cm(j, 8);
                for (int q = 0; q < 4; ++q)
                    fj += cm(j, q) * Kokkos::exp(-cm(j, 4 + q) * stol2);
                if (Kokkos::fabs(fj) < RealType(1e-30)) continue;
                const RealType phase = RealType(2.0 * M_PI) *
                    (h * r_frac(b, j, 0) + k * r_frac(b, j, 1) + l * r_frac(b, j, 2));
                f_re += fj * Kokkos::cos(phase);
                f_im += fj * Kokkos::sin(phase);
            }
            RealType intensity = f_re * f_re + f_im * f_im;

            if (do_lp) {
                // Standard powder LP factor: (1 + cos²2θ) / (sin²θ cosθ).
                const RealType cos_th = Kokkos::sqrt(
                    RealType(1) - sin_th * sin_th);
                const RealType cos_2th = RealType(1) - RealType(2) * sin_th * sin_th;
                intensity *= (RealType(1) + cos_2th * cos_2th) /
                             (sin_th * sin_th * cos_th);
            }

            // Gaussian peak profile, scattered into bin_center ± kernel_half.
            const RealType center = (two_theta_deg - tt_min) / bin_step - RealType(0.5);
            const int c0 = static_cast<int>(Kokkos::floor(center));
            const RealType inv_sigma = RealType(1) / sigma;
            for (int dbin = -kernel_half; dbin <= kernel_half; ++dbin) {
                const int bb = c0 + dbin;
                if (bb < 0 || bb >= n_bins) continue;
                const RealType bin_center_2t = tt_min + (bb + RealType(0.5)) * bin_step;
                const RealType xnorm = (bin_center_2t - two_theta_deg) * inv_sigma;
                const RealType w = Kokkos::exp(RealType(-0.5) * xnorm * xnorm);
                Kokkos::atomic_add(&d_pat(b, bb), intensity * w);
            }
        });
    ExecSpace().fence();

    // Normalize each pattern. L2 = unit norm. AREA = trapezoidal integral = 1,
    // matching mol-cspy's Simpson-normalized patterns.
    const int norm_mode =
        (settings.normalization == PowderNormalization::L2)   ? 2 :
        (settings.normalization == PowderNormalization::AREA) ? 1 : 0;
    if (norm_mode != 0) {
        const RealType dx = bin_step;
        Kokkos::parallel_for("powder_norm",
            Kokkos::RangePolicy<ExecSpace>(0, n_batch),
            KOKKOS_LAMBDA(const int b) {
                RealType metric = 0;
                if (norm_mode == 2) {
                    for (int i = 0; i < n_bins; ++i) {
                        const RealType v = d_pat(b, i);
                        metric += v * v;
                    }
                    metric = Kokkos::sqrt(metric);
                } else {
                    // Trapezoidal ∫p d(2θ) with interior bins weighted 1.
                    RealType sum = 0;
                    for (int i = 0; i < n_bins; ++i) sum += d_pat(b, i);
                    const RealType half_ends = RealType(0.5) *
                        (d_pat(b, 0) + d_pat(b, n_bins - 1));
                    metric = (sum - half_ends) * dx;
                }
                const RealType inv = (metric > RealType(0))
                    ? RealType(1) / metric : RealType(0);
                for (int i = 0; i < n_bins; ++i) d_pat(b, i) *= inv;
            });
        ExecSpace().fence();
    }

    auto h_pat = Kokkos::create_mirror_view(d_pat);
    Kokkos::deep_copy(ExecSpace(), h_pat, d_pat);

    PowderResult result;
    result.n_bins = n_bins;
    result.two_theta.resize(n_bins);
    for (int i = 0; i < n_bins; ++i)
        result.two_theta[i] = settings.two_theta_min +
                              (i + 0.5) * (settings.two_theta_max - settings.two_theta_min)
                              / n_bins;
    result.patterns.resize(n_batch);
    for (int b = 0; b < n_batch; ++b) {
        result.patterns[b].resize(n_bins);
        for (int i = 0; i < n_bins; ++i)
            result.patterns[b][i] = static_cast<double>(h_pat(b, i));
    }
    return result;
}

}  // namespace klasp
