#include "klasp/potentials.hpp"
#include "klasp/constants.hpp"
#include "klasp/energy_kernels.hpp"
#include "klasp/interp.hpp"

#include <fmt/format.h>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace klasp {

using json = nlohmann::json;

namespace {

// Unit conversions: A in eV in the input, Hartree in the kernels.
double a_from_json(double A_ev)  { return A_ev * double(EV_TO_AU); }
double a_to_json(double A_au)    { return A_au / double(EV_TO_AU); }

// The input gives rho in Angstrom; the kernels want B = 1/rho in 1/Bohr.
double b_from_json(double rho_ang) { return (1.0 / rho_ang) * double(BOHR_TO_ANGSTROM); }
double b_to_json(double B_bohr)    { return double(BOHR_TO_ANGSTROM) / B_bohr; }

double c_scale() { return double(EV_TO_AU) * std::pow(double(ANGSTROM_TO_BOHR), 6); }
double c_from_json(double C6) { return C6 * c_scale(); }
double c_to_json(double C_au) { return C_au / c_scale(); }

}  // namespace

PotentialTable resolve_potentials(const json& potentials,
                                  const std::map<std::string, int>& type_map) {
    PotentialTable t;
    const int n = static_cast<int>(type_map.size());
    t.type_names.assign(n, std::string());
    t.element_of.assign(n, std::string());
    t.A.assign(size_t(n) * n, 0.0);
    t.B.assign(size_t(n) * n, 0.0);
    t.C.assign(size_t(n) * n, 0.0);
    for (const auto& [name, idx] : type_map)
        if (idx >= 0 && idx < n) t.type_names[idx] = name;

    if (!potentials.is_object() || !potentials.contains("buckingham")) return t;
    const json& list = potentials["buckingham"];
    if (!list.is_array()) return t;

    for (const auto& pair : list) {
        if (!pair.contains("types")) {
            t.skipped.push_back("an entry with no \"types\"");
            continue;
        }
        const auto types = pair["types"].get<std::vector<std::string>>();
        if (types.size() < 2) {
            t.skipped.push_back("an entry whose \"types\" names fewer than two types");
            continue;
        }
        const auto i1 = type_map.find(types[0]);
        const auto i2 = type_map.find(types[1]);
        if (i1 == type_map.end() || i2 == type_map.end()) continue;   // a type not in use
        const std::string label = types[0] + "," + types[1];
        if (!pair.contains("A") || !pair.contains("rho") || !pair.contains("C6")) {
            t.skipped.push_back(label + ", which is missing A, rho or C6");
            continue;
        }
        const double rho = pair["rho"].get<double>();
        if (!(rho > 0.0)) {
            t.skipped.push_back(label + ", whose rho is not positive");
            continue;
        }

        t.set(i1->second, i2->second,
              a_from_json(pair["A"].get<double>()),
              b_from_json(rho),
              c_from_json(pair["C6"].get<double>()));

        // Remember each type's element so the record can carry it; the pair
        // lookup is by type.
        if (pair.contains("elements")) {
            const auto els = pair["elements"].get<std::vector<std::string>>();
            if (els.size() >= 2) {
                if (t.element_of[i1->second].empty()) t.element_of[i1->second] = els[0];
                if (t.element_of[i2->second].empty()) t.element_of[i2->second] = els[1];
            }
        }
    }

    // Good-Hope for whatever is left, e.g. co-crystal cross-component pairs.
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (!t.unset(i, j)) continue;
            if (t.unset(i, i) || t.unset(j, j)) continue;  // nothing to combine
            t.set(i, j,
                  std::sqrt(t.a(i, i) * t.a(j, j)),
                  0.5 * (t.b(i, i) + t.b(j, j)),
                  std::sqrt(std::max(0.0, t.c(i, i)) * std::max(0.0, t.c(j, j))));
            t.derived.push_back(t.type_names[i] + "," + t.type_names[j]);
        }
    }

    std::vector<bool> paired(static_cast<size_t>(n), false);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n && !paired[i]; ++j) paired[i] = !t.unset(i, j);
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j)
            if (t.unset(i, j) && paired[i] && paired[j])
                t.undefined.push_back(t.type_names[i] + "," + t.type_names[j]);
    return t;
}

json potential_table_to_json(const PotentialTable& t) {
    json list = json::array();
    const int n = t.n_types();
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            if (t.unset(i, j)) continue;   // never listed, never derivable
            json e;
            e["types"] = json::array({t.type_names[i], t.type_names[j]});
            if (!t.element_of[i].empty() && !t.element_of[j].empty())
                e["elements"] = json::array({t.element_of[i], t.element_of[j]});
            e["A"]   = a_to_json(t.a(i, j));
            e["rho"] = b_to_json(t.b(i, j));
            e["C6"]  = c_to_json(t.c(i, j));
            list.push_back(std::move(e));
        }
    }
    return list;
}

void fill_workspace_potentials(const PotentialTable& t, Workspace& ws) {
    const int n = t.n_types();
    // One row and column past the real types, left at zero for the ghost type.
    const int n_with_ghost = n + 1;
    if (n_with_ghost > int(ws.buck_A.extent(0)))
        throw std::runtime_error(fmt::format(
            "{} atom types (plus the ghost type) exceeds the {} this workspace "
            "was allocated for", n, ws.buck_A.extent(0)));

    auto h_A = Kokkos::create_mirror_view(ws.buck_A);
    auto h_B = Kokkos::create_mirror_view(ws.buck_B);
    auto h_C = Kokkos::create_mirror_view(ws.buck_C);
    auto h_R = Kokkos::create_mirror_view(ws.buck_Rmin);
    for (int i = 0; i < n_with_ghost; ++i)
        for (int j = 0; j < n_with_ghost; ++j) {
            const bool real = i < n && j < n;
            h_A(i, j) = real ? RealType(t.a(i, j)) : RealType(0);
            h_B(i, j) = real ? RealType(t.b(i, j)) : RealType(0);
            h_C(i, j) = real ? RealType(t.c(i, j)) : RealType(0);
            h_R(i, j) = buckingham_r_min(h_A(i, j), h_B(i, j), h_C(i, j));
        }
    Kokkos::deep_copy(ws.buck_A, h_A);
    Kokkos::deep_copy(ws.buck_B, h_B);
    Kokkos::deep_copy(ws.buck_C, h_C);
    Kokkos::deep_copy(ws.buck_Rmin, h_R);
    ws.n_types = n_with_ghost;
    ws.ghost_type = t.ghost_type();
}

// ---------------------------------------------------------------------------
// The tabulated model.
// ---------------------------------------------------------------------------

std::string TabulatedPotentials::summary() const {
    return fmt::format("{} knots over {:.2f}-{:.2f} Bohr, {} of {} type pairs, "
                       "{:.1f} KB, worst relative error {:.1e}",
                       n_knots, r_min, r_max, n_tables, n_types * n_types,
                       double(coeffs.size() * sizeof(RealType)) / 1024.0,
                       worst_rel_error);
}

TabulatedSpec parse_tabulated_spec(const json& potentials) {
    TabulatedSpec spec;
    if (!potentials.is_object() || !potentials.contains("tabulated")) return spec;
    const json& t = potentials["tabulated"];
    // `"tabulated": true` is the shorthand for "yes, with the defaults".
    if (t.is_boolean()) { spec.enabled = t.get<bool>(); return spec; }
    if (!t.is_object()) return spec;
    spec.enabled = t.value("enabled", true);
    spec.n_knots = t.value("n_knots", spec.n_knots);
    spec.r_min   = t.value("r_min", spec.r_min);
    spec.r_max   = t.value("r_max", spec.r_max);
    return spec;
}

TangToenniesDamping parse_damping(const json& potentials) {
    TangToenniesDamping d;
    if (!potentials.is_object()) return d;
    d.enabled = potentials.value("damping", d.enabled);
    if (!potentials.contains("damping_range")) return d;
    const json& range = potentials["damping_range"];
    if (!range.is_number() || !(range.get<double>() > 0) || !std::isfinite(range.get<double>()))
        throw std::runtime_error(fmt::format(
            "potentials.damping_range must be a positive number; got {}", range.dump()));
    d.range = RealType(range.get<double>());
    return d;
}

void record_damping(json& potentials, const TangToenniesDamping& d) {
    potentials["damping"] = d.enabled;
    potentials["damping_range"] = double(d.range);
}

std::shared_ptr<TabulatedPotentials> tabulate_potentials(const PotentialTable& t,
                                                         const TangToenniesDamping& damping,
                                                         const TabulatedSpec& spec,
                                                         double cutoff_bohr) {
    if (!spec.enabled) return nullptr;
    const int n_real = t.n_types();
    // One past the real types: the ghost row and column stay untabulated (zero).
    const int n = n_real + 1;
    const double r_max = spec.r_max > 0 ? spec.r_max : cutoff_bohr;
    if (!(r_max > spec.r_min) || spec.n_knots < 4)
        throw std::runtime_error(fmt::format(
            "potentials.tabulated needs r_max > r_min and at least 4 knots; got "
            "r_min={:g}, r_max={:g}, n_knots={}", spec.r_min, r_max, spec.n_knots));

    auto out = std::make_shared<TabulatedPotentials>();
    out->n_types = n;
    out->n_knots = spec.n_knots;
    out->r_min = spec.r_min;
    out->r_max = r_max;

    // Exponential continuation below the table, zero above it (past the taper).
    const RadialEnds ends{RadialBelow::Exponential, RadialAbove::Zero};
    const std::vector<double> knots =
        radial_knots(spec.r_min, r_max, spec.n_knots, RadialGrid::Linear);

    // Each pair tabulates the untapered closed form including the short-range
    // cap, as BuckinghamModel evaluates it. `cap` is the pair's inner turning point.
    auto pair_value = [&](int i, int j, RealType cap, double r) {
        RealType e = 0, de = 0;
        buckingham_untapered(RealType(r), RealType(1.0 / r),
                             RealType(t.a(i, j)), RealType(t.b(i, j)),
                             RealType(t.c(i, j)), damping, cap, e, de);
        return double(e);
    };

    std::vector<RealType> flat;
    std::vector<RadialSpline<>> headers(size_t(n) * n);
    std::vector<size_t> offsets(size_t(n) * n, 0);
    std::vector<bool> present(size_t(n) * n, false);

    for (int i = 0; i < n_real; ++i) {
        for (int j = 0; j < n_real; ++j) {
            if (t.unset(i, j)) continue;   // no potential; a zero spline would agree
            const RealType cap = buckingham_r_min(RealType(t.a(i, j)),
                                                  RealType(t.b(i, j)),
                                                  RealType(t.c(i, j)));
            std::vector<double> samples(knots.size());
            for (size_t k = 0; k < knots.size(); ++k)
                samples[k] = pair_value(i, j, cap, knots[k]);
            RadialTable<> tab(spec.r_min, r_max, samples, ends);
            const size_t off = flat.size();
            const auto& c = tab.coefficients();
            flat.insert(flat.end(), c.begin(), c.end());
            headers[size_t(i) * n + j] = tab.view();
            offsets[size_t(i) * n + j] = off;
            present[size_t(i) * n + j] = true;
            ++out->n_tables;

            // Worst case at interval midpoints, relative to the largest |V| on
            // the grid (the potential passes through zero).
            double scale = 0;
            for (double v : samples) scale = std::max(scale, std::abs(v));
            if (scale > 0) {
                auto sp = tab.view();
                for (size_t k = 0; k + 1 < knots.size(); ++k) {
                    const double rm = 0.5 * (knots[k] + knots[k + 1]);
                    const double got = double(sp.value(RealType(rm)));
                    out->worst_rel_error = std::max(
                        out->worst_rel_error,
                        std::abs(got - pair_value(i, j, cap, rm)) / scale);
                }
            }
        }
    }

    out->coeffs = View1D<RealType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "pair_table_coeffs"),
        flat.size());
    copy_to_view(out->coeffs, flat);

    // Re-point the headers from the host vector to the device array before uploading.
    RealType* base = out->coeffs.data();
    for (size_t k = 0; k < headers.size(); ++k)
        headers[k].c = present[k] ? base + offsets[k] : nullptr;

    out->splines = View1D<RadialSpline<>>("pair_tables", headers.size());
    auto h = Kokkos::create_mirror_view(out->splines);
    for (size_t k = 0; k < headers.size(); ++k) h(k) = headers[k];
    Kokkos::deep_copy(out->splines, h);
    return out;
}

bool tables_agree(const PotentialTable& x, const PotentialTable& y, double rel_tol) {
    if (x.type_names != y.type_names) return false;
    auto close = [rel_tol](double p, double q) {
        const double scale = std::max({1.0, std::abs(p), std::abs(q)});
        return std::abs(p - q) <= rel_tol * scale;
    };
    const int n = x.n_types();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (!close(x.a(i, j), y.a(i, j)) ||
                !close(x.b(i, j), y.b(i, j)) ||
                !close(x.c(i, j), y.c(i, j)))
                return false;
    return true;
}

}  // namespace klasp
