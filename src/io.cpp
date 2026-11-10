#include "klasp/log.hpp"
#include "klasp/io.hpp"
#include "klasp/csp_cif.hpp"
#include "klasp/niggli.hpp"
#include "klasp/constants.hpp"
#include "klasp/cell.hpp"
#include "klasp/element.hpp"
#include "klasp/energy_kernels.hpp"
#include "klasp/multipole.hpp"
#include "klasp/rotation.hpp"
#include "klasp/sgdata.hpp"

#include <gemmi/cif.hpp>
#include <gemmi/to_cif.hpp>
#include <gemmi/symmetry.hpp>

#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <algorithm>
#include <set>
#include <regex>

using json = nlohmann::json;

namespace klasp {

// Forward declarations; defined further down.

static void apply_settings(const json& s, Workspace& ws, int& max_rank);
static void warn_if_charged(const std::vector<double>& site_charges, const char* what);

// Each distinct message once per process.
static void warn_once(const std::string& message) {
    static std::set<std::string> already_said;
    if (already_said.insert(message).second) logging::warn("warning: {}\n", message);
}

// Warn about what a potentials block leaves undefined: types no pair reaches,
// zero pairs between parameterised types, and unreadable entries.
static void warn_potential_gaps(const PotentialTable& t) {
    for (int i = 0; i < t.n_types(); ++i) {
        bool paired = false;
        for (int j = 0; j < t.n_types() && !paired; ++j) paired = !t.unset(i, j);
        if (!paired)
            warn_once("atom type '" + t.type_names[i] + "' has no Buckingham parameters with "
                      "any type, so its sites have no repulsion-dispersion");
    }
    for (const auto& pair : t.undefined)
        warn_once("no Buckingham parameters for the pair (" + pair + "): it is not listed "
                  "and cannot be combined, so those atoms have no repulsion-dispersion "
                  "between them");
    for (const auto& entry : t.skipped)
        warn_once("a buckingham entry was skipped: " + entry);
}

// Warn about what a site leaves undefined: a missing `type`, an unknown element,
// or a multipole array of the wrong length.
static void warn_site_gaps(const json& site, const std::string& molecule_type, int index) {
    const std::string label = site.value("label", std::string());
    const std::string where = "molecule type '" + molecule_type + "', site " +
                              std::to_string(index) + (label.empty() ? "" : " (" + label + ")");
    if (!site.contains("type"))
        warn_once(where + " has no type, so it is type \"X\"");
    if (site.contains("element") || site.contains("type")) {
        const std::string named = site.contains("element") ? site.value("element", std::string())
                                                            : site.value("type", std::string());
        std::string symbol = named;
        if (symbol.size() > 1 && std::islower(static_cast<unsigned char>(symbol[1])))
            symbol = symbol.substr(0, 2);
        else
            symbol = symbol.substr(0, 1);
        if (element(symbol).atomic_number == 0)
            warn_once(where + " is element '" + named + "', which klasp does not know: it gets "
                      "atomic number 0 and a carbon's mass and radius");
    }
    if (site.contains("multipoles") && site["multipoles"].is_object()) {
        const json& mp = site["multipoles"];
        const std::pair<const char*, size_t> ranks[] = {
            {"dipole", 3}, {"quadrupole", 5}, {"octupole", 7}, {"hexadecapole", 9}};
        for (const auto& [key, want] : ranks) {
            if (!mp.contains(key)) continue;
            const size_t got = mp[key].is_array() ? mp[key].size() : 1;
            if (got != want)
                warn_once(where + " has a " + key + " of " + std::to_string(got) +
                          " components where there should be " + std::to_string(want) +
                          ", which shifts every component after it");
        }
    }
}

// Parse a CIF-style symmetry string "-x+1/2, y, -z" into rotation + translation.
static void parse_symop_string(const std::string& s,
                                double rot[3][3], double trans[3]) {
    for (int i = 0; i < 3; ++i) {
        trans[i] = 0.0;
        for (int j = 0; j < 3; ++j)
            rot[i][j] = 0.0;
    }

    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    lower.erase(std::remove(lower.begin(), lower.end(), ' '), lower.end());

    std::vector<std::string> tokens;
    std::stringstream ss(lower);
    std::string tok;
    while (std::getline(ss, tok, ','))
        tokens.push_back(tok);

    std::regex symm_re("([+-]?[xyz0-9/\\.]+)");

    for (int i = 0; i < 3 && i < static_cast<int>(tokens.size()); ++i) {
        auto begin = std::sregex_iterator(tokens[i].begin(), tokens[i].end(), symm_re);
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            std::string sym = (*it)[1].str();
            if (sym.find('x') != std::string::npos) {
                rot[i][0] = (sym.find("-x") != std::string::npos) ? -1.0 : 1.0;
            } else if (sym.find('y') != std::string::npos) {
                rot[i][1] = (sym.find("-y") != std::string::npos) ? -1.0 : 1.0;
            } else if (sym.find('z') != std::string::npos) {
                rot[i][2] = (sym.find("-z") != std::string::npos) ? -1.0 : 1.0;
            } else {
                // Numeric, possibly a fraction like "1/2"
                auto slash = sym.find('/');
                if (slash != std::string::npos) {
                    double num = std::stod(sym.substr(0, slash));
                    double den = std::stod(sym.substr(slash + 1));
                    trans[i] += num / den;
                } else {
                    trans[i] += std::stod(sym);
                }
            }
        }
        // Wrap translation to [0, 1).
        trans[i] = std::fmod(trans[i], 1.0);
        if (trans[i] < 0.0) trans[i] += 1.0;
    }
}

// (-1)^l parity signs for multipole components.
static std::vector<double> multipole_parity_signs(int n_components) {
    std::vector<double> signs(n_components, 1.0);
    int offset = 0;
    for (int l = 0; l < 20; ++l) {
        int size = 2 * l + 1;
        if (offset + size > n_components) break;
        if (l % 2 == 1) {
            for (int i = offset; i < offset + size; ++i)
                signs[i] = -1.0;
        }
        offset += size;
    }
    return signs;
}

CrystalConfig load_structure_json(const std::string& path,
                                  MoleculeData& mol, Workspace& ws) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + path);
    return load_structure_json(json::parse(f), mol, ws);
}

const nlohmann::json& structure_molecule_types(const nlohmann::json& doc) {
    if (doc.contains("molecule_types")) return doc["molecule_types"];
    if (doc.contains("basis") && doc["basis"].contains("molecule_types"))
        return doc["basis"]["molecule_types"];
    throw std::runtime_error(
        "structure JSON has no molecule_types (looked at the top level and under \"basis\")");
}

CrystalConfig load_structure_json(const nlohmann::json& data_in,
                                  MoleculeData& mol, Workspace& ws) {
    json data = data_in;

    // occ format { basis: { molecule_types }, crystal: { cell, molecules } } is
    // remapped to the flat { cell, molecules, molecule_types } layout.
    if (data.contains("crystal") && data.contains("basis")) {
        json flat;
        flat["cell"] = data["crystal"]["cell"];
        flat["molecules"] = data["crystal"]["molecules"];
        flat["molecule_types"] = data["basis"]["molecule_types"];
        if (data["basis"].contains("potentials"))
            flat["potentials"] = data["basis"]["potentials"];
        if (data["basis"].contains("settings"))
            flat["settings"] = data["basis"]["settings"];
        if (data.contains("symmetry"))
            flat["symmetry"] = data["symmetry"];
        if (data.contains("space_group"))
            flat["space_group"] = data["space_group"];
        data = std::move(flat);
    }

    auto& jcell = data["cell"];
    double cell_a = jcell["a"].get<double>();
    double cell_b = jcell["b"].get<double>();
    double cell_c = jcell["c"].get<double>();
    double cell_alpha = jcell["alpha"].get<double>();
    double cell_beta = jcell["beta"].get<double>();
    double cell_gamma = jcell["gamma"].get<double>();

    std::map<std::string, int> type_map;
    int next_type = 0;
    for (auto& mt : data["molecule_types"]) {
        for (auto& site : mt["sites"]) {
            std::string atype = site.value("type", "X");
            if (type_map.find(atype) == type_map.end())
                type_map[atype] = next_type++;
        }
    }
    int n_types = next_type;
    // One past the real types, with a zero row and column in the potential
    // tables; padded sites carry it (see PotentialTable::ghost_type).
    const int ghost_type = n_types;

    std::map<std::string, int> type_name_to_idx;
    for (int i = 0; i < static_cast<int>(data["molecule_types"].size()); ++i) {
        const std::string name = data["molecule_types"][i]["name"].get<std::string>();
        if (type_name_to_idx.count(name))
            warn_once("molecule_types lists '" + name + "' more than once; the later entry is "
                      "the one molecules of that type use");
        type_name_to_idx[name] = i;
        const json& type_sites = data["molecule_types"][i]["sites"];
        for (int j = 0; j < static_cast<int>(type_sites.size()); ++j)
            warn_site_gaps(type_sites[j], name, j);
    }

    std::vector<int> molecule_type_index;
    for (int m = 0; m < static_cast<int>(data["molecules"].size()); ++m) {
        const std::string name = data["molecules"][m]["type"].get<std::string>();
        const auto it = type_name_to_idx.find(name);
        if (it == type_name_to_idx.end()) {
            std::string known;
            for (const auto& [type, index] : type_name_to_idx)
                known += (known.empty() ? "" : ", ") + type;
            throw std::runtime_error("molecules[" + std::to_string(m) + "] is of type '" + name +
                                     "', which molecule_types does not define (it has " +
                                     known + ")");
        }
        molecule_type_index.push_back(it->second);
    }

    // Max multipole rank and n_sites, the stride every molecule is laid out
    // with: the largest component's site count.
    int max_rank = 0;
    int n_sites = 0;
    for (auto& mt : data["molecule_types"])
        n_sites = std::max(n_sites, static_cast<int>(mt["sites"].size()));
    std::vector<int> sites_of_type;
    for (auto& mt : data["molecule_types"])
        sites_of_type.push_back(static_cast<int>(mt["sites"].size()));
    for (auto& mt : data["molecule_types"]) {
        for (auto& site : mt["sites"]) {
            if (!site.contains("multipoles")) continue;
            auto& mp = site["multipoles"];
            if (mp.contains("hexadecapole")) max_rank = std::max(max_rank, 4);
            else if (mp.contains("octupole")) max_rank = std::max(max_rank, 3);
            else if (mp.contains("quadrupole")) max_rank = std::max(max_rank, 2);
            else if (mp.contains("dipole")) max_rank = std::max(max_rank, 1);
        }
    }
    std::vector<std::array<double, 9>> sym_rots;
    std::vector<std::array<double, 3>> sym_trans;
    std::vector<int> sym_indep;
    // Per molecule, whether its one operation inverts it (no symmetry list only).
    std::vector<bool> inverted(data["molecules"].size(), false);

    const bool has_symmetry = data.contains("symmetry") && !data["symmetry"].empty();
    if (has_symmetry) {
        for (auto& se : data["symmetry"]) {
            double rot[3][3], trans[3];
            parse_symop_string(se["op"].get<std::string>(), rot, trans);
            std::array<double, 9> flat_rot;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    flat_rot[i * 3 + j] = rot[i][j];
            sym_rots.push_back(flat_rot);
            sym_trans.push_back({trans[0], trans[1], trans[2]});
            const int copied = se.value("molecule", 0);
            if (copied < 0 || copied >= static_cast<int>(data["molecules"].size()))
                throw std::runtime_error("symmetry entry '" + se["op"].get<std::string>() +
                                         "' copies molecule " + std::to_string(copied) +
                                         ", and there are " +
                                         std::to_string(data["molecules"].size()) + " molecules");
            sym_indep.push_back(copied);
        }
    } else {
        // P1: one symop per molecule. parity = -1 is encoded as S_frac = -I, which
        // makes S_cart = -I for the multipole rotation.
        for (int i = 0; i < static_cast<int>(data["molecules"].size()); ++i) {
            auto& jm = data["molecules"][i];
            int parity = 1;
            if (jm.contains("rotation")) {
                auto R = jm["rotation"];
                RealType Rm[3][3];
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        Rm[r][c] = R[r][c].get<double>();
                parity = rotation_parity(Rm);
            } else {
                parity = jm.value("parity", 1);
            }
            // Inverted molecule: w = 1 and t = 1 - centre, so the centre 1 - t
            // stays inside the cell for any wrapped t.
            if (parity == -1) {
                sym_rots.push_back({-1,0,0, 0,-1,0, 0,0,-1});
                sym_trans.push_back({1.0, 1.0, 1.0});
                inverted[i] = true;
            } else {
                sym_rots.push_back({1,0,0, 0,1,0, 0,0,1});
                sym_trans.push_back({0, 0, 0});
            }
            sym_indep.push_back(i);
        }
    }

    int n_indep = static_cast<int>(data["molecules"].size());
    int n_symops = static_cast<int>(sym_rots.size());
    int n_atoms = n_symops * n_sites;
    if (n_atoms > static_cast<int>(ws.positions.extent(0)))
        throw std::runtime_error("this structure places " + std::to_string(n_atoms) +
                                 " atoms (" + std::to_string(n_symops) +
                                 " molecule copies at up to " + std::to_string(n_sites) +
                                 " sites) and the workspace holds " +
                                 std::to_string(ws.positions.extent(0)));

    mol = allocate_molecule_data(n_indep, n_sites, n_symops, max_rank);
    // ws must already be allocated with sufficient sizes.

    // Cell lengths to Bohr; angles unchanged.
    mol.cell_params[0] = cell_a * ANGSTROM_TO_BOHR;
    mol.cell_params[1] = cell_b * ANGSTROM_TO_BOHR;
    mol.cell_params[2] = cell_c * ANGSTROM_TO_BOHR;
    mol.cell_params[3] = cell_alpha;
    mol.cell_params[4] = cell_beta;
    mol.cell_params[5] = cell_gamma;

    mol.n_indep = n_indep;
    mol.n_symops = n_symops;
    mol.n_sites_per_mol = n_sites;
    mol.max_multipole_rank = max_rank;

    {
        auto h_trans = Kokkos::create_mirror_view(mol.translations);
        auto h_rot = Kokkos::create_mirror_view(mol.rot_vectors);
        for (int m = 0; m < n_indep; ++m) {
            auto& jm = data["molecules"][m];
            auto tr = jm["translation"].get<std::vector<double>>();
            // Wrapped into [0, 1), as the optimiser keeps it. An inverted molecule
            // stores 1 - centre.
            for (int d = 0; d < 3; ++d) {
                const double centre = tr[d] - std::floor(tr[d]);
                h_trans(m, d) = inverted[m] ? 1.0 - centre : centre;
            }
            if (jm.contains("rotation")) {
                // Full 3x3 rotation matrix to its proper part to angle-axis.
                auto R = jm["rotation"];
                RealType Rm[3][3];
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        Rm[r][c] = R[r][c].get<double>();
                double sign = static_cast<double>(rotation_parity(Rm));
                // Only the one-operation-per-molecule layout can carry an inversion.
                if (sign < 0 && has_symmetry)
                    throw std::runtime_error(
                        "molecules[" + std::to_string(m) + "] has an improper rotation "
                        "(determinant -1), which a symmetry list cannot express; mirror the "
                        "body in molecule_types, or give the structure without a symmetry list");
                RealType R_proper[3][3];
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        R_proper[r][c] = sign * Rm[r][c];
                RealType aa[3];
                rotation_matrix_to_angle_axis(R_proper, aa);
                for (int d = 0; d < 3; ++d)
                    h_rot(m, d) = aa[d];
            } else if (jm.contains("orientation")) {
                auto ori = jm["orientation"].get<std::vector<double>>();
                for (int d = 0; d < 3; ++d)
                    h_rot(m, d) = ori[d];
            } else {
                for (int d = 0; d < 3; ++d)
                    h_rot(m, d) = 0.0;
            }
        }
        Kokkos::deep_copy(mol.translations, h_trans);
        Kokkos::deep_copy(mol.rot_vectors, h_rot);
    }

    {
        copy_to_view(mol.sym_rotations, sym_rots);
        copy_to_view(mol.sym_translations, sym_trans);
        copy_to_view(mol.sym_indep_index, sym_indep);

        // Real sites of the molecule each operation produces.
        std::vector<int> site_count(n_symops);
        for (int s = 0; s < n_symops; ++s) {
            const int m = sym_indep[s];
            const int mt = (m >= 0 && m < n_indep)
                         ? type_name_to_idx[data["molecules"][m]["type"].get<std::string>()] : 0;
            site_count[s] = sites_of_type[mt];
        }
        copy_to_view(mol.mol_site_count, site_count);
    }
    validate_symmetry_multiplicity(mol);

    std::vector<std::array<double, 3>> body_positions_all(n_indep * n_sites);
    std::vector<double> body_charges_all(n_indep * n_sites);
    std::vector<int> body_types_all(n_indep * n_sites);
    std::vector<std::array<double, 35>> body_cart_all(n_indep * n_sites);

    // Padded slots stay zero and carry the inert ghost type.
    for (int i = 0; i < n_indep * n_sites; ++i) body_types_all[i] = ghost_type;

    for (int m = 0; m < n_indep; ++m) {
        auto& jm = data["molecules"][m];
        std::string mol_type = jm["type"].get<std::string>();
        int mt_idx = type_name_to_idx[mol_type];
        auto& mt = data["molecule_types"][mt_idx];
        for (int j = 0; j < sites_of_type[mt_idx]; ++j) {
            int idx = m * n_sites + j;
            auto& site = mt["sites"][j];

            // Angstrom to Bohr
            auto pos = site["position"].get<std::vector<double>>();
            double px = pos[0] * ANGSTROM_TO_BOHR;
            double py = pos[1] * ANGSTROM_TO_BOHR;
            double pz = pos[2] * ANGSTROM_TO_BOHR;
            body_positions_all[idx] = {px, py, pz};

            double charge = 0.0;
            if (site.contains("multipoles"))
                charge = site["multipoles"].value("charge", 0.0);
            body_charges_all[idx] = charge;

            std::string atype = site.value("type", "X");
            body_types_all[idx] = type_map[atype];

            // Spherical multipoles, raw e*Bohr^L
            std::vector<double> sph;
            sph.push_back(charge);
            if (site.contains("multipoles") && site["multipoles"].contains("dipole")) {
                for (auto& v : site["multipoles"]["dipole"]) sph.push_back(v.get<double>());
            } else if (max_rank >= 1) {
                sph.insert(sph.end(), 3, 0.0);
            }
            if (site.contains("multipoles") && site["multipoles"].contains("quadrupole")) {
                for (auto& v : site["multipoles"]["quadrupole"]) sph.push_back(v.get<double>());
            } else if (max_rank >= 2) {
                sph.insert(sph.end(), 5, 0.0);
            }
            if (site.contains("multipoles") && site["multipoles"].contains("octupole")) {
                for (auto& v : site["multipoles"]["octupole"]) sph.push_back(v.get<double>());
            } else if (max_rank >= 3) {
                sph.insert(sph.end(), 7, 0.0);
            }
            if (site.contains("multipoles") && site["multipoles"].contains("hexadecapole")) {
                for (auto& v : site["multipoles"]["hexadecapole"]) sph.push_back(v.get<double>());
            } else if (max_rank >= 4) {
                sph.insert(sph.end(), 9, 0.0);
            }

            // Parity is handled by the symop (S_frac = -I), not applied to body data.

            RealType sph_rt[25] = {};
            for (int k = 0; k < static_cast<int>(sph.size()) && k < 25; ++k)
                sph_rt[k] = static_cast<RealType>(sph[k]);
            RealType cart[35];
            spherical_to_cartesian(sph_rt, static_cast<int>(sph.size()),
                                   max_rank, cart);

            for (int c = 0; c < 35; ++c)
                body_cart_all[idx][c] = cart[c];
        }
    }

    // Optional H-bond foreshortening of the body positions.
    double h_ext_ang = 0.0;
    if (data.contains("settings"))
        h_ext_ang = data["settings"].value("h_bond_extend", 0.0);
    if (h_ext_ang != 0.0) {
        const double h_ext_bohr = h_ext_ang * ANGSTROM_TO_BOHR;
        for (int m = 0; m < n_indep; ++m) {
            std::string mol_type = data["molecules"][m]["type"].get<std::string>();
            int mt_idx = type_name_to_idx[mol_type];
            auto& mt = data["molecule_types"][mt_idx];
            const int ns_m = sites_of_type[mt_idx];
            std::vector<std::string> elems(ns_m);
            for (int j = 0; j < ns_m; ++j) {
                std::string el = mt["sites"][j].value("element",
                                  mt["sites"][j].value("type", std::string("C")));
                if (el.size() > 1 && std::islower(static_cast<unsigned char>(el[1])))
                    el = el.substr(0, 2);
                else
                    el = el.substr(0, 1);
                elems[j] = el;
            }
            std::vector<std::array<double, 3>> mol_pos(ns_m);
            for (int j = 0; j < ns_m; ++j)
                mol_pos[j] = body_positions_all[m * n_sites + j];
            apply_h_bond_extend(mol_pos, elems, h_ext_bohr);
            for (int j = 0; j < ns_m; ++j)
                body_positions_all[m * n_sites + j] = mol_pos[j];
        }
    }

    {
        // Already in the view's padded layout (index m*n_sites + j).
        copy_to_view(mol.body_positions, body_positions_all);
        copy_to_view(mol.body_charges, body_charges_all);
        copy_to_view(mol.body_types, body_types_all);
    }

    if (max_rank > 0)
        copy_to_view(mol.body_multipoles, body_cart_all);

    // Direct expansion into the workspace for all molecules.
    {
        RealType cell[3][3];
        cell_params_to_matrix(mol.cell_params, cell);

        auto h_pos = Kokkos::create_mirror_view(ws.positions);
        auto h_q   = Kokkos::create_mirror_view(ws.charges);
        auto h_t   = Kokkos::create_mirror_view(ws.types);
        auto h_mid = Kokkos::create_mirror_view(ws.mol_id);
        auto h_cell = Kokkos::create_mirror_view(ws.cell_matrix);

        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                h_cell(r, c) = cell[r][c];

        auto h_mp = (max_rank > 0) ? Kokkos::create_mirror_view(ws.multipoles)
                                    : decltype(Kokkos::create_mirror_view(ws.multipoles)){};
        auto h_dp = (max_rank > 0) ? Kokkos::create_mirror_view(ws.dipoles)
                                    : decltype(Kokkos::create_mirror_view(ws.dipoles)){};

        for (int m = 0; m < n_indep; ++m) {
            auto& jm = data["molecules"][m];

            RealType R_mol[3][3];
            if (jm.contains("orientation")) {
                auto ori = jm["orientation"].get<std::vector<double>>();
                RealType aa[3] = {static_cast<RealType>(ori[0]), static_cast<RealType>(ori[1]), static_cast<RealType>(ori[2])};
                angle_axis_to_matrix(aa, R_mol);
            } else {
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        R_mol[i][j] = (i == j) ? 1.0 : 0.0;
            }
            // Improper rotation: negate R for parity = -1.
            int parity = jm.value("parity", 1);
            if (parity == -1) {
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        R_mol[i][j] = -R_mol[i][j];
            }

            // Fractional COM to Cartesian
            auto tr = jm["translation"].get<std::vector<double>>();
            double frac[3] = {tr[0], tr[1], tr[2]};
            double com[3] = {0, 0, 0};
            for (int i = 0; i < 3; ++i)
                for (int k = 0; k < 3; ++k)
                    com[i] += frac[k] * cell[i][k];

            for (int j = 0; j < n_sites; ++j) {
                int atom_idx = m * n_sites + j;
                auto& bp = body_positions_all[atom_idx];

                double rotated[3] = {0, 0, 0};
                for (int i = 0; i < 3; ++i)
                    for (int k = 0; k < 3; ++k)
                        rotated[i] += R_mol[i][k] * bp[k];

                h_pos(atom_idx, 0) = rotated[0] + com[0];
                h_pos(atom_idx, 1) = rotated[1] + com[1];
                h_pos(atom_idx, 2) = rotated[2] + com[2];
                h_q(atom_idx) = body_charges_all[atom_idx];
                h_t(atom_idx) = body_types_all[atom_idx];
                h_mid(atom_idx) = m;

                if (max_rank > 0) {
                    RealType body_cart[35], lab_cart[35];
                    for (int c = 0; c < 35; ++c)
                        body_cart[c] = body_cart_all[atom_idx][c];
                    rotate_cartesian_multipole(body_cart, R_mol, max_rank, lab_cart);
                    for (int c = 0; c < 35; ++c)
                        h_mp(atom_idx, c) = lab_cart[c];
                    h_dp(atom_idx, 0) = lab_cart[hermite_index(1, 0, 0)];
                    h_dp(atom_idx, 1) = lab_cart[hermite_index(0, 1, 0)];
                    h_dp(atom_idx, 2) = lab_cart[hermite_index(0, 0, 1)];
                }
            }
        }

        Kokkos::deep_copy(ws.positions, h_pos);
        Kokkos::deep_copy(ws.charges, h_q);
        warn_if_charged(body_charges_all, "structure");
        Kokkos::deep_copy(ws.types, h_t);
        Kokkos::deep_copy(ws.mol_id, h_mid);
        Kokkos::deep_copy(ws.cell_matrix, h_cell);
        ws.n_atoms = n_atoms;

        if (max_rank > 0) {
            Kokkos::deep_copy(ws.multipoles, h_mp);
            Kokkos::deep_copy(ws.dipoles, h_dp);
        }
        ws.max_multipole_rank = max_rank;
    }

    // The record below is generated from the resolved table, not the input.
    PotentialTable resolved_potentials;
    if (data.contains("potentials")) {
        resolved_potentials = resolve_potentials(data["potentials"], type_map);
        fill_workspace_potentials(resolved_potentials, ws);
        warn_potential_gaps(resolved_potentials);
    }

    double pot_cutoff = 15.0;  // default in Angstrom
    if (data.contains("potentials"))
        pot_cutoff = data["potentials"].value("cutoff", 15.0);

    // The spline tapers the energy to zero over this width below the cutoff.
    double spline_width = 2.0;  // Angstrom
    if (data.contains("settings") && data["settings"].value("spline_min", 0.0) > 0.0)
        spline_width = data["settings"]["spline_min"].get<double>();

    double taper_on = (pot_cutoff - spline_width) * ANGSTROM_TO_BOHR;
    double taper_off = pot_cutoff * ANGSTROM_TO_BOHR;

    ws.buck_cutoff = pot_cutoff * ANGSTROM_TO_BOHR;

    if (data.contains("settings")) apply_settings(data["settings"], ws, max_rank);

    TangToenniesDamping buck_damping;
    if (data.contains("potentials"))
        buck_damping = parse_damping(data["potentials"]);

    json basis_out;
    basis_out["molecule_types"] = data["molecule_types"];
    if (data.contains("settings"))
        basis_out["settings"] = data["settings"];
    if (data.contains("potentials")) {
        basis_out["potentials"] = data["potentials"];
        basis_out["potentials"]["buckingham"] = potential_table_to_json(resolved_potentials);
        record_damping(basis_out["potentials"], buck_damping);
    }

    int sg = data.value("space_group", 0);

    // The tabulated model, if requested; built here because it needs the cutoff.
    if (data.contains("potentials")) {
        const TabulatedSpec spec = parse_tabulated_spec(data["potentials"]);
        ws.pair_tables = tabulate_potentials(resolved_potentials, buck_damping,
                                             spec, ws.buck_cutoff);
        if (ws.pair_tables)
            logging::info("tabulated pair potential: {}\n", ws.pair_tables->summary());
    }

    return CrystalConfig{ws.buck_cutoff, taper_on, taper_off, type_map,
                         max_rank, sg, buck_damping, basis_out,
                         std::move(molecule_type_index)};
}

// Shift each H along its bond to the nearest non-H atom within 2.0 Å by
// `extend_bohr` (positive extends, negative shortens).
void apply_h_bond_extend(std::vector<std::array<double, 3>>& positions,
                         const std::vector<std::string>& elements,
                         double extend_bohr) {
    const double max_bond_bohr = 2.0 * ANGSTROM_TO_BOHR;
    const double max_d2 = max_bond_bohr * max_bond_bohr;
    int n = static_cast<int>(positions.size());
    for (int i = 0; i < n; ++i) {
        if (elements[i] != "H") continue;
        int nearest = -1;
        double best_d2 = max_d2;
        for (int j = 0; j < n; ++j) {
            if (j == i || elements[j] == "H") continue;
            double dx = positions[i][0] - positions[j][0];
            double dy = positions[i][1] - positions[j][1];
            double dz = positions[i][2] - positions[j][2];
            double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < best_d2) { best_d2 = d2; nearest = j; }
        }
        if (nearest < 0) continue;
        double d = std::sqrt(best_d2);
        double ux = (positions[i][0] - positions[nearest][0]) / d;
        double uy = (positions[i][1] - positions[nearest][1]) / d;
        double uz = (positions[i][2] - positions[nearest][2]) / d;
        positions[i][0] += extend_bohr * ux;
        positions[i][1] += extend_bohr * uy;
        positions[i][2] += extend_bohr * uz;
    }
}

// Parse molecule body data from a molecule_types JSON array.
// parity: +1 or -1.
static void parse_body_data(const json& molecule_types,
                            int max_rank,
                            const std::map<std::string, int>& type_map,
                            std::vector<std::array<double, 3>>& body_positions,
                            std::vector<double>& body_charges,
                            std::vector<int>& body_types,
                            std::vector<std::array<double, 35>>& body_multipoles,
                            std::vector<std::string>& elements,
                            int parity = 1) {
    auto& mt = molecule_types[0];
    int n_sites = static_cast<int>(mt["sites"].size());
    body_positions.resize(n_sites);
    body_charges.resize(n_sites);
    body_types.resize(n_sites);
    body_multipoles.resize(n_sites);
    elements.resize(n_sites);

    for (int j = 0; j < n_sites; ++j) {
        auto& site = mt["sites"][j];
        warn_site_gaps(site, mt.value("name", std::string("basis")), j);

        // Angstrom to Bohr, with parity
        auto pos = site["position"].get<std::vector<double>>();
        double px = pos[0] * ANGSTROM_TO_BOHR;
        double py = pos[1] * ANGSTROM_TO_BOHR;
        double pz = pos[2] * ANGSTROM_TO_BOHR;
        if (parity == -1) { px = -px; py = -py; pz = -pz; }
        body_positions[j] = {px, py, pz};

        double charge = 0.0;
        if (site.contains("multipoles"))
            charge = site["multipoles"].value("charge", 0.0);
        body_charges[j] = charge;

        std::string atype = site.value("type", "X");
        body_types[j] = type_map.at(atype);

        std::string el = site.value("element", site.value("type", "C"));
        if (el.size() > 1 && std::islower(el[1]))
            el = el.substr(0, 2);
        else
            el = el.substr(0, 1);
        elements[j] = el;

        // Spherical multipoles, raw e*Bohr^L
        std::vector<double> sph;
        sph.push_back(charge);
        if (site.contains("multipoles") && site["multipoles"].contains("dipole")) {
            for (auto& v : site["multipoles"]["dipole"]) sph.push_back(v.get<double>());
        } else if (max_rank >= 1) {
            sph.insert(sph.end(), 3, 0.0);
        }
        if (site.contains("multipoles") && site["multipoles"].contains("quadrupole")) {
            for (auto& v : site["multipoles"]["quadrupole"]) sph.push_back(v.get<double>());
        } else if (max_rank >= 2) {
            sph.insert(sph.end(), 5, 0.0);
        }
        if (site.contains("multipoles") && site["multipoles"].contains("octupole")) {
            for (auto& v : site["multipoles"]["octupole"]) sph.push_back(v.get<double>());
        } else if (max_rank >= 3) {
            sph.insert(sph.end(), 7, 0.0);
        }
        if (site.contains("multipoles") && site["multipoles"].contains("hexadecapole")) {
            for (auto& v : site["multipoles"]["hexadecapole"]) sph.push_back(v.get<double>());
        } else if (max_rank >= 4) {
            sph.insert(sph.end(), 9, 0.0);
        }

        // Parity on spherical multipoles: (-1)^l per rank.
        if (parity == -1) {
            auto signs = multipole_parity_signs(static_cast<int>(sph.size()));
            for (int k = 0; k < static_cast<int>(sph.size()); ++k)
                sph[k] *= signs[k];
        }

        RealType sph_rt[25] = {};
        for (int k = 0; k < static_cast<int>(sph.size()) && k < 25; ++k)
            sph_rt[k] = static_cast<RealType>(sph[k]);
        RealType cart[35];
        spherical_to_cartesian(sph_rt, static_cast<int>(sph.size()),
                               max_rank, cart);

        for (int c = 0; c < 35; ++c)
            body_multipoles[j][c] = cart[c];
    }
}

// Parse potentials from JSON and fill the workspace Buckingham tables.
static void parse_potentials(const json& jpot, const json& jsettings,
                             const std::map<std::string, int>& type_map,
                             int n_types, Workspace& ws,
                             double& cutoff, double& taper_on, double& taper_off,
                             PotentialTable& out_table) {
    double pot_cutoff = jpot.value("cutoff", 15.0);
    cutoff = pot_cutoff * ANGSTROM_TO_BOHR;

    // The resolved table is returned so the caller can record it.
    out_table = resolve_potentials(jpot, type_map);
    fill_workspace_potentials(out_table, ws);
    warn_potential_gaps(out_table);
    ws.buck_cutoff = cutoff;

    const TabulatedSpec spec = parse_tabulated_spec(jpot);
    ws.pair_tables = tabulate_potentials(out_table, parse_damping(jpot),
                                         spec, cutoff);
    if (ws.pair_tables)
        logging::info("tabulated pair potential: {}\n", ws.pair_tables->summary());

    double spline_width = 2.0;
    if (!jsettings.is_null() && jsettings.value("spline_min", 0.0) > 0.0)
        spline_width = jsettings["spline_min"].get<double>();

    taper_on = (pot_cutoff - spline_width) * ANGSTROM_TO_BOHR;
    taper_off = pot_cutoff * ANGSTROM_TO_BOHR;
}

// h_bond_extend and spline_min are applied by their own call sites; they appear
// here only to keep them out of the ignored set.
InputSettings parse_settings(const json& s) {
    InputSettings out;
    if (!s.is_object()) return out;

    if (s.contains("ewald_accuracy")) {
        const auto& v = s["ewald_accuracy"];
        const double acc = v.is_number() ? v.get<double>() : 0.0;
        if (acc > 0.0 && acc < 1.0)
            out.ewald_accuracy = static_cast<RealType>(acc);
        else
            out.warnings.push_back(fmt::format(
                "settings.ewald_accuracy is not a relative tolerance in (0, 1); "
                "keeping {:g}", double(out.ewald_accuracy)));
    }

    // A cap, never a promotion: only lowers the rank the sites carry.
    // --max-order overrides this in either direction.
    if (s.contains("max_interaction_order")) {
        const auto& v = s["max_interaction_order"];
        if (v.is_number_integer() && v.get<int>() >= 0)
            out.max_rank_cap = v.get<int>();
        else
            out.warnings.push_back(
                "settings.max_interaction_order is not a multipole rank; ignoring it");
    }

    static const std::set<std::string> understood = {
        "ewald_accuracy", "max_interaction_order", "h_bond_extend", "spline_min"};
    static const std::map<std::string, std::string> known_ignored = {
        {"use_ewald",          "klasp always uses Ewald for the electrostatics"},
        {"neighbor_radius",    "the real-space range comes from the potential cutoff and kappa"},
        {"gradient_tolerance", "set it with optimizer.grad_tol (--grad-tol)"},
        {"max_iterations",     "set it with optimizer.n_iterations (--iterations)"},
    };
    for (auto it = s.begin(); it != s.end(); ++it) {
        if (understood.count(it.key())) continue;
        auto known = known_ignored.find(it.key());
        out.warnings.push_back(
            known != known_ignored.end()
                ? fmt::format("settings.{} is ignored -- {}", it.key(), known->second)
                : fmt::format("settings.{} is not a setting klasp reads", it.key()));
    }
    return out;
}

void apply_ewald_accuracy_override(double accuracy, Workspace& ws) {
    if (accuracy < 0.0) return;
    if (!(accuracy > 0.0 && accuracy < 1.0))
        throw std::runtime_error(fmt::format(
            "--ewald-accuracy {} is not a relative tolerance in (0, 1)", accuracy));
    ws.ewald_accuracy = static_cast<RealType>(accuracy);
}

// Warn once per distinct message; the predicate lives in io.hpp.
static void warn_if_charged(const std::vector<double>& site_charges,
                            const char* what) {
    if (cell_is_neutral(site_charges)) return;
    static std::set<std::string> already_said;
    const std::string msg = fmt::format(
        "{}: the asymmetric unit carries a net charge of {:+.4g} e. A crystal "
        "cell must be neutral -- Ewald is only conditionally convergent "
        "otherwise and no compensating background is applied, so the lattice "
        "energy is not meaningful.", what, net_site_charge(site_charges));
    if (already_said.insert(msg).second)
        logging::warn("warning: {}\n", msg);
}

static void apply_settings(const json& s, Workspace& ws, int& max_rank) {
    const InputSettings parsed = parse_settings(s);
    ws.ewald_accuracy = parsed.ewald_accuracy;
    if (parsed.max_rank_cap >= 0 && parsed.max_rank_cap < max_rank)
        max_rank = parsed.max_rank_cap;

    static std::set<std::string> already_said;
    for (const auto& w : parsed.warnings)
        if (already_said.insert(w).second)
            logging::warn("warning: {}\n", w);
}

// Put the resolved potential, including both damping keys, back into the JSON
// that is recorded, so the record describes what the kernels were given.
static void record_resolved_potentials(BasisConfig& cfg) {
    if (!cfg.basis_json.contains("potentials")) return;
    cfg.basis_json["potentials"]["buckingham"] = potential_table_to_json(cfg.potentials);
    record_damping(cfg.basis_json["potentials"], cfg.buck_damping);

    static std::set<std::string> already_said;
    for (const auto& pair : cfg.potentials.derived)
        if (already_said.insert(pair).second)
            logging::warn("Deriving Buckingham ({}) via Good-Hope combining rule "
                               "(A=sqrt(Aii*Ajj), B=(Bii+Bjj)/2, C=sqrt(Cii*Cjj))\n", pair);
    if (!cfg.potentials.derived.empty())
        logging::warn("Note: {} cross-type Buckingham pair(s) derived via combining rules and "
                   "recorded in the run's basis. For production CSP, provide a unified "
                   "potential file listing all cross pairs explicitly (W99 self-pair "
                   "combining is not always quantitative).\n", cfg.potentials.derived.size());
}

static int detect_max_rank(const json& molecule_types) {
    int max_rank = 0;
    for (auto& mt : molecule_types) {
        for (auto& site : mt["sites"]) {
            if (!site.contains("multipoles")) continue;
            auto& mp = site["multipoles"];
            if (mp.contains("hexadecapole")) max_rank = std::max(max_rank, 4);
            else if (mp.contains("octupole")) max_rank = std::max(max_rank, 3);
            else if (mp.contains("quadrupole")) max_rank = std::max(max_rank, 2);
            else if (mp.contains("dipole")) max_rank = std::max(max_rank, 1);
        }
    }
    return max_rank;
}

static std::map<std::string, int> build_type_map(const json& molecule_types) {
    std::map<std::string, int> type_map;
    int next_type = 0;
    for (auto& mt : molecule_types) {
        for (auto& site : mt["sites"]) {
            std::string atype = site.value("type", "X");
            if (type_map.find(atype) == type_map.end())
                type_map[atype] = next_type++;
        }
    }
    return type_map;
}

BasisConfig load_basis_json(const std::string& path, Workspace& ws) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + path);
    return load_basis_document(json::parse(f), ws, path);
}

BasisConfig load_basis_document(const json& doc, Workspace& ws, const std::string& source) {
    json data = doc;

    if (!data.contains("basis"))
        throw std::runtime_error("Expected 'basis' key in " + source);

    auto& basis_json = data["basis"];

    auto& molecule_types = basis_json["molecule_types"];
    int n_sites = static_cast<int>(molecule_types[0]["sites"].size());
    int max_rank = detect_max_rank(molecule_types);
    auto type_map = build_type_map(molecule_types);
    int n_types = static_cast<int>(type_map.size());

    if (basis_json.contains("settings"))
        apply_settings(basis_json["settings"], ws, max_rank);

    BasisConfig cfg;
    cfg.max_rank = max_rank;
    cfg.n_sites = n_sites;
    cfg.n_types = n_types;
    cfg.type_map = type_map;
    cfg.basis_json = basis_json;
    if (basis_json.contains("potentials"))
        cfg.buck_damping = parse_damping(basis_json["potentials"]);

    parse_body_data(molecule_types, max_rank, type_map,
                    cfg.body_positions, cfg.body_charges, cfg.body_types,
                    cfg.body_multipoles, cfg.elements);

    // Optional H-bond foreshortening/extension (W99 / neighcrys convention).
    if (basis_json.contains("settings")) {
        double h_ext = basis_json["settings"].value("h_bond_extend", 0.0);
        if (h_ext != 0.0)
            apply_h_bond_extend(cfg.body_positions, cfg.elements,
                                h_ext * ANGSTROM_TO_BOHR);
    }

    json jsettings = basis_json.contains("settings") ? basis_json["settings"] : json(nullptr);
    if (basis_json.contains("potentials")) {
        parse_potentials(basis_json["potentials"], jsettings, type_map, n_types, ws,
                         cfg.cutoff, cfg.taper_on, cfg.taper_off, cfg.potentials);
        record_resolved_potentials(cfg);
    } else {
        cfg.cutoff = 15.0 * ANGSTROM_TO_BOHR;
        cfg.taper_on = 13.0 * ANGSTROM_TO_BOHR;
        cfg.taper_off = 15.0 * ANGSTROM_TO_BOHR;
        ws.buck_cutoff = cfg.cutoff;
        ws.n_types = n_types;
    }

    warn_if_charged(cfg.body_charges, "basis");
    return cfg;
}

BasisConfig load_basis_json_multi(const std::vector<std::string>& paths, Workspace& ws) {
    std::vector<json> docs;
    for (const auto& path : paths) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open " + path);
        docs.push_back(json::parse(f));
    }
    return load_basis_documents(docs, ws, paths);
}

BasisConfig load_basis_documents(const std::vector<json>& docs, Workspace& ws,
                                 const std::vector<std::string>& sources) {
    const auto source = [&](size_t m) {
        return m < sources.size() ? sources[m] : "basis document " + std::to_string(m + 1);
    };
    if (docs.size() == 1)
        return load_basis_document(docs[0], ws, source(0));

    // The first file supplies the potentials and base config.
    auto cfg = load_basis_document(docs[0], ws, source(0));

    int n_mols = static_cast<int>(docs.size());
    cfg.n_mols = n_mols;
    cfg.mol_n_sites.resize(n_mols);
    cfg.mol_body_positions.resize(n_mols);
    cfg.mol_body_charges.resize(n_mols);
    cfg.mol_body_types.resize(n_mols);
    cfg.mol_body_multipoles.resize(n_mols);
    cfg.mol_elements.resize(n_mols);

    cfg.mol_n_sites[0] = cfg.n_sites;
    cfg.mol_body_positions[0] = cfg.body_positions;
    cfg.mol_body_charges[0] = cfg.body_charges;
    cfg.mol_body_types[0] = cfg.body_types;
    cfg.mol_body_multipoles[0] = cfg.body_multipoles;
    cfg.mol_elements[0] = cfg.elements;

    int max_sites = cfg.n_sites;

    // Additional molecules; their potentials are ignored.
    for (int m = 1; m < n_mols; ++m) {
        auto ws_tmp = allocate_workspace(8, 27, 100, 32, 4);
        auto mol_cfg = load_basis_document(docs[m], ws_tmp, source(m));

        // Map local (mol_cfg) type indices to merged (cfg) ones.
        std::vector<int> local_to_merged(mol_cfg.type_map.size(), -1);
        for (auto& [name, local_id] : mol_cfg.type_map) {
            int merged_id;
            auto it = cfg.type_map.find(name);
            if (it == cfg.type_map.end()) {
                merged_id = static_cast<int>(cfg.type_map.size());
                cfg.type_map[name] = merged_id;
            } else {
                merged_id = it->second;
            }
            local_to_merged[local_id] = merged_id;
        }

        cfg.mol_n_sites[m] = mol_cfg.n_sites;
        cfg.mol_body_positions[m] = mol_cfg.body_positions;
        cfg.mol_body_charges[m] = mol_cfg.body_charges;
        cfg.mol_body_types[m].resize(mol_cfg.body_types.size());
        for (size_t i = 0; i < mol_cfg.body_types.size(); ++i)
            cfg.mol_body_types[m][i] = local_to_merged[mol_cfg.body_types[i]];
        cfg.mol_body_multipoles[m] = mol_cfg.body_multipoles;
        cfg.mol_elements[m] = mol_cfg.elements;
        cfg.max_rank = std::max(cfg.max_rank, mol_cfg.max_rank);
        max_sites = std::max(max_sites, mol_cfg.n_sites);

        // Give each molecule type a unique name; files default to "mol".
        if (mol_cfg.basis_json.contains("molecule_types")) {
            for (auto& mt : mol_cfg.basis_json["molecule_types"]) {
                json mt_copy = mt;
                mt_copy["name"] = fmt::format("mol_{}", m);
                cfg.basis_json["molecule_types"].push_back(mt_copy);
            }
        }

        // Merge Buckingham potentials (union, dedup by (type_i, type_j)).
        if (mol_cfg.basis_json.contains("potentials") &&
            mol_cfg.basis_json["potentials"].contains("buckingham")) {
            auto& tgt = cfg.basis_json["potentials"]["buckingham"];
            auto existing_key = [](const json& p) {
                auto t = p["types"].get<std::vector<std::string>>();
                if (t.size() < 2) return std::string();
                return t[0] < t[1] ? t[0] + "|" + t[1] : t[1] + "|" + t[0];
            };
            std::set<std::string, std::less<>> have;
            for (auto& p : tgt) have.insert(existing_key(p));
            for (auto& p : mol_cfg.basis_json["potentials"]["buckingham"]) {
                std::string k = existing_key(p);
                if (!k.empty() && have.insert(k).second)
                    tgt.push_back(p);
            }
        }
    }

    // Rename the first type too, for consistent naming.
    if (cfg.basis_json.contains("molecule_types") && !cfg.basis_json["molecule_types"].empty()) {
        cfg.basis_json["molecule_types"][0]["name"] = "mol_0";
    }

    cfg.n_sites = max_sites;
    cfg.n_types = static_cast<int>(cfg.type_map.size());

    // One resolve over the merged list fills in the cross-component pairs.
    {
        json jsettings = cfg.basis_json.contains("settings")
                       ? cfg.basis_json["settings"] : json(nullptr);
        parse_potentials(cfg.basis_json["potentials"], jsettings, cfg.type_map,
                         cfg.n_types, ws, cfg.cutoff, cfg.taper_on, cfg.taper_off,
                         cfg.potentials);
        record_resolved_potentials(cfg);
    }


    // Pad all molecules to max_sites and build flat body data.
    int total_body = n_mols * max_sites;
    cfg.body_positions.resize(total_body, {0.0, 0.0, 0.0});
    cfg.body_charges.resize(total_body, 0.0);
    // Padding sites carry the inert ghost type.
    cfg.body_types.resize(total_body, cfg.potentials.ghost_type());
    cfg.body_multipoles.resize(total_body);
    for (auto& mp : cfg.body_multipoles) mp.fill(0.0);
    cfg.elements.resize(total_body, "X");

    for (int m = 0; m < n_mols; ++m) {
        int ns = cfg.mol_n_sites[m];
        int offset = m * max_sites;
        for (int i = 0; i < ns; ++i) {
            cfg.body_positions[offset + i] = cfg.mol_body_positions[m][i];
            cfg.body_charges[offset + i] = cfg.mol_body_charges[m][i];
            cfg.body_types[offset + i] = cfg.mol_body_types[m][i];
            cfg.body_multipoles[offset + i] = cfg.mol_body_multipoles[m][i];
            cfg.elements[offset + i] = cfg.mol_elements[m][i];
        }
        // Ghost sites (ns..max_sites) keep zeros.
    }

    warn_if_charged(cfg.body_charges, "basis");
    return cfg;
}

MoleculeData build_csp_molecule(const BasisConfig& basis, int space_group) {
    auto& sg_info = get_space_group(space_group);
    int n_crystal_ops = sg_info.n_symops;
    int n_mols = basis.n_mols;

    // n_symops = n_crystal_ops * n_mols, laid out
    // [mol_0 op_0, mol_1 op_0, ..., mol_0 op_1, mol_1 op_1, ...], identity first.
    int n_symops = n_crystal_ops * n_mols;

    auto csp_mol = allocate_molecule_data(n_mols, basis.n_sites, n_symops, basis.max_rank);
    csp_mol.n_sites_per_mol = basis.n_sites;  // n_indep/n_symops come from fill_symmetry_ops

    copy_to_view(csp_mol.body_positions, basis.body_positions);
    copy_to_view(csp_mol.body_charges, basis.body_charges);
    copy_to_view(csp_mol.body_types, basis.body_types);
    copy_to_view(csp_mol.body_multipoles, basis.body_multipoles);

    // Identity first per molecule; the generator orders its copy through the same
    // function (klasp/symmetry_ops.hpp).
    fill_symmetry_ops(csp_mol.symmetry(), sg_info, n_crystal_ops, n_mols,
                      basis.mol_n_sites, basis.n_sites);
    validate_symmetry_multiplicity(csp_mol);

    return csp_mol;
}

// Symop rotation + translation as a CIF-style string.
static std::string symop_to_string(const double rot[3][3], const double trans[3]) {
    const char* xyz[] = {"x", "y", "z"};
    std::string result;
    for (int i = 0; i < 3; ++i) {
        if (i > 0) result += ", ";
        std::string term;
        for (int j = 0; j < 3; ++j) {
            int r = static_cast<int>(std::round(rot[i][j]));
            if (r == 0) continue;
            if (r == 1) {
                if (!term.empty()) term += "+";
                term += xyz[j];
            } else if (r == -1) {
                term += "-";
                term += xyz[j];
            }
        }
        double t = trans[i];
        if (std::abs(t) > 1e-10) {
            // As a fraction if close to a common value.
            struct { double val; const char* str; } fracs[] = {
                {1.0/6, "1/6"}, {1.0/4, "1/4"}, {1.0/3, "1/3"},
                {1.0/2, "1/2"}, {2.0/3, "2/3"}, {3.0/4, "3/4"},
                {5.0/6, "5/6"}
            };
            const char* frac_str = nullptr;
            for (auto& fr : fracs) {
                if (std::abs(t - fr.val) < 1e-10) { frac_str = fr.str; break; }
            }
            if (frac_str) {
                if (!term.empty()) term += "+";
                term += frac_str;
            }
        }
        if (term.empty()) term = "0";
        result += term;
    }
    return result;
}

nlohmann::json build_crystal_json(const BasisConfig& basis,
                                  int space_group,
                                  const double cell_params[6],
                                  const double* translations,
                                  const double* rot_vectors,
                                  int n_indep,
                                  double energy) {
    json j;

    // Cell: Bohr to Angstrom
    j["cell"] = {
        {"a", cell_params[0] * BOHR_TO_ANGSTROM},
        {"b", cell_params[1] * BOHR_TO_ANGSTROM},
        {"c", cell_params[2] * BOHR_TO_ANGSTROM},
        {"alpha", cell_params[3]},
        {"beta", cell_params[4]},
        {"gamma", cell_params[5]}
    };

    // molecule_types from the original basis JSON (already in Angstrom).
    j["molecule_types"] = basis.basis_json["molecule_types"];

    j["molecules"] = json::array();
    auto& mol_types = basis.basis_json["molecule_types"];
    for (int m = 0; m < n_indep; ++m) {
        json jm;
        int type_idx = (m < static_cast<int>(mol_types.size())) ? m : 0;
        jm["type"] = mol_types[type_idx]["name"].get<std::string>();
        jm["translation"] = {translations[m*3], translations[m*3+1], translations[m*3+2]};
        // Orientation only if non-zero.
        double rv[3] = {rot_vectors[m*3], rot_vectors[m*3+1], rot_vectors[m*3+2]};
        double rv_norm = rv[0]*rv[0] + rv[1]*rv[1] + rv[2]*rv[2];
        if (rv_norm > 1e-20)
            jm["orientation"] = {rv[0], rv[1], rv[2]};
        j["molecules"].push_back(jm);
    }

    // Space group number, so readers can pick the strain mask directly.
    j["space_group"] = space_group;

    if (space_group > 1) {
        auto& sg_info = get_space_group(space_group);
        j["symmetry"] = json::array();

        // The order the operators were applied in.
        const std::vector<int> order = identity_first_order(sg_info);

        for (int oi = 0; oi < sg_info.n_symops; ++oi) {
            double rot[3][3], trans[3];
            get_symop(sg_info, order[oi], rot, trans);
            for (int m = 0; m < n_indep; ++m) {
                json se;
                se["op"] = symop_to_string(rot, trans);
                se["molecule"] = m;
                j["symmetry"].push_back(se);
            }
        }
    }

    if (basis.basis_json.contains("potentials"))
        j["potentials"] = basis.basis_json["potentials"];

    if (basis.basis_json.contains("settings"))
        j["settings"] = basis.basis_json["settings"];

    if (energy != 0.0) {
        j["reference"] = {{"total", energy}};
    }

    return j;
}

void write_crystal_json(const std::string& path,
                        const BasisConfig& basis,
                        int space_group,
                        const double cell_params[6],
                        const double* translations,
                        const double* rot_vectors,
                        int n_indep,
                        double energy) {
    const json j = build_crystal_json(basis, space_group, cell_params, translations,
                                      rot_vectors, n_indep, energy);
    std::ofstream out(path);
    if (!out.is_open())
        throw std::runtime_error("Cannot write to " + path);
    out << j.dump(2) << "\n";
}

json build_occ_json(const BasisConfig& basis,
                    const MoleculeData& mol,
                    double energy_per_mol,
                    const std::vector<int>& molecule_type_index) {
    json j;

    // Cell: Bohr to Angstrom
    j["crystal"] = json::object();
    j["crystal"]["cell"] = {
        {"a", mol.cell_params[0] * BOHR_TO_ANGSTROM},
        {"b", mol.cell_params[1] * BOHR_TO_ANGSTROM},
        {"c", mol.cell_params[2] * BOHR_TO_ANGSTROM},
        {"alpha", mol.cell_params[3]},
        {"beta", mol.cell_params[4]},
        {"gamma", mol.cell_params[5]}
    };

    j["basis"] = json::object();
    j["basis"]["molecule_types"] = basis.basis_json["molecule_types"];
    if (basis.basis_json.contains("potentials"))
        j["basis"]["potentials"] = basis.basis_json["potentials"];
    if (basis.basis_json.contains("settings"))
        j["basis"]["settings"] = basis.basis_json["settings"];

    // Expand all molecules using stored symops. One type name per symop copy,
    // from its independent molecule.
    std::vector<std::string> type_names;
    for (const auto& mt : basis.basis_json["molecule_types"])
        type_names.push_back(mt["name"].get<std::string>());

    RealType cm[3][3], ci[3][3] = {};
    cell_params_to_matrix(mol.cell_params, cm);
    mat3_inv(cm, ci);  // ci stays zero on a degenerate cell

    auto h_trans = to_host(mol.translations);
    auto h_rots = to_host(mol.rot_vectors);
    auto h_sr = to_host(mol.sym_rotations);
    auto h_st = to_host(mol.sym_translations);
    auto h_si = to_host(mol.sym_indep_index);

    j["crystal"]["molecules"] = json::array();

    for (int op = 0; op < mol.n_symops; ++op) {
        int k = h_si(op);  // independent molecule index

        RealType S_frac[3][3];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                S_frac[r][c] = h_sr(op, r*3+c);

        // Fractional COM: S_frac @ frac_trans + t_sym, unwrapped to preserve exact
        // symmetry relations.
        double ft[3] = {h_trans(k,0), h_trans(k,1), h_trans(k,2)};
        double frac_com[3];
        for (int r = 0; r < 3; ++r) {
            frac_com[r] = S_frac[r][0]*ft[0] + S_frac[r][1]*ft[1] + S_frac[r][2]*ft[2]
                        + h_st(op, r);
        }

        // S_cart = cell @ S_frac @ cell_inv
        RealType S_cart[3][3];
        frac_to_cart_symop(cm, ci, S_frac, S_cart);

        RealType aa[3] = {(RealType)h_rots(k,0), (RealType)h_rots(k,1), (RealType)h_rots(k,2)};
        RealType R_mol[3][3];
        angle_axis_to_matrix(aa, R_mol);

        // R_total = S_cart @ R_mol
        RealType R_total[3][3];
        mat3_mul(S_cart, R_mol, R_total);

        json jm;
        const int type = (k >= 0 && k < static_cast<int>(molecule_type_index.size()))
                             ? molecule_type_index[k]
                             : (k >= 0 && k < static_cast<int>(type_names.size())) ? k : 0;
        jm["type"] = type_names[type];
        jm["translation"] = {frac_com[0], frac_com[1], frac_com[2]};
        // Full 3x3 rotation matrix (proper or improper).
        jm["rotation"] = {{R_total[0][0], R_total[0][1], R_total[0][2]},
                          {R_total[1][0], R_total[1][1], R_total[1][2]},
                          {R_total[2][0], R_total[2][1], R_total[2][2]}};
        j["crystal"]["molecules"].push_back(jm);
    }

    if (energy_per_mol != 0.0) {
        j["reference"] = {{"total", energy_per_mol}};
    }
    return j;
}

void write_occ_json(const std::string& path,
                    const BasisConfig& basis,
                    const MoleculeData& mol,
                    double energy_per_mol,
                    const std::vector<int>& molecule_type_index) {
    const json j = build_occ_json(basis, mol, energy_per_mol, molecule_type_index);
    std::ofstream out(path);
    if (!out.is_open())
        throw std::runtime_error("Cannot write to " + path);
    out << j.dump(2) << "\n";
}

namespace {

// Add the CSP dictionary items to a block; gemmi does the quoting.
void apply_csp_metadata(gemmi::cif::Block& block, const CspCifMetadata& csp) {
    for (const auto& [name, value] : csp_cif_pairs(csp))
        block.set_pair(name, gemmi::cif::quote(value));
    const CspCifLoop loop = csp_cif_optimisation_loop(csp);
    if (loop.rows.empty()) return;
    gemmi::cif::Loop& l = block.init_loop("", loop.names);
    for (const auto& row : loop.rows) {
        std::vector<std::string> quoted;
        quoted.reserve(row.size());
        for (const auto& v : row) quoted.push_back(gemmi::cif::quote(v));
        l.add_row(std::move(quoted));
    }
}

}  // namespace

// The block name's fields as data items, under klasp-private names: the CCDC
// CSP dictionary (v0.8) defines none for a seed or stage.
static void write_structure_id(gemmi::cif::Block& block, const CspStructureId& id) {
    if (!id.name.empty())      block.set_pair("_klasp_name", gemmi::cif::quote(id.name));
    if (!id.generator.empty()) block.set_pair("_klasp_generator", id.generator);
    if (id.seed >= 0)          block.set_pair("_klasp_seed", fmt::format("{}", id.seed));
    if (id.stage >= 0)         block.set_pair("_klasp_stage", fmt::format("{}", id.stage));
}

bool write_cif(std::ostream& out,
               const BasisConfig& basis,
               int space_group,
               const double cell_params[6],
               const double* translations,
               const double* rot_vectors,
               int n_indep,
               double energy,
               const CspStructureId& id,
               bool niggli,
               const CspCifMetadata* csp) {
    // Place the structure in its optimized cell and carry only the fractional
    // coordinates into the reduced basis at the end.
    int P[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    double out_params[6];
    for (int i = 0; i < 6; ++i) out_params[i] = cell_params[i];
    bool reduced = false;
    if (niggli && niggli_applicable(space_group)) {
        RealType pin[6], pout[6];
        for (int i = 0; i < 6; ++i) pin[i] = static_cast<RealType>(cell_params[i]);
        if (niggli_reduce_params(pin, pout, P)) {
            for (int i = 0; i < 6; ++i) out_params[i] = static_cast<double>(pout[i]);
            reduced = true;
        }
    }

    // Cell params: Bohr to Angstrom
    double a = out_params[0] * BOHR_TO_ANGSTROM;
    double b = out_params[1] * BOHR_TO_ANGSTROM;
    double c = out_params[2] * BOHR_TO_ANGSTROM;
    double alpha = out_params[3], beta = out_params[4], gamma = out_params[5];

    // The block is named for its database row.
    gemmi::cif::Document doc;
    CspStructureId key = id;
    if (key.space_group <= 0) key.space_group = space_group;
    auto& block = doc.add_new_block(key.block_name());
    write_structure_id(block, key);

    block.set_pair("_cell_length_a", fmt::format("{:.6f}", a));
    block.set_pair("_cell_length_b", fmt::format("{:.6f}", b));
    block.set_pair("_cell_length_c", fmt::format("{:.6f}", c));
    block.set_pair("_cell_angle_alpha", fmt::format("{:.4f}", alpha));
    block.set_pair("_cell_angle_beta", fmt::format("{:.4f}", beta));
    block.set_pair("_cell_angle_gamma", fmt::format("{:.4f}", gamma));

    const double cp_ang[6] = {a, b, c, alpha, beta, gamma};
    const double vol = cell_volume_from_params(cp_ang);

    // Molecular mass of one formula unit (all molecule types).
    double mol_mass = 0;
    auto& mol_types = basis.basis_json["molecule_types"];
    for (int mt = 0; mt < n_indep && mt < static_cast<int>(mol_types.size()); ++mt)
        for (auto& site : mol_types[mt]["sites"])
            mol_mass += element(site.value("element", std::string("C"))).mass;

    auto& sg_info = get_space_group(space_group);
    int Z = sg_info.n_symops;  // formula units per cell
    double density = Z * mol_mass * AMU_TO_GRAM / (vol * 1e-24);

    block.set_pair("_cell_volume", fmt::format("{:.4f}", vol));
    block.set_pair("_cell_formula_units_Z", fmt::format("{}", Z));
    block.set_pair("_exptl_crystal_density_diffrn", fmt::format("{:.4f}", density));

    // Space group (gemmi for H-M symbol and crystal system)
    auto* gemmi_sg = gemmi::find_spacegroup_by_number(space_group);
    block.set_pair("_space_group_IT_number", fmt::format("{}", space_group));
    if (gemmi_sg) {
        block.set_pair("_space_group_name_H-M_alt", fmt::format("'{}'", gemmi_sg->hm));
        // _space_group_* supersedes the archival _symmetry_* names.
        block.set_pair("_space_group_crystal_system", gemmi_sg->crystal_system_str());
    }

    // Chemical formula of the formula unit (all molecule types).
    std::map<std::string, int> formula_map;
    for (int mt = 0; mt < n_indep && mt < static_cast<int>(mol_types.size()); ++mt)
        for (auto& site : mol_types[mt]["sites"]) {
            std::string el = site.value("element", "C");
            formula_map[el]++;
        }
    std::string formula;
    for (auto& [el, count] : formula_map) {
        formula += el;
        if (count > 1) formula += std::to_string(count);
        formula += " ";
    }
    if (!formula.empty()) formula.pop_back();
    block.set_pair("_chemical_formula_sum", fmt::format("'{}'", formula));

    block.set_pair("_klasp_energy_kj_per_mol", fmt::format("{:.6f}", energy));

    // CSP provenance, before the atom loop.
    if (csp) apply_csp_metadata(block, *csp);

    // The original cell: the rotation vectors are relative to it.
    RealType cp_rt[6];
    for (int i = 0; i < 6; ++i) cp_rt[i] = static_cast<RealType>(cell_params[i]);
    RealType cell[3][3];
    cell_params_to_matrix(cp_rt, cell);
    RealType ci[3][3];
    mat3_inv(cell, ci);

    // Expand atoms: rotate body positions, add COM, convert to fractional.
    auto& loop = block.init_loop("_atom_site_", {"label", "type_symbol",
        "fract_x", "fract_y", "fract_z"});

    for (int m = 0; m < n_indep; ++m) {
        RealType R[3][3];
        RealType aa[3] = {(RealType)rot_vectors[m*3], (RealType)rot_vectors[m*3+1], (RealType)rot_vectors[m*3+2]};
        angle_axis_to_matrix(aa, R);

        double frac_com[3] = {translations[m*3], translations[m*3+1], translations[m*3+2]};
        double cart_com[3] = {0, 0, 0};
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k)
                cart_com[i] += frac_com[k] * cell[i][k];

        int body_off = m * basis.n_sites;  // offset for this molecule type
        int ns_m = (m < static_cast<int>(basis.mol_n_sites.size()))
                 ? basis.mol_n_sites[m] : basis.n_sites;
        for (int j = 0; j < ns_m; ++j) {
            double rotated[3] = {0, 0, 0};
            for (int i = 0; i < 3; ++i)
                for (int k = 0; k < 3; ++k)
                    rotated[i] += R[i][k] * basis.body_positions[body_off + j][k];

            double cart[3];
            for (int i = 0; i < 3; ++i)
                cart[i] = rotated[i] + cart_com[i];

            double frac[3] = {0, 0, 0};
            for (int i = 0; i < 3; ++i)
                for (int k = 0; k < 3; ++k)
                    frac[i] += ci[i][k] * cart[k];
            if (reduced) {
                double mapped[3];
                niggli_map_fractional(P, frac, mapped);
                for (int i = 0; i < 3; ++i) frac[i] = mapped[i];
            }

            std::string el = basis.elements[body_off + j];
            std::string label = fmt::format("{}{}_{}", el, j+1, m+1);
            loop.add_row({label, el,
                          fmt::format("{:.8f}", frac[0]),
                          fmt::format("{:.8f}", frac[1]),
                          fmt::format("{:.8f}", frac[2])});
        }
    }

    // Symops in the internal convention (sgdata.hpp), matching
    // expand_asymmetric_unit.
    auto& sym_loop = block.init_loop("_space_group_symop_", {"id", "operation_xyz"});
    {
        const std::vector<int> order = identity_first_order(sg_info);
        for (int oi = 0; oi < sg_info.n_symops; ++oi) {
            double rot[3][3], trans[3];
            get_symop(sg_info, order[oi], rot, trans);
            sym_loop.add_row({std::to_string(oi + 1), "'" + symop_to_string(rot, trans) + "'"});
        }
    }

    gemmi::cif::write_cif_to_stream(out, doc);
    return reduced;
}

bool write_cif(const std::string& path,
               const BasisConfig& basis,
               int space_group,
               const double cell_params[6],
               const double* translations,
               const double* rot_vectors,
               int n_indep,
               double energy,
               const CspStructureId& id,
               bool niggli,
               const CspCifMetadata* csp) {
    std::ofstream out(path);
    return write_cif(out, basis, space_group, cell_params, translations,
                     rot_vectors, n_indep, energy, id, niggli, csp);
}

void write_cif_explicit(std::ostream& out,
                        const double cell_params[6],
                        int n_atoms,
                        const int* Z,
                        const double* frac,
                        int predicted_space_group,
                        double energy,
                        const CspStructureId& id,
                        const CspCifMetadata* csp) {
    gemmi::cif::Document doc;
    CspStructureId key = id;
    if (key.space_group <= 0) key.space_group = predicted_space_group;
    auto& block = doc.add_new_block(key.block_name());
    write_structure_id(block, key);

    // P1: the space group's operations are already applied. The predicted group
    // is recorded under a klasp data name.
    block.set_pair("_space_group_IT_number", "1");
    block.set_pair("_space_group_name_H-M_alt", gemmi::cif::quote("P 1"));
    block.set_pair("_klasp_predicted_space_group_IT_number",
                   fmt::format("{}", predicted_space_group));
    if (const auto* sg = gemmi::find_spacegroup_by_number(predicted_space_group))
        block.set_pair("_klasp_predicted_space_group_name_H-M",
                       gemmi::cif::quote(sg->hm));

    block.set_pair("_cell_length_a", fmt::format("{:.6f}", cell_params[0]));
    block.set_pair("_cell_length_b", fmt::format("{:.6f}", cell_params[1]));
    block.set_pair("_cell_length_c", fmt::format("{:.6f}", cell_params[2]));
    block.set_pair("_cell_angle_alpha", fmt::format("{:.4f}", cell_params[3]));
    block.set_pair("_cell_angle_beta", fmt::format("{:.4f}", cell_params[4]));
    block.set_pair("_cell_angle_gamma", fmt::format("{:.4f}", cell_params[5]));
    block.set_pair("_cell_volume", fmt::format("{:.4f}", cell_volume_from_params(cell_params)));
    block.set_pair("_klasp_energy_kj_per_mol", fmt::format("{:.6f}", energy));
    if (csp) apply_csp_metadata(block, *csp);

    auto& loop = block.init_loop("_atom_site_", {"label", "type_symbol",
        "fract_x", "fract_y", "fract_z"});
    for (int a = 0; a < n_atoms; ++a) {
        const char* sym = element_symbol(Z[a]);
        loop.add_row({fmt::format("{}{}", sym, a + 1), sym,
                      fmt::format("{:.6f}", frac[3 * a + 0]),
                      fmt::format("{:.6f}", frac[3 * a + 1]),
                      fmt::format("{:.6f}", frac[3 * a + 2])});
    }
    gemmi::cif::write_cif_to_stream(out, doc);
}

}  // namespace klasp
