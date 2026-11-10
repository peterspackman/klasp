// The `energy` subcommand: evaluate or relax one or more structures under
// either the multipole force field or the PET ML potential. See cmd_energy.hpp
// for the energy normalisation.

#include "klasp/log.hpp"
#include "klasp/cmd_energy.hpp"

#include "klasp/types.hpp"
#include "klasp/cell.hpp"
#include "klasp/molecule.hpp"
#include "klasp/io.hpp"
#include "klasp/constants.hpp"
#include "klasp/batch.hpp"
#include "klasp/batch_kernels.hpp"
#include "klasp/csp.hpp"
#include "klasp/sgdata.hpp"
#include "klasp/element.hpp"
#include "klasp/ff_structure.hpp"
#ifdef KLASP_HAS_PET
#include "klasp/pet_bridge.hpp"
#include "klasp/pet_relax.hpp"
#endif

#include <fmt/format.h>
#include <limits>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace klasp {

namespace {

// One structure's result, whichever backend produced it. Energies are kJ/mol
// per molecule throughout.
struct EnergyRow {
    std::string name;
    int n_mols = 1;
    int n_atoms = 0;
    double e0 = 0;             // initial (single-point)
    double e1 = 0;             // after relaxation; equals e0 when not relaxing
    double max_force = -1;     // Ha/Bohr, <0 = not reported
    int iters = 0;
    bool converged = true;
    bool relaxed = false;
    bool ok = true;
    std::string error;
    std::string input;                         // the file it came from
    FFEnergyComponents components;             // ff: the lattice energy's terms, final geometry
    bool have_components = false;
    int frames = 0;                            // trajectory frames recorded
    std::map<std::string, std::string> outputs;   // what was written, by kind
    std::map<std::string, double> timings;        // ff: seconds, see FFTimings
};

void print_rows(const std::vector<EnergyRow>& rows, bool relaxed, bool ranked) {
    std::vector<int> order(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) order[i] = static_cast<int>(i);
    if (ranked)
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            if (!rows[a].ok || !rows[b].ok) return rows[a].ok && !rows[b].ok;
            return rows[a].e1 < rows[b].e1;
        });

    logging::info("\n  {:<30s} {:>5s} {:>6s} {:>16s}", "structure", "mols", "atoms",
               "E (kJ/mol/mol)");
    if (relaxed) logging::info(" {:>12s} {:>10s} {:>5s} {:>6s}", "dE", "max|F|", "conv", "iters");
    logging::info("\n");

    for (int i : order) {
        const auto& r = rows[i];
        if (!r.ok) {
            logging::info("  {:<30s} {:>5s} {:>6s} {:>16s}   {}\n", r.name, "-", "-", "FAILED",
                       r.error);
            continue;
        }
        logging::info("  {:<30s} {:>5d} {:>6d} {:>16.4f}", r.name, r.n_mols, r.n_atoms, r.e1);
        if (relaxed) {
            logging::info(" {:>12.4f}", r.e1 - r.e0);
            if (r.max_force >= 0) logging::info(" {:>10.2e}", r.max_force);
            else                  logging::info(" {:>10s}", "-");
            logging::info(" {:>5s} {:>6d}", r.converged ? "yes" : "no", r.iters);
        }
        logging::info("\n");
    }

    int n_ok = 0, n_conv = 0;
    for (const auto& r : rows) { if (r.ok) { ++n_ok; if (r.converged) ++n_conv; } }
    if (rows.size() > 1) {
        if (relaxed)
            logging::info("\n  {}/{} evaluated, {}/{} converged.\n", n_ok, rows.size(), n_conv, n_ok);
        else
            logging::info("\n  {}/{} evaluated.\n", n_ok, rows.size());
    }
}

// A structure JSON may carry a `reference.total` from occ, which tapers
// differently, so the gap is reported rather than asserted. e_per_mol is the
// single-point at the input geometry.
void report_reference(const std::string& path, double e_per_mol) {
    std::ifstream f(path);
    if (!f) return;
    nlohmann::json jd;
    try { f >> jd; } catch (...) { return; }
    if (!jd.contains("reference") || !jd["reference"].contains("total")) return;
    const double ref = jd["reference"]["total"].get<double>();
    const double diff = e_per_mol - ref;
    logging::info("  reference (from input): {:.6f} kJ/mol/molecule, diff {:+.6f}\n", ref, diff);
    if (std::abs(diff) > 5.0)
        logging::info("  NOTE: {:.1f} kJ/mol from the stored reference -- larger than the "
                   "~2-4 kJ/mol\n        taper-convention offset these references usually "
                   "carry.\n", diff);
}

// Every structure's numbers and files as JSON. Energies are kJ/mol per molecule.
void write_summary(const std::string& path, const std::vector<EnergyRow>& rows,
                   const EnergyOptions& o) {
    nlohmann::json j;
    j["units"] = {{"energy", "kJ/mol per molecule"}};
    j["backend"] = o.backend;
    j["relax"] = o.relax;
    j["structures"] = nlohmann::json::array();
    for (const auto& r : rows) {
        nlohmann::json s;
        s["name"] = r.name;
        s["input"] = r.input;
        s["ok"] = r.ok;
        if (!r.ok) {
            s["error"] = r.error;
            j["structures"].push_back(std::move(s));
            continue;
        }
        s["n_molecules"] = r.n_mols;
        s["n_atoms"] = r.n_atoms;
        s["initial_energy"] = r.e0;
        s["energy"] = r.e1;
        s["relaxed"] = r.relaxed;
        if (r.relaxed) {
            s["converged"] = r.converged;
            s["iterations"] = r.iters;
            if (r.max_force >= 0) s["grad_max"] = r.max_force;
            if (r.frames > 0) s["trajectory_frames"] = r.frames;
        }
        if (r.have_components)
            s["components"] = {{"total", r.components.total},
                               {"electrostatic", r.components.electrostatic},
                               {"buckingham", r.components.buckingham}};
        if (!r.outputs.empty()) s["outputs"] = r.outputs;
        if (!r.timings.empty()) s["timings"] = r.timings;
        j["structures"].push_back(std::move(s));
    }
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("energy: cannot write the summary to " + path);
    f << j.dump(2) << "\n";
}

// --- force-field backend ---------------------------------------------------

void write_relaxed_occ_json(const std::string& out, const BatchWorkspace& bws,
                            const BatchOptimizeResult& res, MoleculeData& mol,
                            const CrystalConfig& config, double energy_per_mol,
                            int slot = 0);

nlohmann::json read_json_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open " + path);
    return nlohmann::json::parse(f);
}

void write_text_file(const std::string& path, const std::function<void(std::ostream&)>& write) {
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot write " + path);
    write(f);
}

// One structure file under the force field, through klasp/ff_structure.hpp
// (shared with the Python module). Perturbed starts follow the input's row.
std::vector<EnergyRow> ff_run(const std::string& path, const EnergyOptions& o) {
    const nlohmann::json doc = read_json_file(path);
    const std::string file = std::filesystem::path(path).filename().string();
    const std::string stem = std::filesystem::path(path).stem().string();

    FFModelOptions model;
    model.max_order = o.max_order;
    model.ewald_accuracy = o.ewald_accuracy;
    model.no_taper = o.no_taper;
    model.mp_cutoff_ang = o.mp_cutoff_ang;
    model.mp_single = o.mp_single;

    std::vector<FFStructure> results;
    if (o.relax) {
        FFRelaxOptions ro;
        ro.model = model;
        ro.fixed_cell = o.fixed_cell;
        ro.pressure_gpa = o.pressure_gpa;   // 0 keeps the number a lattice energy
        if (o.iterations > 0) ro.iterations = o.iterations;
        if (o.grad_tol > 0) ro.grad_tol = o.grad_tol;
        ro.perturb = std::max(0, o.perturb);
        ro.trajectory = o.trajectory;
        results = ff_relax(doc, ro);
    } else {
        results.push_back(ff_evaluate(doc, model));
    }

    std::vector<EnergyRow> rows;
    for (const FFStructure& s : results) {
        EnergyRow r;
        r.name = file + s.label;
        r.input = path;
        r.n_mols = s.n_molecules;
        r.n_atoms = s.n_atoms;
        r.e0 = s.initial_energy;
        r.e1 = s.energy;
        r.relaxed = s.relaxed;
        r.converged = s.converged;
        r.iters = s.iterations;
        r.max_force = s.relaxed ? s.grad_max : -1;
        r.components = s.components;
        r.have_components = !s.structure.is_null();
        r.frames = static_cast<int>(s.trajectory.size());
        r.timings = timings_by_name(s.timings);
        if (!std::isfinite(r.e1)) { r.ok = false; r.error = "non-finite energy"; }
        rows.push_back(std::move(r));
    }

    // Files are written for the structure as given; perturbed starts are only
    // reported.
    const FFStructure& s = results.front();
    EnergyRow& row = rows.front();
    if (!row.ok) return rows;
    if (o.relax && !o.output_dir.empty()) {
        std::filesystem::create_directories(o.output_dir);
        const std::string base = (std::filesystem::path(o.output_dir) / stem).string();
        write_text_file(base + ".json", [&](std::ostream& f) { f << s.structure.dump(2) << "\n"; });
        write_text_file(base + ".cif", [&](std::ostream& f) { write_structure_cif(f, s, stem); });
        row.outputs["structure"] = base + ".json";
        row.outputs["cif"] = base + ".cif";
        if (o.trajectory) {
            write_text_file(base + ".traj.xyz", [&](std::ostream& f) { write_trajectory_xyz(f, s); });
            row.outputs["trajectory"] = base + ".traj.xyz";
        }
    }
    if (!o.write_json.empty()) {
        write_text_file(o.write_json, [&](std::ostream& f) { f << s.structure.dump(2) << "\n"; });
        row.outputs["write_json"] = o.write_json;
    }
    return rows;
}

// A landscape document: one basis, one space group, many (cell, placements).
// Only the cell and rigid-body DOFs vary per structure, so the whole set relaxes
// in one batch_optimize_asymunit call.
bool is_batch_document(const nlohmann::json& doc) {
    const auto it = doc.find("structures");
    return it != doc.end() && it->is_array();
}

std::vector<EnergyRow> ff_run_batch(const std::string& path, const EnergyOptions& o) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open " + path);
    const nlohmann::json doc = nlohmann::json::parse(f);
    const auto& structures = doc.at("structures");
    const int n_batch = static_cast<int>(structures.size());
    if (n_batch == 0) throw std::runtime_error(path + ": \"structures\" is empty");

    // Rebuild a single-structure document per entry, so both paths share parsing.
    auto as_single = [&](int i) {
        nlohmann::json d = doc;
        d.erase("structures");
        // A document with a basis is read from `crystal`, so the entry's geometry
        // goes there.
        nlohmann::json& target = d.contains("basis") ? d["crystal"] : d;
        target["cell"] = structures[i].at("cell");
        target["molecules"] = structures[i].at("molecules");
        return d;
    };

    auto ws = allocate_workspace(512, 27 * 27 * 27, 50000, 32, 4);
    MoleculeData mol{};
    auto config = load_structure_json(as_single(0), mol, ws);
    if (o.ewald_accuracy >= 0) apply_ewald_accuracy_override(o.ewald_accuracy, ws);
    if (o.no_taper) { config.taper_on = 0.0; config.taper_off = 0.0; }
    int max_rank = config.max_rank;
    if (o.max_order >= 0) max_rank = o.max_order;
    const int n_mols = std::max(1, mol.n_symops);

    const std::string stem = std::filesystem::path(path).stem().string();
    std::vector<EnergyRow> rows(n_batch);
    std::vector<std::string> names(n_batch);
    for (int i = 0; i < n_batch; ++i) {
        names[i] = structures[i].value("name", fmt::format("{}_{}", stem, i));
        rows[i].name = names[i];
        rows[i].n_mols = n_mols;
        rows[i].n_atoms = n_mols * mol.n_sites_per_mol;
    }

    auto bws = allocate_batch_workspace(n_batch, ws, mol);
    auto h_bt = Kokkos::create_mirror_view(bws.dofs.translations);
    auto h_br = Kokkos::create_mirror_view(bws.dofs.rot_vectors);
    auto h_cp = Kokkos::create_mirror_view(bws.geom.cell_params);

    // A second workspace, so parsing the rest cannot disturb the batch's.
    auto parse_ws = allocate_workspace(512, 27 * 27 * 27, 50000, 32, 4);
    for (int i = 0; i < n_batch; ++i) {
        MoleculeData m{};
        load_structure_json(as_single(i), m, parse_ws);
        if (m.n_indep != mol.n_indep || m.n_symops != mol.n_symops ||
            m.n_sites_per_mol != mol.n_sites_per_mol)
            throw std::runtime_error(fmt::format(
                "{}: structure \"{}\" does not match the first. A batch shares one basis "
                "and one space group, so n_indep, n_symops and n_sites must agree",
                path, names[i]));

        const RealType E0 = batch_evaluate_energy(m, parse_ws, max_rank, config.taper_on,
                                                  config.taper_off, config.buck_damping);
        rows[i].e0 = rows[i].e1 = double(E0) * AU_TO_KJ_PER_MOL / n_mols;

        auto h_t = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m.translations);
        auto h_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m.rot_vectors);
        for (int k = 0; k < mol.n_indep; ++k)
            for (int d = 0; d < 3; ++d) {
                h_bt(i, k, d) = h_t(k, d);
                h_br(i, k, d) = h_r(k, d);
            }
        for (int j = 0; j < 6; ++j) h_cp(i, j) = m.cell_params[j];
    }
    Kokkos::deep_copy(bws.dofs.translations, h_bt);
    Kokkos::deep_copy(bws.dofs.rot_vectors, h_br);
    Kokkos::deep_copy(bws.geom.cell_params, h_cp);
    batch_setup_cells(bws.tables, bws.geom);

    if (!o.relax) return rows;

    if (!o.fixed_cell) {
        int lt = TRICLINIC;
        if (config.space_group > 0)
            lt = static_cast<int>(get_space_group(config.space_group).lattice);
        bws.dofs.d_g_strain = SView2D<RealType, ExecSpace>("d_g_strain", n_batch, 6);
        bws.dofs.strain     = SView2D<RealType, ExecSpace>("strain", n_batch, 6);
        bws.dofs.cell_ref   = SView3D<RealType, ExecSpace>("cell_ref", n_batch, 3, 3);
        batch_set_strain_dofs(bws.geom, bws.dofs, lt);
        Kokkos::deep_copy(bws.dofs.cell_ref, bws.geom.cell);
        Kokkos::deep_copy(bws.dofs.strain, RealType(0));
        Kokkos::deep_copy(bws.dofs.d_g_strain, RealType(0));
    }

    BatchOptimizeSettings bs;
    bs.n_iterations = o.iterations > 0 ? o.iterations : 300;
    bs.grad_tol = o.grad_tol > 0 ? o.grad_tol : 1e-8;
    bs.model.taper_on = config.taper_on;
    bs.model.taper_off = config.taper_off;
    bs.model.buck_damping = config.buck_damping;
    bs.pressure_gpa = o.pressure_gpa;
    bs.store_diagnostics = true;

    // The exact rank, as the single points above use.
    BatchOptimizeResult res;
    switch (max_rank) {
        case 0:  res = batch_optimize_asymunit<0>(bws, bs); break;
        case 1:  res = batch_optimize_asymunit<1>(bws, bs); break;
        case 2:  res = batch_optimize_asymunit<2>(bws, bs); break;
        case 3:  res = batch_optimize_asymunit<3>(bws, bs); break;
        default: res = batch_optimize_asymunit<4>(bws, bs); break;
    }

    auto h_E = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res.final_energies);
    for (int b = 0; b < n_batch; ++b) {
        rows[b].e1 = double(h_E(b)) * AU_TO_KJ_PER_MOL / n_mols;
        rows[b].relaxed = true;
        rows[b].iters = res.n_iterations;
        if (res.diagnostics.valid() && res.diagnostics.grad_max.extent(0) > size_t(b)) {
            rows[b].max_force = double(res.diagnostics.grad_max(b));
            const int itc = res.diagnostics.iters_to_converge(b);
            rows[b].converged = itc >= 0;
            if (itc >= 0) rows[b].iters = itc;
        }
        if (!std::isfinite(rows[b].e1)) {
            rows[b].ok = false;
            rows[b].error = "non-finite energy";
        }
    }

    if (!o.output_dir.empty()) {
        std::filesystem::create_directories(o.output_dir);
        for (int b = 0; b < n_batch; ++b)
            if (rows[b].ok)
                write_relaxed_occ_json(o.output_dir + "/" + names[b] + ".json", bws, res, mol,
                                       config, rows[b].e1, b);
    }
    return rows;
}

// Copy one batch slot's relaxed DOFs back onto mol and write the occ-format
// JSON, readable by `klasp energy`. The cell comes from the relaxed lattice
// vectors, since the strain DOFs do not update cell_params.
void write_relaxed_occ_json(const std::string& out, const BatchWorkspace& bws,
                            const BatchOptimizeResult& res, MoleculeData& mol,
                            const CrystalConfig& config, double energy_per_mol,
                            int slot) {
    auto h_cell = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bws.geom.cell);
    RealType m[3][3];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) m[r][c] = h_cell(slot, r, c);
    matrix_to_cell_params(m, mol.cell_params);
    // The orientations are in the optimiser's frame and the document is read back
    // in the canonical one; see align_orientations_to_canonical_cell.
    std::vector<RealType> rv(3 * static_cast<size_t>(mol.n_indep));
    for (int k = 0; k < mol.n_indep; ++k)
        for (int d = 0; d < 3; ++d) rv[3 * k + d] = res.final_rotations(slot, k, d);
    align_orientations_to_canonical_cell(m, mol.cell_params, rv.data(), mol.n_indep);

    auto h_t = Kokkos::create_mirror_view(mol.translations);
    auto h_r = Kokkos::create_mirror_view(mol.rot_vectors);
    for (int k = 0; k < mol.n_indep; ++k)
        for (int d = 0; d < 3; ++d) {
            h_t(k, d) = res.final_translations(slot, k, d);
            h_r(k, d) = rv[3 * k + d];
        }
    Kokkos::deep_copy(mol.translations, h_t);
    Kokkos::deep_copy(mol.rot_vectors, h_r);

    BasisConfig bc;
    bc.basis_json = config.basis_json;
    write_occ_json(out, bc, mol, energy_per_mol, config.molecule_type_index);
}

#ifdef KLASP_HAS_PET

// mol_atom_counts, when given, receives each molecule's real atom count in the
// compacted structure, in emission order (n_sites is the padded stride).
CrystalGeometry load_pet_structure(const std::string& path, int& n_mols, int& n_sites,
                                   PetSymmetry* sym_out = nullptr, bool expand_p1 = false,
                                   std::vector<int>* mol_atom_counts = nullptr) {
    auto ws = allocate_workspace(512, 27 * 27 * 27, 50000, 32, 4);
    MoleculeData mol{};
    auto config = load_structure_json(path, mol, ws);
    expand_asymmetric_unit<0>(mol, ws);

    const int N = ws.n_atoms;
    n_mols = mol.n_symops;
    n_sites = mol.n_sites_per_mol;

    // Symmetry operators (fractional) for symmetry-constrained relaxation, any
    // Z'. With --expand-p1 every copy is its own independent molecule under the
    // identity operator. type_n_sites holds real site counts per molecule type;
    // n_sites stays the padded stride used to index h_pos.
    std::vector<int> type_n_sites;
    {
        std::ifstream tf(path);
        nlohmann::json tj; tf >> tj;
        for (const auto& mt : structure_molecule_types(tj))
            type_n_sites.push_back(static_cast<int>(mt["sites"].size()));
    }

    if (sym_out) {
        auto h_sii0 = Kokkos::create_mirror_view(mol.sym_indep_index);
        Kokkos::deep_copy(h_sii0, mol.sym_indep_index);
        // An independent molecule's type is the one the file named for it, not its
        // list position.
        auto type_of = [&](int idx) {
            const int t = config.molecule_type_index.at(idx);
            return type_n_sites.empty() ? n_sites : type_n_sites[t];
        };
        *sym_out = PetSymmetry{};
        sym_out->n_op = mol.n_symops;
        if (expand_p1) {
            sym_out->n_indep = mol.n_symops;          // each copy independent
            sym_out->indep_of_op.resize(mol.n_symops);
            sym_out->sites_of_indep.resize(mol.n_symops);
            sym_out->rot.assign(static_cast<std::size_t>(mol.n_symops) * 9, 0.0);
            sym_out->trans.assign(static_cast<std::size_t>(mol.n_symops) * 3, 0.0);
            for (int op = 0; op < mol.n_symops; ++op) {
                sym_out->indep_of_op[op] = op;        // identity op for its own indep
                // Its sites come from the component its operator generated.
                sym_out->sites_of_indep[op] = type_of(h_sii0(op));
                sym_out->rot[op * 9 + 0] = sym_out->rot[op * 9 + 4] = sym_out->rot[op * 9 + 8] = 1.0;
            }
        } else {
            auto h_sr = Kokkos::create_mirror_view(mol.sym_rotations);
            auto h_st = Kokkos::create_mirror_view(mol.sym_translations);
            auto h_si = Kokkos::create_mirror_view(mol.sym_indep_index);
            Kokkos::deep_copy(h_sr, mol.sym_rotations);
            Kokkos::deep_copy(h_st, mol.sym_translations);
            Kokkos::deep_copy(h_si, mol.sym_indep_index);
            sym_out->n_indep = mol.n_indep;
            sym_out->indep_of_op.resize(mol.n_symops);
            sym_out->sites_of_indep.assign(mol.n_indep, n_sites);
            sym_out->rot.resize(static_cast<std::size_t>(mol.n_symops) * 9);
            sym_out->trans.resize(static_cast<std::size_t>(mol.n_symops) * 3);
            for (int op = 0; op < mol.n_symops; ++op) {
                sym_out->indep_of_op[op] = h_si(op);
                if (h_si(op) >= 0 && h_si(op) < mol.n_indep)
                    sym_out->sites_of_indep[h_si(op)] = type_of(h_si(op));
                for (int e = 0; e < 9; ++e) sym_out->rot[op * 9 + e] = h_sr(op, e);
                for (int e = 0; e < 3; ++e) sym_out->trans[op * 3 + e] = h_st(op, e);
            }
        }
    }

    std::ifstream jf(path);
    nlohmann::json jd;
    jf >> jd;
    std::vector<std::vector<int>> type_site_Z;  // [mol_type][site] -> Z
    for (const auto& mt : structure_molecule_types(jd)) {
        std::vector<int> zs;
        for (const auto& site : mt["sites"])
            zs.push_back(element(site.value("element", site.value("label", std::string("X"))))
                             .atomic_number);
        type_site_Z.push_back(std::move(zs));
    }

    auto h_pos = Kokkos::create_mirror_view(ws.positions);
    Kokkos::deep_copy(h_pos, ws.positions);
    auto h_cell = Kokkos::create_mirror_view(ws.cell_matrix);
    Kokkos::deep_copy(h_cell, ws.cell_matrix);
    auto h_sii = Kokkos::create_mirror_view(mol.sym_indep_index);
    Kokkos::deep_copy(h_sii, mol.sym_indep_index);

    // Real atoms only: co-crystal padding slots are skipped (see
    // klasp/padding.hpp).
    CrystalGeometry s;
    s.positions.reserve(N);
    s.atomic_numbers.reserve(N);
    if (mol_atom_counts) mol_atom_counts->assign(std::max(0, n_mols), 0);
    for (int i = 0; i < N; ++i) {
        const int sop = i / n_sites, site = i % n_sites;
        const int t = config.molecule_type_index.at(h_sii(sop));  // indep -> mol type
        if (site >= (int)type_site_Z[t].size()) continue;              // padding
        if (mol_atom_counts && sop < (int)mol_atom_counts->size()) ++(*mol_atom_counts)[sop];
        s.positions.push_back({h_pos(i, 0), h_pos(i, 1), h_pos(i, 2)});
        s.atomic_numbers.push_back(type_site_Z[t][site]);
    }
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) s.cell[r][c] = h_cell(r, c);
    return s;
}

// Write a relaxed structure as extended XYZ (Angstrom, ASE Lattice convention:
// row-major a,b,c). klasp stores lattice vectors as columns, so a = cell column 0.
void write_ext_xyz(const std::string& out, const CrystalGeometry& s, double energy_ev) {
    std::ofstream f(out);
    const int N = static_cast<int>(s.atomic_numbers.size());
    f << N << "\n";
    f << "Lattice=\"";
    for (int vec = 0; vec < 3; ++vec)
        for (int comp = 0; comp < 3; ++comp)
            f << (vec || comp ? " " : "") << s.cell[comp][vec] * BOHR_TO_ANGSTROM;
    f << "\" Properties=species:S:1:pos:R:3 pbc=\"T T T\" energy=" << energy_ev << "\n";
    for (int i = 0; i < N; ++i) {
        f << element_symbol(s.atomic_numbers[i]) << " "
          << s.positions[i][0] * BOHR_TO_ANGSTROM << " "
          << s.positions[i][1] * BOHR_TO_ANGSTROM << " "
          << s.positions[i][2] * BOHR_TO_ANGSTROM << "\n";
    }
}


// --- PET backend -----------------------------------------------------------

void pet_isolated(const std::vector<std::string>& paths, const EnergyOptions& o,
                  const PetEvaluator& eval) {
    // The vacuum reference a PET lattice energy is measured against; shares
    // pet_isolated_reference with the CSP PET stage.
    for (const auto& p : paths) {
        auto ws = allocate_workspace(512, 27 * 27 * 27, 50000, 32, 4);
        BasisConfig basis = load_basis_json(p, ws);
        const double ref = pet_isolated_reference(eval, basis);
        logging::info("  {:<30s} reference = {:.4f} kJ/mol per formula unit ({} component{})\n",
                   std::filesystem::path(p).filename().string(), ref,
                   basis.n_mols, basis.n_mols == 1 ? "" : "s");
    }
    logging::info("\n  E_lattice/molecule = E_crystal(kJ/mol/mol) - reference.\n");
}

EnergyResult cmd_energy_pet(const std::vector<std::string>& paths, const EnergyOptions& o) {
    std::string mj = o.model_json, ms = o.model_st;
    if (mj.empty() || ms.empty()) {
        std::string rj, rs;
        resolve_pet_model(o.model, rj, rs);
        if (mj.empty()) mj = rj;
        if (ms.empty()) ms = rs;
    }
    PetEvaluator eval(mj, ms);

    if (o.isolated) {
        // Prints its own table; no per-structure rows.
        pet_isolated(paths, o, eval);
        return EnergyResult{};
    }

    std::vector<CrystalGeometry> structs;
    std::vector<PetSymmetry> syms;
    std::vector<EnergyRow> rows;
    std::vector<int> row_of_struct;
    std::vector<std::vector<int>> mol_atoms_of_struct;
    int n_no_sym = 0;

    for (const auto& p : paths) {
        EnergyRow r;
        r.name = std::filesystem::path(p).filename().string();
        r.input = p;
        try {
            int nm = 0, ns = 0;
            PetSymmetry sym;
            std::vector<int> mol_atoms;
            CrystalGeometry s = load_pet_structure(p, nm, ns, o.symmetry ? &sym : nullptr,
                                                   o.expand_p1, &mol_atoms);
            if (o.symmetry && !sym.valid()) {
                ++n_no_sym;
                r.ok = false;
                r.error = "symmetry operators unavailable (needs Z'=1)";
                rows.push_back(std::move(r));
                continue;
            }
            r.n_mols = std::max(1, nm);
            r.n_atoms = static_cast<int>(s.atomic_numbers.size());
            row_of_struct.push_back(static_cast<int>(rows.size()));
            mol_atoms_of_struct.push_back(std::move(mol_atoms));
            structs.push_back(std::move(s));
            syms.push_back(std::move(sym));
            rows.push_back(std::move(r));
        } catch (const std::exception& e) {
            r.ok = false;
            r.error = e.what();
            rows.push_back(std::move(r));
        }
    }

    if (structs.empty()) {
        print_rows(rows, o.relax, paths.size() > 1);
        // Distinguish symmetry rejections from parse failures.
        throw std::runtime_error(
            n_no_sym == int(paths.size())
                ? "energy: --symmetry could not build operators for any input "
                  "(it currently requires Z'=1); drop --symmetry to relax freely"
                : "energy: no structures could be loaded");
    }
    if (n_no_sym)
        logging::info("  {} structure{} skipped: --symmetry currently requires Z'=1.\n",
                   n_no_sym, n_no_sym == 1 ? "" : "s");

    // Single-point energies for every structure, so --relax can report a dE.
    for (size_t i = 0; i < structs.size(); ++i) {
        auto e = eval.evaluate(structs[i], true);
        EnergyRow& r = rows[row_of_struct[i]];
        r.e0 = r.e1 = e.energy[0] * AU_TO_KJ_PER_MOL / r.n_mols;
    }

    if (o.relax) {
        PetRelaxOptions opts;
        opts.relax_cell = !o.fixed_cell;
        opts.verbose = paths.size() == 1;
        opts.device = o.device;
        if (o.iterations > 0) opts.max_iter_full = o.iterations;
        if (o.grad_tol > 0) opts.force_tol = o.grad_tol;
        if (o.rigid && !o.symmetry) {
            // Stage A rigid groups index the compacted structure, so walk each
            // molecule's real atom count. Single-structure only (one group set per
            // PetRelaxOptions).
            opts.max_iter_rigid = 60;
            int first = 0;
            for (int n : mol_atoms_of_struct.front()) {
                if (n <= 0) continue;
                std::vector<int> g(n);
                for (int k = 0; k < n; ++k) g[k] = first + k;
                first += n;
                opts.rigid_groups.push_back(std::move(g));
            }
        }

        // Only pet_relax runs Stage A, so --rigid takes that path.
        std::vector<PetRelaxResult> res;
        if (!opts.rigid_groups.empty())
            res.push_back(pet_relax(eval, structs.front(), opts));
        else
            // In chunks the device can hold, as rerank does.
            res = pet_relax_chunked(eval, structs,
                                    o.symmetry ? syms : std::vector<PetSymmetry>{}, opts);

        for (size_t i = 0; i < structs.size() && i < res.size(); ++i) {
            EnergyRow& r = rows[row_of_struct[i]];
            r.relaxed = true;
            r.e1 = res[i].energy * AU_TO_KJ_PER_MOL / r.n_mols;
            r.max_force = res[i].max_force;
            r.iters = res[i].full_iterations;
            r.converged = res[i].converged;
            if (!std::isfinite(r.e1)) { r.ok = false; r.error = "non-finite energy"; }
            if (!o.output_dir.empty() && r.ok) {
                std::filesystem::create_directories(o.output_dir);
                const std::string stem =
                    std::filesystem::path(paths[row_of_struct[i]]).stem().string();
                write_ext_xyz(o.output_dir + "/" + stem + ".xyz", res[i].structure,
                              res[i].energy / EV_TO_AU);
            }
        }
    }

    print_rows(rows, o.relax, paths.size() > 1);
    if (!o.summary.empty()) write_summary(o.summary, rows, o);
    // No reference comparison: a stored `reference.total` is a force-field
    // lattice energy, while PET reports a total energy. Subtract --isolated
    // for a lattice energy.
    if (!o.output_dir.empty() && o.relax)
        logging::info("\n  Wrote relaxed structures to {}/\n", o.output_dir);

    EnergyResult result;
    result.structures = static_cast<int>(rows.size());
    result.energy_kj_per_mol.reserve(rows.size());
    for (const auto& r : rows) {
        if (!r.ok) ++result.failed;
        result.energy_kj_per_mol.push_back(
            r.ok ? r.e1 : std::numeric_limits<double>::quiet_NaN());
    }
    if (result.failed) throw std::runtime_error(
        fmt::format("energy: {} of {} structures failed", result.failed, rows.size()));
    return result;
}

#endif  // KLASP_HAS_PET

}  // namespace

EnergyResult cmd_energy(const std::vector<std::string>& paths, const EnergyOptions& opts) {
    if (paths.empty()) throw std::runtime_error("energy: no input structures");
    if (opts.perturb > 0 && paths.size() > 1)
        throw std::runtime_error(
            "energy: --perturb samples around one structure; pass a single input");
    if (opts.perturb > 0 && !opts.relax)
        throw std::runtime_error("energy: --perturb only means something with --relax");
    if (opts.rigid && paths.size() > 1)
        throw std::runtime_error(
            "energy: --rigid builds one set of rigid groups for the whole batch, so it "
            "takes a single input");
    if (opts.rigid && opts.symmetry)
        throw std::runtime_error(
            "energy: --rigid and --symmetry are alternatives -- a symmetry-constrained "
            "relaxation already moves only the asymmetric unit");
    if (opts.backend != "ff" && opts.backend != "pet")
        throw std::runtime_error("energy: --backend must be ff or pet");
    if (opts.trajectory && !opts.relax)
        throw std::runtime_error("energy: --trajectory records a relaxation; add --relax");
    if (opts.trajectory && opts.output_dir.empty())
        throw std::runtime_error(
            "energy: --trajectory writes <name>.traj.xyz into --output-dir; give one");
    if (opts.trajectory && opts.backend == "pet")
        throw std::runtime_error(
            "energy: --trajectory records the force-field optimiser; the PET relaxers keep none");
    if (!opts.write_json.empty() && paths.size() > 1)
        throw std::runtime_error("energy: --write-json takes a single input structure");

    if (opts.backend == "pet") {
#ifdef KLASP_HAS_PET
        return cmd_energy_pet(paths, opts);
#else
        throw std::runtime_error(
            "energy: --backend pet requires a build with KLASP_ENABLE_PET=ON");
#endif
    }

    if (opts.isolated)
        throw std::runtime_error("energy: --isolated is a PET reference; add --backend pet");
    if (opts.symmetry || opts.rigid || opts.expand_p1)
        throw std::runtime_error(
            "energy: --symmetry, --rigid and --expand-p1 apply to --backend pet");

    std::vector<EnergyRow> rows;
    for (const auto& p : paths) {
        try {
            {
                std::ifstream probe(p);
                const nlohmann::json head =
                    probe.is_open() ? nlohmann::json::parse(probe, nullptr, false)
                                    : nlohmann::json();
                if (!head.is_discarded() && is_batch_document(head)) {
                    if (opts.trajectory)
                        throw std::runtime_error(
                            "--trajectory is not supported for a document of many structures");
                    if (!opts.write_json.empty())
                        throw std::runtime_error(
                            "--write-json takes a single structure; a document of many "
                            "structures writes each one with --output-dir");
                    for (auto& r : ff_run_batch(p, opts)) rows.push_back(std::move(r));
                    continue;
                }
            }
            for (auto& r : ff_run(p, opts)) rows.push_back(std::move(r));
        } catch (const std::exception& e) {
            EnergyRow r;
            r.name = std::filesystem::path(p).filename().string();
            r.input = p;
            r.ok = false;
            r.error = e.what();
            rows.push_back(std::move(r));
        }
    }

    print_rows(rows, opts.relax, rows.size() > 1 || opts.perturb > 0);
    if (paths.size() == 1 && rows.size() == 1 && rows[0].ok)
        report_reference(paths[0], rows[0].e0);
    if (!opts.summary.empty()) write_summary(opts.summary, rows, opts);

    EnergyResult result;
    result.structures = static_cast<int>(rows.size());
    result.energy_kj_per_mol.reserve(rows.size());
    for (const auto& r : rows) {
        if (!r.ok) ++result.failed;
        result.energy_kj_per_mol.push_back(
            r.ok ? r.e1 : std::numeric_limits<double>::quiet_NaN());
    }
    if (result.failed) throw std::runtime_error(
        fmt::format("energy: {} of {} structures failed", result.failed, rows.size()));
    return result;
}

}  // namespace klasp
