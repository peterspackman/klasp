// klasp wyckoff: which special positions a molecule can occupy.

#include "klasp/wyckoff.hpp"
#include "klasp/compack.hpp"
#include "klasp/element.hpp"
#include "klasp/io.hpp"
#include "klasp/log.hpp"
#include "klasp/constants.hpp"
#include "klasp/exit_code.hpp"

#include <fmt/format.h>

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace klasp {

namespace {

// "156,183,194" or "1-230" or a mix. Refuses anything outside 1..230.
std::vector<int> parse_space_groups(const std::string& spec) {
    std::set<int> out;
    size_t pos = 0;
    while (pos <= spec.size()) {
        const size_t comma = spec.find(',', pos);
        std::string tok = spec.substr(pos, comma == std::string::npos
                                          ? std::string::npos : comma - pos);
        if (!tok.empty()) {
            const size_t dash = tok.find('-', 1);
            try {
                if (dash == std::string::npos) {
                    out.insert(std::stoi(tok));
                } else {
                    const int lo = std::stoi(tok.substr(0, dash));
                    const int hi = std::stoi(tok.substr(dash + 1));
                    for (int s = lo; s <= hi; ++s) out.insert(s);
                }
            } catch (const std::exception&) {
                throw std::runtime_error("--sg: cannot read '" + tok + "'");
            }
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    for (int s : out)
        if (s < 1 || s > 230)
            throw std::runtime_error("--sg: " + std::to_string(s)
                                     + " is not a space group number (1 to 230)");
    return {out.begin(), out.end()};
}

}  // namespace

int cmd_wyckoff(const WyckoffOptions& o) {
    // Sized for a basis read; load_basis_json fills the potential tables.
    Workspace ws = allocate_workspace(256, 512, 1024, 16);
    const BasisConfig basis = load_basis_json(o.basis_path, ws);

    if (basis.n_mols != 1) {
        logging::error("wyckoff: this reads one molecule, and {} has {}. A co-crystal "
                       "would need a site for each component, which is a different "
                       "question from the one asked here.\n",
                       o.basis_path, basis.n_mols);
        return +ExitCode::Failed;
    }

    // The molecule's own symmetry, from the detector compack uses. Uses the flat
    // arrays; mol_body_positions is empty for a single molecule.
    const auto& pos_bohr = basis.body_positions;
    const auto& els = basis.elements;
    if (pos_bohr.empty() || els.size() != pos_bohr.size()) {
        logging::error("wyckoff: {} gave {} positions and {} element symbols\n",
                       o.basis_path, pos_bohr.size(), els.size());
        return +ExitCode::Failed;
    }
    std::vector<double> xyz;
    std::vector<int> z;
    for (size_t i = 0; i < pos_bohr.size(); ++i) {
        for (int d = 0; d < 3; ++d)
            xyz.push_back(pos_bohr[i][d] * BOHR_TO_ANGSTROM);
        z.push_back(element(els[i]).atomic_number);
    }
    const int n = static_cast<int>(z.size());
    RigidBody body = make_rigid_body(xyz.data(), z.data(), n, /*heavy_only=*/false);
    detect_self_symmetry(body, o.self_symmetry_tolerance);
    const std::vector<double>& mol_ops = body.self_ops;
    const int mol_order = static_cast<int>(mol_ops.size() / 9);

    fmt::print("molecule: {} atoms, {} symmetry operation{} at a tolerance of {} A\n",
               n, mol_order, mol_order == 1 ? "" : "s", o.self_symmetry_tolerance);
    if (mol_order == 1) {
        fmt::print("\nNo symmetry beyond the identity, so no special position is open to "
                   "it:\nevery site of every group asks for at least one operation the "
                   "molecule does not have.\n");
        return +ExitCode::Ok;
    }

    const auto sgs = parse_space_groups(o.space_groups);
    fmt::print("\n{:>4} {:>6} {:>5} {:>5} {:>4} {:>5}  {}\n",
               "SG", "symops", "site", "Z", "free", "oris", "site position");

    int n_fitting = 0;
    for (int sg_num : sgs) {
        const WyckoffReport rep = wyckoff_report(sg_num, mol_ops);
        for (const auto& e : rep.entries) {
            const bool fits = e.n_orientations > 0;
            if (!fits && !o.all_sites) continue;
            if (fits) ++n_fitting;
            fmt::print("{:>4} {:>6} {:>5} {:>5} {:>4} {:>5}  ({:.3f}, {:.3f}, {:.3f}){}\n",
                       sg_num, rep.n_symops, e.site_order, e.multiplicity,
                       e.n_free + e.orientation_dof, e.n_orientations,
                       e.rep[0], e.rep[1], e.rep[2],
                       fits ? "" : "   (molecule too unsymmetric)");
        }
    }

    fmt::print("\n{} site{} this molecule can occupy across {} space group{}.\n",
               n_fitting, n_fitting == 1 ? "" : "s",
               sgs.size(), sgs.size() == 1 ? "" : "s");
    fmt::print("Z is how many molecules the cell would then hold, against {} for a "
               "general\nposition in the same group. `free` counts the degrees of "
               "freedom left: how far\nthe molecule may still slide along the site plus "
               "how far it may still turn.\n",
               sgs.size() == 1 ? std::to_string(get_space_group(sgs[0]).n_symops)
                               : std::string("the group's operation count"));
    return +ExitCode::Ok;
}

}  // namespace klasp
