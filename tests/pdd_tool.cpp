// Print the PDD of structure JSON files (as `klasp extract -f json` and
// `klasp energy --relax -o` write them), one JSON object per line.
//
//   pdd_tool <database> <structure.json>... [--k 100,200]
//
// The database supplies the basis. Rows are the asymmetric unit's atoms in
// basis order.
#include "klasp/constants.hpp"
#include "klasp/db_tool.hpp"
#include "klasp/flex_descriptors.hpp"
#include "klasp/io.hpp"
#include "klasp/sgdata.hpp"

#include <Kokkos_Core.hpp>
#include <nlohmann/json.hpp>

#include "tool_cli.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using nlohmann::json;

int main(int argc, char** argv) {
    CLI::App app{"Print the PDD of structure JSON files.", "klasp_pdd"};
    klasp::tools::tool_init(app);
    std::string db_path;
    std::vector<std::string> inputs;
    std::vector<int> ks{klasp::AMD_K};
    app.add_option("database", db_path, "CSP database holding the basis")->required();
    app.add_option("inputs", inputs, "Structure JSON files")->required();
    app.add_option("--k", ks, "Neighbours per row, one or more")->delimiter(',');
    KLASP_TOOL_PARSE(app, argc, argv);

    int rc = 0;
    Kokkos::initialize(argc, argv);
    {
        const klasp::BasisConfig basis = klasp::load_csp_basis("", db_path);
        for (const auto& path : inputs) {
            std::ifstream f(path);
            if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); rc = 2; continue; }
            const json doc = json::parse(f);
            const int sg = doc.at("space_group").get<int>();
            const auto& c = doc.at("cell");
            const double cell[6] = {c.at("a").get<double>() * klasp::ANGSTROM_TO_BOHR,
                                    c.at("b").get<double>() * klasp::ANGSTROM_TO_BOHR,
                                    c.at("c").get<double>() * klasp::ANGSTROM_TO_BOHR,
                                    c.at("alpha").get<double>(), c.at("beta").get<double>(),
                                    c.at("gamma").get<double>()};
            std::vector<double> tr, rv;
            for (const auto& m : doc.at("molecules"))
                for (int i = 0; i < 3; ++i) {
                    tr.push_back(m.at("translation")[i].get<double>());
                    rv.push_back(m.at("orientation")[i].get<double>());
                }
            const int n_mols = static_cast<int>(doc.at("molecules").size());
            const klasp::CrystalGeometry g =
                klasp::expand_rigid_structure(basis, sg, cell, tr.data(), rv.data(), n_mols);
            // Atoms are laid out operator-major, so the first block is the
            // identity image of every independent molecule.
            const int n_rows = static_cast<int>(g.positions.size()) /
                               klasp::get_space_group(sg).n_symops;

            json out{{"file", path}, {"space_group", sg}, {"rows", n_rows}};
            for (int k : ks) {
                const klasp::FlexDescriptors d =
                    klasp::compute_flex_descriptors({g}, /*want_pdd=*/true, 0.0, k);
                const auto& p = d.pdd.at(0);
                out["pdd"][std::to_string(k)] =
                    std::vector<double>(p.begin(), p.begin() + static_cast<size_t>(n_rows) * k);
            }
            std::printf("%s\n", out.dump().c_str());
        }
    }
    Kokkos::finalize();
    return rc;
}
