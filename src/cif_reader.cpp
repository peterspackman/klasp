#include "klasp/cif_reader.hpp"

#include "klasp/cell.hpp"
#include "klasp/element.hpp"

#include <gemmi/cif.hpp>
#include <gemmi/smcif.hpp>
#include <gemmi/symmetry.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <map>
#include <vector>

namespace klasp {

namespace {

struct CellAtom {
    double frac[3];
    double cart[3];
    int z;
    int site;
    int symop;
};

// A bond carrying the lattice shift applied to `b` for the two atoms to touch.
struct Bond {
    int a, b;
    int shift[3];
};

void fill_cell(const gemmi::UnitCell& uc, CifCell& out) {
    RealType p[6] = {static_cast<RealType>(uc.a),     static_cast<RealType>(uc.b),
                     static_cast<RealType>(uc.c),     static_cast<RealType>(uc.alpha),
                     static_cast<RealType>(uc.beta),  static_cast<RealType>(uc.gamma)};
    for (int i = 0; i < 6; ++i) out.params[i] = static_cast<double>(p[i]);

    // Shared conversion, so CIF and stored klasp cells are oriented identically.
    RealType m[3][3];
    cell_params_to_matrix(p, m);
    RealType inv[3][3] = {};
    mat3_inv(m, inv);  // zeros on a degenerate cell
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            out.matrix[i * 3 + j] = static_cast<double>(m[i][j]);
            out.inverse[i * 3 + j] = static_cast<double>(inv[i][j]);
        }
    out.volume = static_cast<double>(cell_volume(m));
}

void to_cart(const CifCell& c, const double frac[3], double cart[3]) {
    for (int i = 0; i < 3; ++i)
        cart[i] = c.matrix[i * 3 + 0] * frac[0] + c.matrix[i * 3 + 1] * frac[1]
                + c.matrix[i * 3 + 2] * frac[2];
}

// Expand the asymmetric unit, wrap into [0, 1) and merge coincident images. The
// merge is global across sites, since many CIFs list a whole molecule with the
// full operator set; merges across different sites are counted in `n_coincident`.
std::vector<CellAtom> expand_sites(const gemmi::SmallStructure& st, const CifCell& cell,
                                   const CifReadOptions& opts, int* n_partial,
                                   int* n_sites_kept, int* n_coincident) {
    std::vector<CellAtom> atoms;
    const double merge2 = opts.merge_tolerance * opts.merge_tolerance;
    *n_partial = 0;
    *n_sites_kept = 0;
    *n_coincident = 0;

    // Bucket grid at least as wide as the tolerance: a coincident atom is in this
    // bucket or one of its 26 periodic neighbours.
    int ncell[3];
    for (int i = 0; i < 3; ++i) {
        const double len = cell.params[i];
        ncell[i] = std::max(1, std::min(512, static_cast<int>(len / opts.merge_tolerance)));
    }
    auto bucket_of = [&](const double f[3], int di, int dj, int dk) {
        auto wrap = [](int v, int n) { return ((v % n) + n) % n; };
        const int i = wrap(static_cast<int>(std::floor(f[0] * ncell[0])) + di, ncell[0]);
        const int j = wrap(static_cast<int>(std::floor(f[1] * ncell[1])) + dj, ncell[1]);
        const int k = wrap(static_cast<int>(std::floor(f[2] * ncell[2])) + dk, ncell[2]);
        return (static_cast<long long>(i) * ncell[1] + j) * ncell[2] + k;
    };
    std::map<long long, std::vector<int>> buckets;

    for (std::size_t s = 0; s < st.sites.size(); ++s) {
        const gemmi::SmallStructure::Site& site = st.sites[s];
        const int z = site.element.atomic_number();
        if (z <= 0) continue;                                  // unparseable element
        if (opts.skip_hydrogens && z == 1) continue;
        if (site.occ < opts.min_occupancy) { ++(*n_partial); continue; }
        ++(*n_sites_kept);

        // Identity, then gemmi's `images` (which exclude it).
        for (int op = 0; op <= static_cast<int>(st.cell.images.size()); ++op) {
            gemmi::Fractional f = site.fract;
            if (op > 0) f = st.cell.images[op - 1].apply(site.fract);

            CellAtom a;
            a.frac[0] = f.x - std::floor(f.x);
            a.frac[1] = f.y - std::floor(f.y);
            a.frac[2] = f.z - std::floor(f.z);
            to_cart(cell, a.frac, a.cart);

            int match = -1;
            for (int di = -1; di <= 1 && match < 0; ++di)
                for (int dj = -1; dj <= 1 && match < 0; ++dj)
                    for (int dk = -1; dk <= 1 && match < 0; ++dk) {
                        auto it = buckets.find(bucket_of(a.frac, di, dj, dk));
                        if (it == buckets.end()) continue;
                        for (int k : it->second) {
                            double d[3];
                            for (int i = 0; i < 3; ++i) {
                                d[i] = a.frac[i] - atoms[k].frac[i];
                                d[i] -= std::round(d[i]);
                            }
                            double dc[3];
                            to_cart(cell, d, dc);
                            if (dc[0]*dc[0] + dc[1]*dc[1] + dc[2]*dc[2] < merge2) {
                                match = k;
                                break;
                            }
                        }
                    }
            if (match >= 0) {
                if (atoms[match].site != static_cast<int>(s)) ++(*n_coincident);
                continue;
            }

            a.z = z;
            a.site = static_cast<int>(s);
            a.symop = op;
            buckets[bucket_of(a.frac, 0, 0, 0)].push_back(static_cast<int>(atoms.size()));
            atoms.push_back(a);
        }
    }
    return atoms;
}

// Bonds between cell atoms over every lattice shift that can bring them within
// bonding distance: for fractional offset df, |df_k + s_k| <= threshold |b_k*|.
std::vector<Bond> find_bonds(const std::vector<CellAtom>& atoms, const CifCell& cell,
                             double tolerance) {
    std::vector<Bond> bonds;
    const int n = static_cast<int>(atoms.size());
    std::vector<double> radius(n);
    for (int i = 0; i < n; ++i) radius[i] = element(element_symbol(atoms[i].z)).cov_radius;

    // Lattice vectors are the columns of the cell matrix; the rows of its inverse
    // are the reciprocal vectors.
    double lat[3][3];
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) lat[j][i] = cell.matrix[i * 3 + j];
    double recip[3];
    for (int k = 0; k < 3; ++k)
        recip[k] = std::sqrt(cell.inverse[k * 3 + 0] * cell.inverse[k * 3 + 0]
                           + cell.inverse[k * 3 + 1] * cell.inverse[k * 3 + 1]
                           + cell.inverse[k * 3 + 2] * cell.inverse[k * 3 + 2]);

    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            const double threshold = radius[i] + radius[j] + tolerance;
            const double t2 = threshold * threshold;
            int lo[3], hi[3];
            for (int k = 0; k < 3; ++k) {
                const double df = atoms[j].frac[k] - atoms[i].frac[k];
                const double w = threshold * recip[k];
                lo[k] = static_cast<int>(std::ceil(-df - w));
                hi[k] = static_cast<int>(std::floor(-df + w));
            }
            for (int sa = lo[0]; sa <= hi[0]; ++sa)
                for (int sb = lo[1]; sb <= hi[1]; ++sb)
                    for (int sc = lo[2]; sc <= hi[2]; ++sc) {
                        // A self-bond to an image means a polymer; take only half the shifts for i == j
                        // since -s and +s are the same bond.
                        if (i == j) {
                            if (sa < 0) continue;
                            if (sa == 0 && sb < 0) continue;
                            if (sa == 0 && sb == 0 && sc <= 0) continue;
                        }
                        double d2 = 0.0;
                        for (int k = 0; k < 3; ++k) {
                            const double e = atoms[j].cart[k] + sa * lat[0][k] + sb * lat[1][k]
                                           + sc * lat[2][k] - atoms[i].cart[k];
                            d2 += e * e;
                        }
                        if (d2 >= t2 || d2 < 1e-8) continue;
                        Bond b;
                        b.a = i;
                        b.b = j;
                        b.shift[0] = sa;
                        b.shift[1] = sb;
                        b.shift[2] = sc;
                        bonds.push_back(b);
                    }
        }
    }
    return bonds;
}

// Connected components, unwrapped by accumulating bond shifts along the traversal.
std::vector<CifMolecule> perceive_molecules(const std::vector<CellAtom>& atoms,
                                            const std::vector<Bond>& bonds,
                                            const CifCell& cell) {
    const int n = static_cast<int>(atoms.size());
    std::vector<std::vector<int>> adj(n);   // indices into `bonds`
    for (std::size_t e = 0; e < bonds.size(); ++e) {
        adj[bonds[e].a].push_back(static_cast<int>(e));
        adj[bonds[e].b].push_back(static_cast<int>(e));
    }

    double lat[3][3];
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) lat[j][i] = cell.matrix[i * 3 + j];

    std::vector<int> component(n, -1);
    std::vector<std::array<int, 3>> shift(n, {0, 0, 0});
    std::vector<CifMolecule> molecules;

    std::vector<int> stack;
    for (int seed = 0; seed < n; ++seed) {
        if (component[seed] >= 0) continue;
        const int c = static_cast<int>(molecules.size());
        std::vector<int> members;

        component[seed] = c;
        shift[seed] = {0, 0, 0};
        stack.clear();
        stack.push_back(seed);
        bool periodic = false;
        while (!stack.empty()) {
            const int v = stack.back();
            stack.pop_back();
            members.push_back(v);
            for (int e : adj[v]) {
                const Bond& b = bonds[e];
                // Bond stored as a -> b + shift; reverse traversal subtracts it.
                const bool forward = (b.a == v);
                const int w = forward ? b.b : b.a;
                const int sgn = forward ? 1 : -1;
                int want[3];
                for (int k = 0; k < 3; ++k) want[k] = shift[v][k] + sgn * b.shift[k];
                if (component[w] >= 0) {
                    // Already placed at a different image: the component is periodic, not a molecule.
                    for (int k = 0; k < 3; ++k)
                        if (shift[w][k] != want[k]) periodic = true;
                    continue;
                }
                component[w] = c;
                for (int k = 0; k < 3; ++k) shift[w][k] = want[k];
                stack.push_back(w);
            }
        }

        // Canonical atom order (asymmetric site, then operator) so copies of one molecule
        // are ordered identically.
        std::sort(members.begin(), members.end(), [&atoms](int x, int y) {
            if (atoms[x].site != atoms[y].site) return atoms[x].site < atoms[y].site;
            return atoms[x].symop < atoms[y].symop;
        });

        CifMolecule m;
        m.periodic = periodic;
        m.positions.reserve(members.size() * 3);
        m.numbers.reserve(members.size());
        m.site.reserve(members.size());
        m.symop.reserve(members.size());
        for (int v : members) {
            double p[3];
            for (int k = 0; k < 3; ++k) {
                p[k] = atoms[v].cart[k] + shift[v][0] * lat[0][k] + shift[v][1] * lat[1][k]
                     + shift[v][2] * lat[2][k];
                m.centroid[k] += p[k];
            }
            for (int k = 0; k < 3; ++k) m.positions.push_back(p[k]);
            m.numbers.push_back(atoms[v].z);
            m.site.push_back(atoms[v].site);
            m.symop.push_back(atoms[v].symop);
        }
        const double inv = 1.0 / static_cast<double>(members.size());
        for (int k = 0; k < 3; ++k) m.centroid[k] *= inv;
        molecules.push_back(std::move(m));
    }
    return molecules;
}

bool block_to_structure(const gemmi::cif::Block& block, const std::string& source,
                        const CifReadOptions& opts, CifStructure* out) {
    gemmi::SmallStructure st = gemmi::make_small_structure_from_block(block);
    if (st.cell.a <= 0 || st.cell.b <= 0 || st.cell.c <= 0) return false;
    if (st.sites.empty()) return false;

    out->name = st.name;
    out->source = source;
    fill_cell(st.cell, out->cell);
    if (out->cell.volume < 1e-6) return false;

    if (st.spacegroup) {
        out->space_group = st.spacegroup->number;
        out->space_group_name = st.spacegroup->xhm();
    } else {
        // gemmi could not name the group; the cell images from listed symops are still valid.
        out->space_group = 0;
        out->space_group_name = st.cell.images.empty() ? "P 1 (assumed)" : "unknown";
    }
    out->n_symops = static_cast<int>(st.cell.images.size()) + 1;

    const std::vector<CellAtom> atoms =
        expand_sites(st, out->cell, opts, &out->n_partial_occupancy, &out->n_sites,
                     &out->n_coincident);
    out->n_atoms = static_cast<int>(atoms.size());
    if (atoms.empty()) return false;

    const std::vector<Bond> bonds = find_bonds(atoms, out->cell, opts.bond_tolerance);
    out->molecules = perceive_molecules(atoms, bonds, out->cell);
    return true;
}

std::vector<CifStructure> from_document(const gemmi::cif::Document& doc, const std::string& source,
                                        const CifReadOptions& opts, std::string* error) {
    std::vector<CifStructure> out;
    for (const gemmi::cif::Block& block : doc.blocks) {
        CifStructure s;
        if (block_to_structure(block, source, opts, &s)) out.push_back(std::move(s));
    }
    if (out.empty() && error) *error = source + ": no data block with a usable crystal";
    return out;
}

}  // namespace

std::vector<CifStructure> read_cif(const std::string& path, const CifReadOptions& opts,
                                   std::string* error) {
    if (error) error->clear();
    try {
        const gemmi::cif::Document doc = gemmi::cif::read_file(path);
        return from_document(doc, path, opts, error);
    } catch (const std::exception& e) {
        if (error) *error = path + ": " + e.what();
        return {};
    }
}

CifStructure structure_from_cell(const double cell_params[6], const double* frac,
                                 const int* numbers, int n_atoms,
                                 const CifReadOptions& opts) {
    CifStructure out;
    out.space_group = 1;
    out.space_group_name = "P 1";
    out.n_symops = 1;

    RealType p[6];
    for (int i = 0; i < 6; ++i) p[i] = static_cast<RealType>(cell_params[i]);
    for (int i = 0; i < 6; ++i) out.cell.params[i] = cell_params[i];
    RealType m[3][3];
    cell_params_to_matrix(p, m);
    RealType inv[3][3] = {};
    mat3_inv(m, inv);  // zeros on a degenerate cell
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            out.cell.matrix[i * 3 + j] = static_cast<double>(m[i][j]);
            out.cell.inverse[i * 3 + j] = static_cast<double>(inv[i][j]);
        }
    out.cell.volume = static_cast<double>(cell_volume(m));
    if (out.cell.volume < 1e-6 || n_atoms <= 0) return out;

    std::vector<CellAtom> atoms;
    atoms.reserve(static_cast<std::size_t>(n_atoms));
    for (int i = 0; i < n_atoms; ++i) {
        if (numbers[i] <= 0) continue;
        if (opts.skip_hydrogens && numbers[i] == 1) continue;
        CellAtom a;
        for (int d = 0; d < 3; ++d) {
            const double f = frac[i * 3 + d];
            a.frac[d] = f - std::floor(f);
        }
        to_cart(out.cell, a.frac, a.cart);
        a.z = numbers[i];
        // Computed atoms: each is its own site, with no operator.
        a.site = i;
        a.symop = 0;
        atoms.push_back(a);
    }
    out.n_sites = static_cast<int>(atoms.size());
    out.n_atoms = static_cast<int>(atoms.size());
    if (atoms.empty()) return out;

    const std::vector<Bond> bonds = find_bonds(atoms, out.cell, opts.bond_tolerance);
    out.molecules = perceive_molecules(atoms, bonds, out.cell);
    return out;
}

std::vector<CifStructure> read_cif_string(const std::string& text, const std::string& name,
                                          const CifReadOptions& opts, std::string* error) {
    if (error) error->clear();
    try {
        const gemmi::cif::Document doc =
            gemmi::cif::read_memory(text.data(), text.size(), "string");
        return from_document(doc, name, opts, error);
    } catch (const std::exception& e) {
        if (error) *error = name + ": " + e.what();
        return {};
    }
}

}  // namespace klasp
