#include "klasp/types.hpp"
#include "klasp/ewald.hpp"

namespace klasp {

Workspace allocate_workspace(int max_atoms, int max_shifts,
                             int max_kvecs, int max_types,
                             int max_multipole_rank) {
    Workspace ws;
    ws.positions   = View2D<RealType>("positions",   max_atoms, 3);
    ws.charges     = View1D<RealType>("charges",     max_atoms);
    ws.types       = View1D<int>("types",            max_atoms);
    ws.cell_matrix = View2D<RealType>("cell_matrix", 3, 3);

    ws.shifts      = View2D<int>("shifts",           max_shifts, 3);
    ws.shift_cart  = View2D<RealType>("shift_cart",  max_shifts, 3);

    ws.kvecs       = View2D<RealType>("kvecs",       max_kvecs, 3);
    ws.kprefactors = View1D<RealType>("kprefactors", max_kvecs);

    ws.buck_A = View2D<RealType>("buck_A", max_types, max_types);
    ws.buck_B = View2D<RealType>("buck_B", max_types, max_types);
    ws.buck_C = View2D<RealType>("buck_C", max_types, max_types);
    ws.buck_Rmin = View2D<RealType>("buck_Rmin", max_types, max_types);

    ws.mol_id = View1D<int>("mol_id", max_atoms);
    Kokkos::deep_copy(ws.mol_id, -1);  // -1 = no exclusion

    ws.max_multipole_rank = max_multipole_rank;
    if (max_multipole_rank > 0) {
        ws.multipoles = View2D<RealType>("multipoles", max_atoms, 35);
        ws.dipoles = View2D<RealType>("dipoles", max_atoms, 3);
    }

    ws.n_atoms = 0;
    ws.n_shifts = 0;
    ws.n_kvecs = 0;
    ws.n_types = 0;
    ws.kappa = 0.0;
    ws.rcut = 0.0;
    ws.buck_cutoff = 0.0;
    ws.ewald_accuracy = DEFAULT_EWALD_ACCURACY;

    return ws;
}

}  // namespace klasp
