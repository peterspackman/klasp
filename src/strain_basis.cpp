// Construction of the symmetry-adapted strain DOF basis; see
// include/klasp/strain_basis.hpp.

#include "klasp/strain_basis.hpp"
#include "klasp/batch.hpp"
#include "klasp/sgdata.hpp"

#include <cmath>

namespace klasp {

// Both entry points delegate to the device-callable build_strain_basis.

// How many DOFs a lattice allows. With fix_angles, the length-only count, which
// truncates the full basis since lengths come first.
static int lattice_max_dof(int lt_int, bool fix_angles) {
    if (!fix_angles) return 6;
    switch (static_cast<LatticeType>(lt_int)) {
        case CUBIC: case RHOMBOHEDRAL:            return 1;
        case TETRAGONAL: case HEXAGONAL:          return 2;
        case ORTHORHOMBIC:                        return 3;
        case MONOCLINIC: case TRICLINIC: default: return 3;
    }
}

int set_strain_basis(int lt_int, RealType* basis, bool fix_angles) {
    // No cell: rhombohedral gets its isotropic DOF only.
    return build_strain_basis(lt_int, nullptr, basis, lattice_max_dof(lt_int, fix_angles));
}

int set_strain_basis_cell(int lt_int, const RealType* cell, RealType* basis,
                          bool fix_angles) {
    return build_strain_basis(lt_int, cell, basis, lattice_max_dof(lt_int, fix_angles));
}

// ---------------------------------------------------------------------------

template <typename Space>
int batch_set_strain_dofs(const BatchGeometryT<Space>& geom,
                          RigidDofStateT<Space>& dofs, int lattice_type,
                          bool fix_angles) {
    const int n_batch = geom.n_batch;
    if ((int)dofs.cell_dof_basis.extent(0) < n_batch)
        dofs.cell_dof_basis = SView3D<RealType, Space>("cell_dof_basis", n_batch, 6, 6);

    // Only rhombohedral needs the per-structure cell; otherwise one basis is
    // built on the host and broadcast.
    auto h_basis = Kokkos::create_mirror_view(dofs.cell_dof_basis);
    int n_dof = 0;
    if (static_cast<LatticeType>(lattice_type) == RHOMBOHEDRAL) {
        auto h_cell = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), geom.cell);
        for (int b = 0; b < n_batch; ++b) {
            RealType cell[9];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) cell[r*3 + c] = h_cell(b, r, c);
            RealType basis[36];
            n_dof = set_strain_basis_cell(lattice_type, cell, basis, fix_angles);
            for (int s = 0; s < 6; ++s)
                for (int v = 0; v < 6; ++v) h_basis(b, s, v) = basis[s*6 + v];
        }
    } else {
        RealType basis[36];
        n_dof = set_strain_basis(lattice_type, basis, fix_angles);
        for (int b = 0; b < n_batch; ++b)
            for (int s = 0; s < 6; ++s)
                for (int v = 0; v < 6; ++v) h_basis(b, s, v) = basis[s*6 + v];
    }
    Kokkos::deep_copy(dofs.cell_dof_basis, h_basis);
    dofs.n_cell_dof = n_dof;
    // Lets the streaming refill rebuild a slot's rows (batch_stream.cpp).
    dofs.cell_dof_lattice = lattice_type;
    return n_dof;
}

template int batch_set_strain_dofs<ExecSpace>(const BatchGeometry&, RigidDofState&, int, bool);

}  // namespace klasp
