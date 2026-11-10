#pragma once
// The structure generator contract. A request says which structures to make, a
// DofBatch says where to put them; anything else a generator needs belongs to
// the generator object. DofBatch is device memory a kernel writes into, so it
// must not own, allocate, or hide behind a virtual call.

#include "klasp/types.hpp"

#include <concepts>

namespace klasp {

// Maximum independent molecules in the asymmetric unit (Z'). The DOF layout in
// BatchWorkspace and StructureResult is sized against it.
static constexpr int MAX_INDEP = 8;

// `seed_start` and `n_try` are a half-open window of the space group's global
// seed sequence. The window is rank-independent, so the rank count never
// changes which structures a run visits.
struct GeneratorRequest {
    int space_group = 1;
    int seed_start  = 0;   // first seed of the window
    int n_try       = 0;   // seeds in the window; the yield is at most this
};

// The rigid-body DOFs of up to `capacity` structures, packed into rows [0, N)
// where N is what generate() returned. Rows past N are untouched.
//
// `seeds` is each row's identity downstream (pending pool, database key, resume
// watermark): a fresh generator writes seeds(i) = seed_start + i; a replaying
// one writes back the seeds it was given.
struct DofBatch {
    View2D<RealType> cell_params;    // (capacity, 6)          Bohr, degrees
    View3D<RealType> translations;   // (capacity, n_indep, 3) fractional COM
    View3D<RealType> rot_vectors;    // (capacity, n_indep, 3) angle-axis
    View1D<int>      seeds;          // (capacity)             identity per row
    int capacity = 0;
    int n_indep  = 1;
};

// A generator fills a DofBatch from a GeneratorRequest and returns the number
// of rows written. `generate` is const: one generator serves every block of a run.
template <typename G>
concept StructureGenerator =
    requires(const G& g, const GeneratorRequest& req, DofBatch& out) {
        { g.generate(req, out) } -> std::same_as<int>;
    };

}  // namespace klasp
