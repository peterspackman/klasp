#pragma once

// Finite-difference check of the analytic gradient, as a subcommand.

#include <string>

namespace klasp {

struct FdcheckOptions {
    std::string input;              // structure JSON
    bool   no_taper = false;
    // Negative: use settings.ewald_accuracy from the input.
    double ewald_accuracy = -1;
    // A component passes if it agrees absolutely or relatively.
    double abs_tol = 1e-5;
    double rel_tol = 1e-3;
    // Cap the multipole rank, as `csp --max-order` does; -1 keeps the input's.
    int    max_order = -1;
};

// Returns 0 when every component agrees, 1 otherwise, so a CI step can use it.
int cmd_fdcheck(const FdcheckOptions& o);

}  // namespace klasp
