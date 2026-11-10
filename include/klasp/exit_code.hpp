#pragma once

// Every exit status klasp can return. 0-2 follow shell convention, refusal
// codes are grouped for range tests, and 70 and 75 are sysexits.h EX_SOFTWARE
// and EX_TEMPFAIL.

namespace klasp {

enum class ExitCode : int {
    Ok = 0,
    // An input it cannot read, a model that is not installed. Explained on stderr.
    Failed = 1,
    // Unknown option, value out of range, unparseable config file, or a CLI11
    // parse failure.
    Usage = 2,

    // The run refused to start. Nothing was written.
    // The output directory holds a database from a newer klasp.
    RefusedNewerSchema = 3,
    // --resume was asked for and the recorded settings disagree with this run's,
    // or there is nothing there to resume.
    RefusedResumeMismatch = 4,
    // --resume was asked for and the stored landscape cannot be read back.
    RefusedResumeUnreadable = 5,
    // A fresh run would overwrite an existing landscape.
    RefusedWouldOverwrite = 6,

    // The command's own check failed (fdcheck found a bad gradient), as opposed
    // to a program error.
    CheckFailed = 7,

    // The MPI implementation would not grant MPI_THREAD_MULTIPLE, which rank 0
    // needs to drain results while the work loop runs.
    UnsupportedMpiThreading = 70,
    // Out of wall clock before the work finished. Schedulers can requeue on this.
    OutOfTime = 75,
};

// True for a code meaning "refused to start; nothing was written".
inline bool refused_to_start(int code) {
    return code >= static_cast<int>(ExitCode::RefusedNewerSchema)
        && code <= static_cast<int>(ExitCode::RefusedWouldOverwrite);
}

inline int operator+(ExitCode c) { return static_cast<int>(c); }

}  // namespace klasp
