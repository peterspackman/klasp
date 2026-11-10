#pragma once

// The collectives a CSP run performs. SerialComm is the single-process case.

#include <vector>

namespace klasp {

class Comm {
public:
    virtual ~Comm() = default;

    virtual int rank() const = 0;
    virtual int size() const = 0;

    virtual void barrier() = 0;

    virtual void broadcast(int* v, int n, int root) = 0;
    virtual void broadcast(long long* v, int n, int root) = 0;
    virtual void broadcast(double* v, int n, int root) = 0;

    // Wait for every rank to reach the end of the run; give up on the job after
    // `budget_s` if one never does.
    virtual void exit_rendezvous(double budget_s) = 0;
};

// One process, so every agreement is already reached.
class SerialComm : public Comm {
public:
    int rank() const override { return 0; }
    int size() const override { return 1; }
    void barrier() override {}
    void broadcast(int*, int, int) override {}
    void broadcast(long long*, int, int) override {}
    void broadcast(double*, int, int) override {}
    void exit_rendezvous(double) override {}
};

}  // namespace klasp
