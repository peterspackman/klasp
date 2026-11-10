#pragma once

#include <Kokkos_Core.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

class KokkosListener : public Catch::EventListenerBase {
public:
    using EventListenerBase::EventListenerBase;

    void testRunStarting(Catch::TestRunInfo const&) override {
        Kokkos::initialize();
    }

    void testRunEnded(Catch::TestRunStats const&) override {
        Kokkos::finalize();
    }
};

CATCH_REGISTER_LISTENER(KokkosListener)
