# Kokkos and KokkosFFT require 3.22.
cmake_minimum_required(VERSION 3.22)
project(klasp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Module scanning needs clang-scan-deps, which ROCm does not ship; set before
# the fetches so subprojects inherit it.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Bakes this machine's instruction set into the objects: turn off when the
# build and run hosts differ or objects are cached across machines (ccache
# hashes the literal "-march=native"). Not applied to CUDA.
option(KLASP_ARCH_NATIVE "Optimize for the building machine's CPU (-march=native)" ON)
if(KLASP_ARCH_NATIVE AND NOT KLASP_BACKEND STREQUAL "cuda")
    add_compile_options(-march=native)
endif()

# --- Python module ---
#
# Declared here because the PIC setting below must be in place before the
# dependencies are fetched. `pip install .` turns this on via scikit-build-core.
option(KLASP_BUILD_PYTHON "Build the nanobind Python module (needs Python headers)" OFF)
if(KLASP_BUILD_PYTHON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# --- CPM.cmake ---
# Version in the filename, so a bump is not ignored on an existing build tree.
set(CPM_DOWNLOAD_VERSION 0.43.1)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT EXISTS "${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
        "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
        "${CPM_DOWNLOAD_LOCATION}"
    )
endif()
include(${CPM_DOWNLOAD_LOCATION})

# --- Kokkos backend selection ---
# Set via -DKLASP_BACKEND=serial|openmp|hip|cuda (default: serial).
#
# CPM's OPTIONS are not cache variables, so every flag is a FORCE'd cache var
# for KokkosFFT's find_package(Kokkos) to see.
set(KLASP_BACKEND "serial" CACHE STRING "Kokkos backend: serial, openmp, hip, cuda")

set(Kokkos_ENABLE_SERIAL ON  CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_HIP    OFF CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_CUDA   OFF CACHE BOOL "" FORCE)

if(KLASP_BACKEND STREQUAL "openmp")
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
elseif(KLASP_BACKEND STREQUAL "hip")
    set(Kokkos_ENABLE_HIP    ON CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
elseif(KLASP_BACKEND STREQUAL "cuda")
    set(Kokkos_ENABLE_CUDA           ON CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_CUDA_LAMBDA    ON CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_OPENMP         ON CACHE BOOL "" FORCE)
endif()

CPMAddPackage(
    NAME Kokkos
    GITHUB_REPOSITORY kokkos/kokkos
    GIT_TAG 5.2.2
)

# --- KokkosFFT (wraps cuFFT/FFTW/rocFFT via Kokkos backend) ---
# Requires FFTW on CPU or cuFFT on CUDA. Only enable when available.
if(KLASP_BACKEND STREQUAL "cuda")
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
    set(KokkosFFT_ENABLE_CUFFT ON CACHE BOOL "" FORCE)
    set(KokkosFFT_ENABLE_FFTW OFF CACHE BOOL "" FORCE)
    CPMAddPackage(
        NAME KokkosFFT
        GITHUB_REPOSITORY kokkos/kokkos-fft
        GIT_TAG v2.0.0
        OPTIONS
            "KokkosFFT_ENABLE_INTERNAL_KOKKOS OFF"
            "KokkosFFT_ENABLE_EXAMPLES OFF"
            "KokkosFFT_ENABLE_TESTS OFF"
            "KokkosFFT_ENABLE_BENCHMARK OFF"
            "KokkosFFT_ENABLE_CUFFT ON"
            "KokkosFFT_ENABLE_FFTW OFF"
    )
    set(KLASP_HAS_KOKKOSFFT ON)
elseif(KLASP_BACKEND STREQUAL "hip")
    set(KokkosFFT_ENABLE_HIPFFT ON CACHE BOOL "" FORCE)
    set(KokkosFFT_ENABLE_FFTW OFF CACHE BOOL "" FORCE)
    CPMAddPackage(
        NAME KokkosFFT
        GITHUB_REPOSITORY kokkos/kokkos-fft
        GIT_TAG v2.0.0
        OPTIONS
            "KokkosFFT_ENABLE_INTERNAL_KOKKOS OFF"
            "KokkosFFT_ENABLE_EXAMPLES OFF"
            "KokkosFFT_ENABLE_TESTS OFF"
            "KokkosFFT_ENABLE_BENCHMARK OFF"
            "KokkosFFT_ENABLE_HIPFFT ON"
            "KokkosFFT_ENABLE_FFTW OFF"
    )
    set(KLASP_HAS_KOKKOSFFT ON)
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW3 QUIET fftw3 fftw3f)
    endif()
    if(FFTW3_FOUND)
        CPMAddPackage(
            NAME KokkosFFT
            GITHUB_REPOSITORY kokkos/kokkos-fft
            GIT_TAG v2.0.0
            OPTIONS
                "KokkosFFT_ENABLE_INTERNAL_KOKKOS OFF"
                "KokkosFFT_ENABLE_EXAMPLES OFF"
                "KokkosFFT_ENABLE_TESTS OFF"
                "KokkosFFT_ENABLE_BENCHMARK OFF"
        )
        set(KLASP_HAS_KOKKOSFFT ON)
    else()
        message(STATUS "FFTW not found — the fft_pack generator will use the pocketfft fallback")
        set(KLASP_HAS_KOKKOSFFT OFF)
    endif()
endif()

# --- nlohmann/json ---
CPMAddPackage(
    NAME nlohmann_json
    VERSION 3.12.0
    GITHUB_REPOSITORY nlohmann/json
    OPTIONS "JSON_BuildTests OFF"
)

# --- CLI11 ---
CPMAddPackage("gh:CLIUtils/CLI11@2.7.2")

# --- fmt ---
# FMT_INSTALL off: klasp does not install fmt.
# FMT_MODULE forced off: fmt enables it under CMake 3.28 + Ninja + clang 16,
# and the module scan then fails.
# Held at 11: nvcc 13 cannot compile fmt 12's format.cc.
set(FMT_MODULE OFF CACHE BOOL "" FORCE)
CPMAddPackage(
    NAME fmt
    GITHUB_REPOSITORY fmtlib/fmt
    GIT_TAG 11.1.4
    OPTIONS "FMT_INSTALL OFF" "FMT_MODULE OFF"
)

# --- gemmi (CIF/structure handling) ---
# Bypass gemmi's CMakeLists and expose only the include directory.
CPMAddPackage(
    NAME gemmi
    GITHUB_REPOSITORY project-gemmi/gemmi
    GIT_TAG v0.7.5
    DOWNLOAD_ONLY YES
)
if(gemmi_ADDED)
    add_library(gemmi::headers INTERFACE IMPORTED GLOBAL)
    target_include_directories(gemmi::headers INTERFACE "${gemmi_SOURCE_DIR}/include")
    # Space-group tables live in this one translation unit.
    add_library(gemmi_symmetry STATIC "${gemmi_SOURCE_DIR}/src/symmetry.cpp")
    target_link_libraries(gemmi_symmetry PUBLIC gemmi::headers)
endif()

# --- SQLite modern C++ wrapper ---
CPMAddPackage(
    NAME sqlite_modern_cpp
    GITHUB_REPOSITORY SqliteModernCpp/sqlite_modern_cpp
    GIT_TAG 6e3009973025e0016d5573529067714201338c80
    DOWNLOAD_ONLY YES
)
# --- SQLite3 amalgamation ---
# URL and SHA3-256 come from the sqlite-amalgamation PRODUCT line at
# https://sqlite.org/download.html.
CPMAddPackage(
    NAME sqlite3
    VERSION 3.53.4
    URL https://sqlite.org/2026/sqlite-amalgamation-3530400.zip
    URL_HASH SHA3_256=628a44cfe82c66aed1ccbbe85a562d2e33ebe64b3288981ed76285612227934e
    DOWNLOAD_ONLY YES
)
add_library(sqlite3 STATIC ${sqlite3_SOURCE_DIR}/sqlite3.c)
target_include_directories(sqlite3 PUBLIC ${sqlite3_SOURCE_DIR})
target_compile_definitions(sqlite3 PRIVATE SQLITE_THREADSAFE=1)
set_target_properties(sqlite3 PROPERTIES C_STANDARD 11)
# Silence third-party warnings (e.g. a GCC 14 stringop-overread false positive)
# for this target only.
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sqlite3 PRIVATE -w)
endif()
if(sqlite_modern_cpp_ADDED)
    add_library(sqlite_modern_cpp INTERFACE)
    target_include_directories(sqlite_modern_cpp INTERFACE ${sqlite_modern_cpp_SOURCE_DIR}/hdr)
    target_link_libraries(sqlite_modern_cpp INTERFACE sqlite3)
endif()

# --- MPI ---
find_package(MPI REQUIRED)

include(GNUInstallDirs)

# --- Precision selection ---
option(KLASP_SINGLE_PRECISION "Use float instead of double" OFF)
if(KLASP_SINGLE_PRECISION)
    add_compile_definitions(KLASP_SINGLE_PRECISION)
endif()
if(KLASP_BACKEND STREQUAL "hip" OR KLASP_BACKEND STREQUAL "cuda")
    add_compile_definitions(KLASP_HAS_DEVICE_SPACE)
endif()

# --- Warnings ---
# Applied to klasp's own targets only. No -Werror: three vendor toolchains
# compile the kernels.
add_library(klasp_warnings INTERFACE)
if(MSVC)
    target_compile_options(klasp_warnings INTERFACE /W3)
else()
    target_compile_options(klasp_warnings INTERFACE
        -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare)
endif()

# --- PET Kokkos backend (ML relaxation stage) ---
# Consumed through src/pet_bridge.cpp only. To develop against a checkout:
#   -D CPM_pet_kokkos_SOURCE=/path/to/pet-kokkos
# Passes through PET_PRECISION (mixed | fp32 | fp64), PET_WITH_CUTLASS,
# PET_WITH_VESIN.
option(KLASP_ENABLE_PET "Build the PET-MAD Kokkos energy/force backend" ON)
if(KLASP_ENABLE_PET)
    CPMAddPackage(
        NAME pet_kokkos
        GITHUB_REPOSITORY peterspackman/pet-kokkos
        GIT_TAG a7781b216e67b9ca15e96f38afee647a1d09ba72
        OPTIONS "PET_ARCH_NATIVE ${KLASP_ARCH_NATIVE}"
    )
endif()

# --- Library ---
#
# klasp_core holds everything except MPI; klasp_mpi supplies the MPI
# implementations of WorkSource / ResultSink / Comm and is the only target that
# links libmpi. klasp-energy and the Python module depend on that split.
add_library(klasp_core
    src/types.cpp
    src/config.cpp
    src/log.cpp
    src/cell.cpp
    src/ewald.cpp
    src/interp.cpp
    src/molecule.cpp
    src/chain_rule.cpp
    src/io.cpp
    src/potentials.cpp
    src/batch.cpp
    src/csp.cpp
    src/csp_run.cpp
    src/sg_worker.cpp
    src/result_consumer.cpp
    src/status.cpp
    src/database.cpp
    src/run_state.cpp
    src/descriptors.cpp
    src/compack.cpp
    src/wyckoff.cpp
    src/cmd_wyckoff.cpp
    src/cif_reader.cpp
    src/cif_packing.cpp
    src/progress.cpp
    src/flex_descriptors.cpp
    src/powder.cpp
    src/batch_shared.cpp
    src/batch_lbfgs_step.cpp
    src/batch_kernels_asymunit.cpp
    src/batch_kernels_tabulated.cpp
    src/batch_kernels_rigid_press.cpp
    src/batch_stream.cpp
    src/batch_stream_mc.cpp
    src/fft_generator.cpp
    src/strain_ad.cpp
    src/strain_basis.cpp
    src/sgdata.cpp
    src/symmetry_ops.cpp
    src/niggli.cpp
    src/db_tool.cpp
    src/landscape_generator.cpp
    src/csp_cif.cpp
    src/cmd_energy.cpp
    src/ff_structure.cpp
    src/structure_generation.cpp
    src/cmd_fdcheck.cpp
    src/kernel_profiler.cpp
    src/device_relax.cpp
)
if(KLASP_ENABLE_PET)
    target_sources(klasp_core PRIVATE src/pet_bridge.cpp src/pet_relax.cpp src/cmd_rerank.cpp)
endif()
# Only pet_bridge.cpp includes pet-kokkos headers.
target_include_directories(klasp_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_SOURCE_DIR}/src)
# Fetched and vendored dependencies are not installed, so they stay behind
# BUILD_INTERFACE; Kokkos exports itself.
target_link_libraries(klasp_core PUBLIC
    Kokkos::kokkos
    $<BUILD_INTERFACE:fmt::fmt>
    $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
    $<BUILD_INTERFACE:sqlite_modern_cpp>
    $<BUILD_INTERFACE:gemmi::headers>
    $<BUILD_INTERFACE:gemmi_symmetry>)
target_link_libraries(klasp_core INTERFACE klasp_warnings)
target_link_libraries(klasp_core PRIVATE klasp_warnings)
# Required for the Python module (a shared object).
set_target_properties(klasp_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(KLASP_ENABLE_PET)
    target_link_libraries(klasp_core PUBLIC $<BUILD_INTERFACE:pet_kokkos>)
    target_compile_definitions(klasp_core PUBLIC KLASP_HAS_PET)
    # Install-time model directory; one entry in pet_model_search_dirs().
    # Weights are not shipped.
    target_compile_definitions(klasp_core PRIVATE
        KLASP_INSTALL_MODEL_DIR="${CMAKE_INSTALL_FULL_DATADIR}/klasp/models")
endif()

# The MPI driver.
add_library(klasp_mpi src/orchestrator.cpp)
target_link_libraries(klasp_mpi PUBLIC klasp_core MPI::MPI_CXX)
target_link_libraries(klasp_mpi PRIVATE klasp_warnings)
set_target_properties(klasp_mpi PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- Git version info (generated at build time into klasp/version.hpp) ---
include(${CMAKE_SOURCE_DIR}/cmake/klasp_version.cmake)
add_dependencies(klasp_core klasp_generate_version)
if(KLASP_HAS_KOKKOSFFT)
    target_link_libraries(klasp_core PUBLIC KokkosFFT::fft)
    target_compile_definitions(klasp_core PUBLIC KLASP_HAS_KOKKOSFFT)
endif()

# --- The command line ---
# Subcommand options and handlers, shared by both programs and the tests.
# Links no MPI.
add_library(klasp_cli STATIC
    src/cli/csp_command.cpp
    src/cli/local_commands.cpp
    src/cli/front_end.cpp
)
target_include_directories(klasp_cli PUBLIC ${CMAKE_SOURCE_DIR}/src/cli)
target_link_libraries(klasp_cli PUBLIC klasp_core nlohmann_json::nlohmann_json CLI11::CLI11 fmt::fmt)
target_link_libraries(klasp_cli PRIVATE klasp_warnings)

add_executable(klasp src/main.cpp)
target_link_libraries(klasp PRIVATE klasp_cli klasp_mpi MPI::MPI_CXX)

# --- energy-only CLI, without MPI ---
# Links no MPI, so it can be spawned from inside a running MPI job.
add_executable(klasp-energy src/main_energy.cpp)
target_link_libraries(klasp-energy PRIVATE klasp_cli)

# --- Developer tools ---
# Benchmarks and validation drivers, not installed or run by ctest.
option(KLASP_BUILD_TOOLS "Build the developer tools: the PET validation and benchmark drivers, and the PDD dump" OFF)

# --- PET developer tools ---
if(KLASP_ENABLE_PET)
    if(KLASP_BUILD_TOOLS)
        add_executable(klasp_pet_smoke tests/pet_smoke.cpp)
        target_link_libraries(klasp_pet_smoke PRIVATE klasp_core nlohmann_json::nlohmann_json CLI11::CLI11 Kokkos::kokkos)

        # Times a whole batched relaxation to tolerance.
        add_executable(klasp_pet_relax_bench tests/pet_relax_bench.cpp)
        target_link_libraries(klasp_pet_relax_bench PRIVATE klasp_core nlohmann_json::nlohmann_json CLI11::CLI11 Kokkos::kokkos gemmi::headers)
    endif()
endif()

if(KLASP_BUILD_TOOLS)
    add_executable(klasp_pdd tests/pdd_tool.cpp)
    target_link_libraries(klasp_pdd PRIVATE klasp_core nlohmann_json::nlohmann_json CLI11::CLI11 Kokkos::kokkos)
endif()

# --- Tests ---
option(KLASP_BUILD_TESTS "Build tests" ON)
if(KLASP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# --- Python module ---
#
# Added after the tests block: the module registers a ctest, which needs
# enable_testing() first.
if(KLASP_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# --- Install ---
install(TARGETS klasp klasp-energy
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(TARGETS klasp_core klasp_mpi klasp_warnings
    EXPORT klaspTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Vendored archives klasp_core needs at link time; installed, not exported.
install(TARGETS sqlite3 ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(KLASP_ENABLE_PET)
    install(TARGETS pet_kokkos ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

install(DIRECTORY include/klasp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)
# version.hpp is generated, so it is not in include/ to be picked up above.
install(FILES ${CMAKE_BINARY_DIR}/include/klasp/version.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/klasp
    OPTIONAL
)

install(EXPORT klaspTargets
    FILE klaspTargets.cmake
    NAMESPACE klasp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/klasp
)

# Minimal config: find_dependency reaches Kokkos only. A consumer supplies
# headers for fmt, nlohmann_json and gemmi, and links libsqlite3.a /
# libpet_kokkos.a from ${CMAKE_INSTALL_LIBDIR}.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/klaspConfig.cmake
"include(CMakeFindDependencyMacro)
find_dependency(Kokkos)
include(\"\${CMAKE_CURRENT_LIST_DIR}/klaspTargets.cmake\")
")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/klaspConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/klasp
)

install(DIRECTORY example-inputs/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/klasp/examples
    FILES_MATCHING PATTERN "*.json"
)
