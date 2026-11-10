# Writes the git commit hash and build date to include/klasp/version.hpp at
# build time, so the hash tracks the checked-out commit.

set(KLASP_VERSION_IN  "${CMAKE_SOURCE_DIR}/cmake/klasp_version.hpp.in")
set(KLASP_VERSION_OUT "${CMAKE_BINARY_DIR}/include/klasp/version.hpp")

add_custom_target(klasp_generate_version ALL
    BYPRODUCTS "${KLASP_VERSION_OUT}"
    COMMAND ${CMAKE_COMMAND}
        -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
        -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DIN_FILE=${KLASP_VERSION_IN}
        -DOUT_FILE=${KLASP_VERSION_OUT}
        -P ${CMAKE_SOURCE_DIR}/cmake/klasp_version_write.cmake
    COMMENT "Generating klasp/version.hpp"
    VERBATIM
)
