# Runs at build time: writes version.hpp from IN_FILE with the current git
# commit and dirty state, only when the contents change.

find_package(Git QUIET)

set(KLASP_GIT_HASH       "unknown")
set(KLASP_GIT_HASH_SHORT "unknown")
set(KLASP_GIT_DIRTY      "false")
set(KLASP_GIT_BRANCH     "unknown")

if(GIT_EXECUTABLE AND EXISTS "${SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY "${SOURCE_DIR}"
        OUTPUT_VARIABLE KLASP_GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY "${SOURCE_DIR}"
        OUTPUT_VARIABLE KLASP_GIT_HASH_SHORT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY "${SOURCE_DIR}"
        OUTPUT_VARIABLE KLASP_GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    # Tracked files only, so untracked files do not mark the build dirty.
    execute_process(
        COMMAND ${GIT_EXECUTABLE} status --porcelain --untracked-files=no
        WORKING_DIRECTORY "${SOURCE_DIR}"
        OUTPUT_VARIABLE _git_status
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(NOT "${_git_status}" STREQUAL "")
        set(KLASP_GIT_DIRTY "true")
    endif()
endif()

string(TIMESTAMP KLASP_BUILD_DATE "%Y-%m-%d %H:%M:%S UTC" UTC)

configure_file("${IN_FILE}" "${OUT_FILE}.tmp" @ONLY)

if(EXISTS "${OUT_FILE}")
    file(READ "${OUT_FILE}"     _prev)
    file(READ "${OUT_FILE}.tmp" _next)
    if("${_prev}" STREQUAL "${_next}")
        file(REMOVE "${OUT_FILE}.tmp")
        return()
    endif()
endif()

file(RENAME "${OUT_FILE}.tmp" "${OUT_FILE}")
