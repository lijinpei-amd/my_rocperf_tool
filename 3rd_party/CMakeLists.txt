add_subdirectory(abseil-cpp EXCLUDE_FROM_ALL)

set(FMT_TEST OFF CACHE BOOL "Build fmt test suite (off: bundled fmt)")
set(FMT_INSTALL OFF CACHE BOOL "Generate fmt install rules (off: bundled fmt)")
set(FMT_DOC OFF CACHE BOOL "Build fmt docs (off: bundled fmt)")
# Force static fmt regardless of project-wide BUILD_SHARED_LIBS so the tool
# has no libfmt.so runtime dependency. Directory-scope set (no CACHE) is
# inherited by add_subdirectory but doesn't pollute the parent.
set(BUILD_SHARED_LIBS OFF)
# Per fmt docs (https://fmt.dev/latest/get-started/#cmake): build static fmt
# with PIC so it can be linked into shared libraries (libmy_rocperf_tool.so).
# Independent of the project's global -fPIC flag.
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
add_subdirectory(fmt EXCLUDE_FROM_ALL)

# rocprof-trace-decoder lives in the ROCm/rocm-systems monorepo. Build the
# library subdirectory directly so we depend on neither the upstream top-level
# CMakeLists (docs/coverage/packaging) nor a system install.
set(TTD_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocm-systems/projects/rocprof-trace-decoder"
    CACHE INTERNAL "rocprof-trace-decoder source root")
# Suppress the upstream CPack packaging include — we only need the lib target.
set(TTD_ALL_VERSIONS ON)
list(APPEND CMAKE_MODULE_PATH "${TTD_PROJECT_SOURCE_DIR}/cmake")
add_subdirectory(rocm-systems/projects/rocprof-trace-decoder/source rocprof-trace-decoder EXCLUDE_FROM_ALL)
