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
