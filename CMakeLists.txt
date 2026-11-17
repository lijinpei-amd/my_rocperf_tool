cmake_minimum_required(VERSION 3.28)
project(my_rocperf_tool)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fno-rtti")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

find_package(rocprofiler-sdk REQUIRED)
find_package(LLVM REQUIRED CONFIG)
# LLVM 20 is the first release with gfx950, which the tests assemble and whose
# scheduling model gives its MFMA cycles. Not find_package(LLVM 20): LLVM's
# version file accepts only an exact major.minor match.
if(LLVM_PACKAGE_VERSION VERSION_LESS 20)
  message(FATAL_ERROR
    "LLVM ${LLVM_PACKAGE_VERSION} found in ${LLVM_DIR}; LLVM 20 or newer is "
    "required (point CMAKE_PREFIX_PATH or LLVM_DIR at a newer LLVM)")
endif()

add_subdirectory(3rd_party)

add_definitions(${LLVM_DEFINITIONS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIR})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(tests)
