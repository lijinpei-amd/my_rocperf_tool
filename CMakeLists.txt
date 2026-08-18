cmake_minimum_required(VERSION 3.28)
project(my_rocperf_tool)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fno-rtti")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(rocprofiler-sdk REQUIRED)
find_package(LLVM REQUIRED CONFIG)

add_subdirectory(3rd_party)

add_definitions(${LLVM_DEFINITIONS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIR})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(lib)
add_subdirectory(tools)
