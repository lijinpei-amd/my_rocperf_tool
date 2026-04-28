cmake_minimum_required(VERSION 3.28)
project(my_rocperf_tool)

option(ROCPROF_TRACE_DECODER_PATH "installation path of rocprof-trace-decoder")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fno-rtti")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(rocprofiler-sdk REQUIRED)
find_package(LLVM REQUIRED CONFIG)
find_package(SQLite3)

add_subdirectory(3rd_party)

add_definitions(${LLVM_DEFINITIONS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIR})

if(EXISTS "${ROCPROF_TRACE_DECODER_PATH}/include/rocprof-trace-decoder/trace_decoder_api.h")
  include_directories("${ROCPROF_TRACE_DECODER_PATH}/include/rocprof-trace-decoder")
  link_directories("${ROCPROF_TRACE_DECODER_PATH}/lib")
  add_compile_definitions(ROCPROF_TRACE_DECODER_NEW=1)
else()
  include_directories("${ROCPROF_TRACE_DECODER_PATH}/include")
  link_directories("${ROCPROF_TRACE_DECODER_PATH}/releases/linux_glibc_2_28_x86_64/")
  add_compile_definitions(ROCPROF_TRACE_DECODER_NEW=0)
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(lib)
add_subdirectory(tools)
