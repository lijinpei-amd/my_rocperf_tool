add_library(my_rocperf_tool SHARED
    check_rocprofiler_status.cpp
    disassembler.cpp
    init_llvm.cpp
    rocperf_tool.cpp
    )
target_link_libraries(my_rocperf_tool PUBLIC
    rocprofiler-sdk::rocprofiler-sdk absl::log LLVM)
