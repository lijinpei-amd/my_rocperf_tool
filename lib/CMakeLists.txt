# Core library: offline trace decoding / disassembly. Deliberately free of the
# rocprofiler-sdk runtime (and its libamd_comgr -> libLLVM.so.23 chain) so the
# offline CLI can link a single LLVM. disassembler.h references rocprofiler-sdk
# types in the registerSymbol() signature but calls no rocprofiler-sdk
# functions, so headers-only linkage suffices here.
add_library(my_rocperf_tool SHARED
    att_output_dir.cpp
    disassembler.cpp
    init_llvm.cpp
    loop_detection.cpp
    )
# -external-nolink is an INTERFACE target that carries the HIP/HSA/comgr include
# dirs and compile definitions (e.g. ROCPROFILER_SDK_USE_SYSTEM_RCCL, and the
# HIP headers that define hipStream_t) WITHOUT adding any runtime .so to the
# link line -- so the SDK headers compile but comgr/libLLVM.so.23 stay out.
target_link_libraries(my_rocperf_tool PUBLIC
    rocprofiler-sdk::rocprofiler-sdk-headers
    rocprofiler-sdk::rocprofiler-sdk-external-nolink
    absl::log LLVM SQLite::SQLite3 rocprof-trace-decoder)

# Live tracing library: registers rocprofiler-sdk callbacks to trace code-object
# loads inside a running GPU process. This is the only piece that needs the
# rocprofiler-sdk runtime, so the comgr/LLVM-23 dependency is confined here.
add_library(my_rocperf_tool_live SHARED
    check_rocprofiler_status.cpp
    rocperf_tool.cpp
    )
target_link_libraries(my_rocperf_tool_live PUBLIC
    my_rocperf_tool rocprofiler-sdk::rocprofiler-sdk)
