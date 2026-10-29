# Core library: offline trace decoding / disassembly. Deliberately free of the
# rocprofiler-sdk runtime (and its libamd_comgr -> libLLVM.so.23 chain) so the
# offline CLI can link a single LLVM. disassembler.h takes the SDK's
# code-object load record (rocprofiler_callback_tracing_code_object_load_data_t)
# in the live-tracing add_code_object() overload, and disassembler.cpp reads its
# fields, but nothing here calls a rocprofiler-sdk function, so headers-only
# linkage suffices.
add_library(my_rocperf_tool SHARED
    att_output_dir.cpp
    disassembler.cpp
    loop_detection.cpp
    loop_profile.cpp
    wave_analysis.cpp
    )
# -external-nolink is an INTERFACE target that carries the HIP/HSA/comgr include
# dirs and compile definitions (e.g. ROCPROFILER_SDK_USE_SYSTEM_RCCL, and the
# HIP headers that define hipStream_t) WITHOUT adding any runtime .so to the
# link line -- so the SDK headers compile but comgr/libLLVM.so.23 stay out.
target_link_libraries(my_rocperf_tool PUBLIC
    rocprofiler-sdk::rocprofiler-sdk-headers
    rocprofiler-sdk::rocprofiler-sdk-external-nolink
    absl::log LLVM rocprof-trace-decoder)

# Live tracing library: registers rocprofiler-sdk callbacks to trace code-object
# loads inside a running GPU process. This is the only piece that needs the
# rocprofiler-sdk runtime, so the comgr/LLVM-23 dependency is confined here.
add_library(my_rocperf_tool_live SHARED
    check_rocprofiler_status.cpp
    rocperf_tool.cpp
    )
target_link_libraries(my_rocperf_tool_live PUBLIC
    my_rocperf_tool rocprofiler-sdk::rocprofiler-sdk)
