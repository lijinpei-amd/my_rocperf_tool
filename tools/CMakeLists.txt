add_executable(load_issue_stall load_issue_stall.cpp)
target_link_libraries(load_issue_stall absl::flags absl::flags_commandlineflag absl::flags_parse absl::flags_reflection absl::flags_usage fmt::fmt my_rocperf_tool rocprof-trace-decoder)
