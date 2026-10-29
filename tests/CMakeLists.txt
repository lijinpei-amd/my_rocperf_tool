include(GoogleTest)

# Unit tests against the core library's module interfaces. Built by default so
# `cmake --build build` keeps them compiling alongside the CLI.
add_executable(my_rocperf_tool_tests
    att_output_dir_test.cpp
    disassembler_test.cpp
    loop_detection_test.cpp
    )
target_link_libraries(my_rocperf_tool_tests GTest::gtest_main my_rocperf_tool)
gtest_discover_tests(my_rocperf_tool_tests)
