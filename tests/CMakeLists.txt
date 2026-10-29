include(GoogleTest)

# Code objects for disassembler_test.cpp, assembled by the llvm-mc of the LLVM
# the library links, so the encodings are the ones its disassembler expects.
# Relocatable objects suffice: ObjectFileInfo reads only e_flags and .text,
# and there is no linker to turn them into executables. llvm-objcopy stands in
# for one where a test needs .text placed as a linked code object has it.
find_program(LLVM_MC llvm-mc PATHS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH
             REQUIRED)
find_program(LLVM_OBJCOPY llvm-objcopy PATHS "${LLVM_TOOLS_BINARY_DIR}"
             NO_DEFAULT_PATH REQUIRED)
set(test_code_object_dir "${CMAKE_CURRENT_BINARY_DIR}/code_objects")
set(test_code_objects)
# Assembles code_objects/<name>.s for `mcpu` into <name>.o, then applies any
# further arguments to it as llvm-objcopy options.
function(add_test_code_object name mcpu)
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/code_objects/${name}.s")
  set(obj "${test_code_object_dir}/${name}.o")
  set(objcopy)
  if(ARGN)
    set(objcopy COMMAND "${LLVM_OBJCOPY}" ${ARGN} "${obj}")
  endif()
  add_custom_command(
      OUTPUT "${obj}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${test_code_object_dir}"
      COMMAND "${LLVM_MC}" -triple=amdgcn-amd-amdhsa -mcpu=${mcpu}
              -filetype=obj "${src}" -o "${obj}"
      ${objcopy}
      DEPENDS "${src}"
      COMMENT "Assembling test code object ${name}.o"
      VERBATIM)
  set(test_code_objects ${test_code_objects} "${obj}" PARENT_SCOPE)
endfunction()
add_test_code_object(gfx942 gfx942)
add_test_code_object(gfx1200 gfx1200)
# The address changes match section names as assembled, before the rename.
add_test_code_object(gfx942_two_texts gfx942
    --change-section-address .text=0x2000
    --change-section-address .text.low=0x1000
    --rename-section .text.low=.text)
add_custom_target(test_code_objects DEPENDS ${test_code_objects})

# Unit tests against the core library's module interfaces. Built by default so
# `cmake --build build` keeps them compiling alongside the CLI.
add_executable(my_rocperf_tool_tests
    att_output_dir_test.cpp
    disassembler_test.cpp
    loop_detection_test.cpp
    loop_profile_test.cpp
    loop_report_test.cpp
    wave_analysis_test.cpp
    )
add_dependencies(my_rocperf_tool_tests test_code_objects)
target_compile_definitions(my_rocperf_tool_tests PRIVATE
    TEST_CODE_OBJECT_DIR="${test_code_object_dir}")
target_link_libraries(my_rocperf_tool_tests GTest::gtest_main my_rocperf_tool)
gtest_discover_tests(my_rocperf_tool_tests)
