# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

my_rocperf_tool is a C++17 performance analysis tool for AMD GPUs. It decodes Advanced Thread Trace (ATT) data from rocprofiler and correlates per-instruction execution statistics (stall cycles, duration, idle time) with LLVM MC-based disassembly of AMDGPU code objects.

## Build Commands

```bash
git submodule update --init --recursive
cmake -B build -G Ninja -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build
ctest --test-dir build --output-on-failure
```

CMake minimum 3.28, Ninja generator, C++17, compiled with `-fPIC -fno-rtti`.

### Dependencies

- **rocprofiler-sdk** — AMD ROCm profiler SDK (point `CMAKE_PREFIX_PATH` at the ROCm install)
- **LLVM** (CONFIG mode) — MC disassembler infrastructure
- **rocprof-trace-decoder** — git submodule at `3rd_party/rocm-systems` (built from `projects/rocprof-trace-decoder/source`)
- **abseil-cpp**, **fmt** — git submodules under `3rd_party/`
- **googletest** — git submodule at `3rd_party/googletest` (gtest only, no gmock), for `tests/`

### Build Outputs

- `build/lib/libmy_rocperf_tool.so` — shared library
- `build/tools/load_issue_stall` — CLI tool
- `build/tests/my_rocperf_tool_tests` — unit tests, built by default

## Code Style

Pre-commit hooks enforce clang-format and other checks. Set up with `pre-commit install`, run manually with `pre-commit run --all-files`.

## Architecture

**Library (`lib/` → `libmy_rocperf_tool.so`):**

- **Disassembler / ObjectFileInfo** (`disassembler.h/cpp`) — LLVM MC-based AMDGPU disassembler. Parses ELF code objects, identifies ISA from ELF machine flags (matched against `include/my_rocperf_tool/data/comgr-isa-metadata.def`), caches subtarget info, and provides on-demand instruction decoding via `decode_at(addr)`. Needs no setup: the first construction registers the AMDGPU target (once per process). Not thread-safe (one per thread), and neither copyable nor movable because each MCContext points at the Disassembler's MC options.
- **RocPerfTool** (`rocperf_tool.h/cpp`) — Registers rocprofiler-sdk callbacks for code object load tracing; feeds loaded code objects to the Disassembler.
- **InstStatistics** (`inst_statistics.h`) — Header-only; accumulates per-PC execution statistics (stall, duration, idle, category) from decoded thread trace data.
- **InitLLVM** (`init_llvm.h`) — Header-only, optional process setup for executables: `llvm::InitLLVM`'s crash handlers and stack traces, and `llvm_shutdown` on destruction. Registers no targets and parses no LLVM options.

**CLI Tool (`tools/load_issue_stall.cpp`):**

Reads an ATT output directory containing code object files (`*_code_object_id_*.out`) and ATT trace files (`*.att`). Decodes traces via `rocprof_trace_decoder_parse_data` with SE-data, ISA, and trace callbacks, then dumps annotated disassembly with per-instruction timing stats. Uses `absl::flags` for CLI argument parsing.

The `*_results.db` SQLite databases rocprofv3 writes alongside these files are not read. Code object load addresses are not needed: the trace decoder resolves each hardware PC to a `{code_object_id, ELF vaddr}` pair before the tool sees it, and every address downstream stays in ELF vaddr space.

## Namespace

All project code lives in the `my_rocperf_tool` namespace. LLVM `DenseMapInfo` specializations are in the `llvm` namespace as required.

## Testing

`tests/` holds one googletest executable, `my_rocperf_tool_tests`, linked against the core library. `gtest_discover_tests` registers each test case with CTest, so `ctest --test-dir build --output-on-failure` runs them all (or run the binary directly, with `--gtest_filter` to pick cases).

Tests use module interfaces only:

- `loop_detection_test.cpp` — `detect_loops` on synthetic waves built from plain `rocprofiler_thread_trace_decoder_*` structs: iteration counts, back-edges, per-loop totals, null-PC skipping.
- `disassembler_test.cpp` — `Disassembler` with no `InitLLVM`, concurrent first construction, and compile-time checks that it cannot be copied or moved.
- `att_output_dir_test.cpp` — `AttOutputDir` on temp directories of empty, rocprofv3-named files: prefix parsing, per-subdirectory runs, trace sort order, and the "cannot read <path>: <reason>" failures.

The CLI's end-to-end output is not covered.
