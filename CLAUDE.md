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
- **LLVM** (CONFIG mode) — MC disassembler infrastructure; the tests also need its `llvm-mc` and `llvm-objcopy` (from `LLVM_TOOLS_BINARY_DIR`), which build the test code objects
- **rocprof-trace-decoder** — git submodule at `3rd_party/rocm-systems` (built from `projects/rocprof-trace-decoder/source`)
- **abseil-cpp**, **fmt** — git submodules under `3rd_party/`
- **googletest** — git submodule at `3rd_party/googletest` (gtest only, no gmock), for `tests/`

### Build Outputs

- `build/lib/libmy_rocperf_tool.so` — core library
- `build/lib/libmy_rocperf_tool_live.so` — live tracing library
- `build/tools/load_issue_stall` — CLI tool
- `build/tests/my_rocperf_tool_tests` — unit tests, built by default

## Code Style

Pre-commit hooks enforce clang-format and other checks. Set up with `pre-commit install`, run manually with `pre-commit run --all-files`.

## Architecture

Domain terms (run, trace, wave, record, loop record, bubble, ...) are defined in `CONTEXT.md`. Headers are in `include/my_rocperf_tool/`, unit tests in `tests/`.

**Core library (`lib/` → `libmy_rocperf_tool.so`):** offline decoding and analysis. It uses only rocprofiler-sdk's headers, so the CLI links a single LLVM.

- **Disassembler / ObjectFileInfo** (`disassembler.h/cpp`) — LLVM MC-based AMDGPU disassembler. `Disassembler::add_code_object` loads an ELF code object under its id, from a file or a live load record; `find_code_object(id)` returns its `ObjectFileInfo`, or null for an id never loaded. An `ObjectFileInfo` answers questions about the instruction at an ELF vaddr: `size_at`, `kind_at` (`InstKind` from `inst_kind.h`: MFMA, classic s_barrier, other), `text_at` / `copy_text_at` (LLVM's printed text, which the trace decoder classifies, so its bytes are part of the contract), and `inst_vaddrs` lists a range's instructions. A vaddr that is not an instruction's start in a `.text` section asserts. A load fails with an `llvm::Error` "<path>: <reason>" (unreadable, not AMDGPU, or an ISA missing from `data/comgr-isa-metadata.def`) and leaves the id unknown. Each `.text` section is decoded whole on first query, with one MC stack and opcode→kind table per ISA. Needs no setup: the first construction registers the AMDGPU target, once per process. Not thread-safe (one per thread), and neither copyable nor movable, because MC objects keep the address of its MC options. Tests: `disassembler_test.cpp`.
- **AttOutputDir** (`att_output_dir.h/cpp`) — `scan` walks an ATT output directory tree for code objects (`<prefix>_<arch>_code_object_id_<id>.out`) and traces (`<prefix>_<agent>_shader_engine_<se>_<dispatch>.att`), parsed into sorted `CodeObjectPath` / `AttPath` lists. The prefix, qualified by the file's subdirectory, names the run; code object and dispatch ids are unique only within a run, and traces sort by run first. `scan` fails if any directory cannot be listed, since a partial scan would drop whole runs; a file whose target cannot be stat'ed is kept, so reading it reports why. Both `scan` and `read_att_data` fail with "cannot read <path>: <reason>". Tests: `att_output_dir_test.cpp`.
- **detect_loops** (`loop_detection.h/cpp`) — Finds a wave's loops by back-edge and sums `LoopStats` over each loop's iterations, keeping those with 2 or more. Expects coalesced s_barrier records. `BackEdge::contains(pc)` is the one body-range test and `LoopStats::operator+=` the one merge. Tests: `loop_detection_test.cpp`.
- **WaveAnalysis** (`wave_analysis.h/cpp`) — Analyzes one trace's waves, streamed in by `add_wave` one at a time (the decoder owns a wave only during its callback, so none is kept). Takes plain `WaveAnalysisOptions` (SIMD filter, loop detection, MFMA co-exec, per-wave loop rows, `DurationCheck` off/warn/error) and an `InstClassifier` (PC → `InstKind`), asked once per PC; the CLI wraps `ObjectFileInfo::kind_at`, tests pass a table. Each wave goes through the SIMD filter, s_barrier token coalescing, the duration check, `detect_loops`, then the `LoopProfile`, in that order. `std::move(analysis).finish()` returns a `WaveAnalysisResult`: `WaveCounts`, the `LoopProfile`, each wave's loops (numbered by `WaveLoopInfo::wave_index`, since waves reuse cu/simd/wave_id slots), diagnostics as data (duration < stall under warn, MFMA shadow overlaps) and, under error, a `DurationCheckFailure`. It prints nothing. Tests: `wave_analysis_test.cpp`.
- **LoopProfile** (`loop_profile.h/cpp`) — The per-loop, per-instruction metrics of one trace, as running sums: memory grows with the loop bodies and the waves' loops, not with the records. Waves are added one at a time, each with the loops `detect_loops` found in it: `add_wave` walks the wave's records once per loop, skipping those outside the loop's body, and sums count, duration, stall, idle, issue, relative issue and, under `mfma_coexec`, bubbles; it returns MFMA shadow overlaps. `metrics(LoopRecord, LoopBody)` answers for one loop record, a wave's (`LoopRecord::wave` = its `wave_index`) or the aggregate (`nullopt`), with a `LoopInstRow` per body instruction and a `LoopSummary` of per-iteration averages and MFMA occupancy. Every number counts that record's own executions only: one wave's, or those of the waves where the loop was detected. Without `wave_rows` a wave's record gets its summary but no rows; the CLI sets it only when `--dump_loop_contents` prints per-wave records, so not under `--aggregate_wave_only`. The `LoopBody` is plain data that the CLI lists from `ObjectFileInfo`, so the module needs no code object. Defines `kMfmaShadowCycles` and `kAttTimestampGranule`. Tests: `loop_profile_test.cpp`.
- **LoopReport** (`loop_report.h/cpp`) — Renders loop records as text, CSV or JSON, under `LoopOutputConfig`. Its input, `TraceLoopReport`, is one trace resolved up front as plain data: run, dispatch and SE, `WaveCounts`, the SIMD filter, and `LoopGroup`s (a wave's, or the aggregate) of `ReportedLoop`s, whose rows view the loop body's text (it must outlive the report). `render_loop_report` appends one trace's report to its `LoopReportChunk` and depends only on its arguments, so traces render on the CLI's workers; `write_loop_report_document` frames the chunks in trace order: CSV header, JSON brackets, and `,\n` between non-empty JSON chunks. The formats differ on purpose in names, visibility and missing values (JSON omits keys, CSV leaves fields blank, text pads); one column list yields the CSV header and every row. In text, a trace with no group gets a "no loops detected" banner. Tests: `loop_report_test.cpp`.
- **trace_pc.h** — `DenseMapInfo` for the decoder's `{vaddr, code object}` PC.

**Live library (`lib/` → `libmy_rocperf_tool_live.so`):** `RocPerfTool` (`rocperf_tool.h/cpp`) registers rocprofiler-sdk code object load tracing and feeds each loaded code object to a Disassembler. The only piece linking the rocprofiler-sdk runtime. Its `rocprofiler_configure` entry point is commented out, so nothing instantiates it yet. No tests.

**CLI Tool (`tools/load_issue_stall.cpp`):**

Reports the loops of every trace under `--att_output_dir`. `main` parses and checks the `absl::flags` and maps them once to `WaveAnalysisOptions`; `loop_output_config_from_flags` maps them to `LoopOutputConfig`. `run_main` scans the directory with `AttOutputDir` and requires code objects for every run. `run_trace` drives `rocprof_trace_decoder_parse_data` for one trace: the SE-data callback hands over the trace, the ISA callback answers from the run's Disassembler, and the trace callback streams each wave into a `WaveAnalysis`. It then resolves the trace's `TraceLoopReport` (`build_loop_groups`, with `LoopBodies` listing each loop body once per trace) and renders it into the trace's chunk. A trace fails, adding nothing to the report, if it is unreadable, a code object of its run failed to load, it names a code object its run lacks, or the duration check rejects it.

Traces decode serially until one reaches the decoder (its lazy trie fill races otherwise), then on `--jobs` workers, each building a Disassembler per run on its first trace of that run. Each trace's stderr text is buffered and printed in trace order, ahead of the stdout document, so neither depends on `--jobs`. A `[summary]` on stderr follows: totals, failed traces by cause, traces that decoded no waves, `--simd` skips and loop-free dispatches. Exit status is 2 when the tool's own checks reject a flag (`--att_output_dir` missing, `--simd` out of range, a bad `--verify_trace_duration`) or the directory (unreadable, no traces, a run without code objects), 1 if any trace failed, else 0; absl's own flag-parse errors (unknown flag, malformed value) and `--help` also exit 1. No tests.

The `*_results.db` SQLite databases rocprofv3 writes alongside the code objects and traces are not read. Code object load addresses are not needed: the trace decoder resolves each hardware PC to a `{code_object_id, ELF vaddr}` pair before the tool sees it, and every address downstream stays in ELF vaddr space.

## Namespace

All project code lives in the `my_rocperf_tool` namespace. LLVM `DenseMapInfo` specializations are in the `llvm` namespace as required.

## Testing

`tests/` holds one googletest executable, `my_rocperf_tool_tests`, linked against the core library. Each core-library module with tests names its `<module>_test.cpp` under Architecture; `trace_pc.h` has none. `gtest_discover_tests` registers each test case with CTest, so `ctest --test-dir build --output-on-failure` runs them all (or run the binary directly, with `--gtest_filter` to pick cases).

Tests use module interfaces only, and no captured traces. Their shared inputs:

- `synthetic_wave.h` builds decoded waves, a table classifier and loop body listings for the wave analysis and loop profile tests.
- `tests/code_objects/*.s` are assembled at build time by `llvm-mc` into relocatable gfx942 and gfx1200 objects, so `.text` starts at vaddr 0; `llvm-objcopy` gives `gfx942_two_texts` two `.text` sections, at 0x1000 and 0x2000.

The CLI's end-to-end output is not covered.
