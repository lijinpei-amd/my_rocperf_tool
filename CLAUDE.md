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

- **Disassembler / ObjectFileInfo** (`disassembler.h/cpp`) — LLVM MC-based AMDGPU disassembler. `Disassembler::add_code_object` loads an ELF code object under its id, from a file or a live load record; `find_code_object(id)` returns its `ObjectFileInfo`, or null for an id never loaded. An `ObjectFileInfo` answers questions about the instruction at an ELF vaddr: `size_at`, `kind_at` (`InstKind` from `inst_kind.h`: an `InstClass` of MFMA, classic s_barrier or other, and an MFMA's `mfma_cycles` from the ISA's LLVM scheduling model, resolved per instruction), `text_at` / `copy_text_at` (LLVM's printed text, which the trace decoder classifies, so its bytes are part of the contract), and `inst_vaddrs` lists a range's instructions. A vaddr that is not an instruction's start in a `.text` section asserts. A load fails with an `llvm::Error` "<path>: <reason>" (unreadable, not AMDGPU, or an ISA missing from `data/comgr-isa-metadata.def`) and leaves the id unknown. Each `.text` section is decoded whole on first query, with one MC stack and opcode→class table per ISA. Needs no setup: the first construction registers the AMDGPU target, once per process. Not thread-safe (one per thread), and neither copyable nor movable, because MC objects keep the address of its MC options. Tests: `disassembler_test.cpp`.
- **AttOutputDir** (`att_output_dir.h/cpp`) — `scan` walks an ATT output directory tree for code objects (`<prefix>_<arch>_code_object_id_<id>.out`) and traces (`<prefix>_<agent>_shader_engine_<se>_<dispatch>.att`), parsed into sorted `CodeObjectPath` / `AttPath` lists. The prefix, qualified by the file's subdirectory, names the run; code object and dispatch ids are unique only within a run, and traces sort by run first. `scan` fails if any directory cannot be listed, since a partial scan would drop whole runs; a file whose target cannot be stat'ed is kept, so reading it reports why. Both `scan` and `read_att_data` fail with "cannot read <path>: <reason>". Tests: `att_output_dir_test.cpp`.
- **detect_loops** (`loop_detection.h/cpp`) — Finds a wave's loops by back-edge and sums `LoopStats` over each loop's iterations, keeping those with 2 or more. Expects coalesced s_barrier records. `BackEdge::contains(pc)` is the one body-range test and `LoopStats::operator+=` the one merge. Tests: `loop_detection_test.cpp`.
- **WaveAnalysis** (`wave_analysis.h/cpp`) — Analyzes one trace's waves, streamed by `add_wave`; the decoder owns each wave only during its callback. `WaveAnalysisOptions` selects the SIMD, loop detection, MFMA co-exec, per-wave loop rows, duration check and optional `InstProfileKind` / `AggregateScope`. The SIMD filter precedes barrier-token coalescing and the duration check; the selected profiles see those coalesced records. The `InstClassifier` is cached per PC. `std::move(analysis).finish()` returns `WaveCounts`, `LoopProfile`, optional `InstProfile`, each wave's detected loops (numbered by decode-order `wave_index`), diagnostics and any `DurationCheckFailure`. It prints nothing. Tests: `wave_analysis_test.cpp`.
- **LoopProfile** (`loop_profile.h/cpp`) — Accumulates per-loop instruction sums over waves where the loop was detected. `metrics(LoopRecord, LoopBody)` resolves a per-wave record (`wave` names its `wave_index`), per-slot record (`slot` names its `WaveSlot`), or trace aggregate (neither set). A record sets at most one of `wave` and `slot`. Per-slot and aggregate rows are always kept; per-wave rows are kept only under `--report=loop_detail --aggregate=wave`. Every row and per-iteration summary counts its record's own executions; a wave without a detected loop contributes to no record of that loop. `LoopBody` is plain instruction data supplied by the CLI. Each loop walk uses its own `MfmaShadow`. Tests: `loop_profile_test.cpp`.
- **LoopReport** (`loop_report.h/cpp`) — Renders `TraceLoopReport` under `ReportConfig` as text, CSV or JSON. Its `LoopGroup`s use wave, SIMD-slot or trace-wide scope and hold resolved `ReportedLoop`s; their rows view body text that must outlive rendering. `LoopDetail` prints body rows; `LoopSummary` prints only summaries. `render_loop_report` appends to a `ReportChunk` without external state, so traces render on workers. Text gives a trace without groups a loop-free banner; the empty `all` group keeps its aggregate heading. Tests: `loop_report_test.cpp`.
- **MfmaShadow** (`mfma_shadow.h/cpp`) — Shared issue arithmetic (`issue_cycles`, clamped at zero), forward-only MFMA shadow walk and overlap diagnostics. Each MFMA's shadow spans its `InstKind::mfma_cycles`. Defines `kAttTimestampGranule` and `MfmaShadowOverlap`. Used by both profiles; covered by their tests and `WaveAnalysisMfmaTest`.
- **aggregate_scope.h** — `AggregateScope::{Wave, SimdSlot, All}` and `WaveSlot` ordered by `(cu, simd, wave_id)`, with its LLVM `DenseMapInfo`. Covered by loop and instruction profile tests.
- **InstProfile** (`inst_profile.h/cpp`) — Whole-wave instruction sums, with no decoded records retained. Static rows key by PC and sort by `(code_object_id, address)`; dynamic rows key by `(PC, occurrence)` and sort by average relative issue, then PC and occurrence. A wave's first attributed issue anchors its `rel`; its idle starts at `begin_time`, and one MFMA shadow spans the whole wave. `std::move(profile).groups()` consumes the profile and returns plain `InstGroup`s in slot order, with same-slot waves in decode order, releasing each group's row map as it goes. `InstSums::n` counts executions and `exec_waves` counts waves executing the key. Summaries average over every analyzed wave in the group, including waves with no attributed records. Without waves there is no group even under `all`. Tests: `inst_profile_test.cpp`.
- **InstReport** (`inst_report.h/cpp`) — Renders a resolved `TraceInstReport` with SE labels, per-wave group summaries and instruction rows. Static rows identify PCs; dynamic rows also show occurrence. Stats are sums by default and per-execution averages under `show_avg`. Text and JSON respect `show_stats`; CSV keeps stats, with one column list for header and rows. A group with no rows still has a summary; a trace with no groups gets a text banner. Row text must outlive rendering. Tests: `inst_report_test.cpp`.
- **ReportConfig / ReportDocument** (`report_config.h`, `report_document.h/cpp`) — Shared kind, scope, format and row-display options, `ReportChunk` and `write_report_document`. The document writer emits one CSV header or one JSON array, joins non-empty chunks in trace order and writes zero bytes for `None` in every format. Covered by `loop_report_test.cpp` and `inst_report_test.cpp`.
- **report_writer.h** (private, `lib/report_writer.h`) — Formatting helpers shared by the loop and instruction renderers: `writef`, CSV and JSON escaping, dispatch labels and bubble colors. Covered by `loop_report_test.cpp` and `inst_report_test.cpp`.
- **trace_pc.h** — `DenseMapInfo` for the decoder's `{vaddr, code object}` PC. Covered by `inst_profile_test.cpp` and `wave_analysis_test.cpp`.

**Live library (`lib/` → `libmy_rocperf_tool_live.so`):** `RocPerfTool` (`rocperf_tool.h/cpp`) registers rocprofiler-sdk code object load tracing and feeds each loaded code object to a Disassembler. The only piece linking the rocprofiler-sdk runtime. Its `rocprofiler_configure` entry point is commented out, so nothing instantiates it yet. No tests.

**CLI Tool (`tools/load_issue_stall.cpp`):**

Selects one report for each trace under `--att_output_dir`:

| Option | Values / behavior |
|---|---|
| `--report` | `loop_summary` (default), `loop_detail`, `static_inst`, `dynamic_inst`, `none` |
| `--aggregate` | `wave` (default), `simd_slot`, `all`; every group stays within one trace / SE |
| `--output_format` | `text` (default), `csv`, `json` |
| `--inst_stats`, `--inst_addr`, `--inst_avg` | Shape instruction rows; averages are per execution |
| `--mfma_coexec`, `--color_mfma_bubble` | Compute MFMA-shadow bubbles and color text rows |

Only loop reports detect loops. `none` still decodes and prints the stderr summary, with zero stdout bytes. Static memory grows with executed PCs per group; dynamic memory grows with distinct `(PC, occurrence)` keys, up to the longest wave's records per group. Per-wave instruction reports retain a row set per wave until rendering. Lossy traces include waves caught mid-run, which skews their relative issue and occurrences.

`main` validates the `absl::flags` and maps them to `WaveAnalysisOptions` and `ReportConfig` using `report_config_from_flags`. `run_main` scans with `AttOutputDir` and requires code objects for every run. `run_trace` drives `rocprof_trace_decoder_parse_data`: callbacks supply trace data, answer ISA queries from the run's Disassembler and stream waves into `WaveAnalysis`. Loop reports resolve `TraceLoopReport` with `build_loop_groups` and `LoopBodies`, listing each body once. Instruction reports resolve `TraceInstReport` with `InstTexts`, listing each PC's text once in a node-based map so row views stay valid. Rendering finishes before either text cache is destroyed. An unreadable trace, failed code object load, unknown code object or duration-check rejection contributes no report.

Traces decode serially until one reaches the decoder (its lazy trie fill races otherwise), then on `--jobs` workers, each building a Disassembler per run on its first trace of that run. Each trace's stderr text is buffered and printed in trace order, ahead of the stdout document, so neither depends on `--jobs`. A `[summary]` on stderr follows: totals, failed traces by cause, traces that decoded no waves, `--simd` skips and, only for loop reports, loop-free dispatches. Exit status is 2 when the tool's own checks reject a flag (`--att_output_dir` missing, `--simd` out of range, a bad `--report`, `--aggregate` or `--verify_trace_duration`) or the directory (unreadable, no traces, a run without code objects), 1 if any trace failed, else 0; absl's own flag-parse errors (unknown flag, malformed value) and `--help` also exit 1. No tests.

The `*_results.db` SQLite databases rocprofv3 writes alongside the code objects and traces are not read. Code object load addresses are not needed: the trace decoder resolves each hardware PC to a `{code_object_id, ELF vaddr}` pair before the tool sees it, and every address downstream stays in ELF vaddr space.

## Namespace

All project code lives in the `my_rocperf_tool` namespace. LLVM `DenseMapInfo` specializations are in the `llvm` namespace as required.

## Testing

`tests/` holds one googletest executable, `my_rocperf_tool_tests`, linked against the core library. Architecture names each module's tests. `trace_pc.h`, `aggregate_scope.h`, `mfma_shadow`, `report_document` and private `report_writer.h` have no dedicated test file: instruction-profile and wave-analysis tests cover `trace_pc.h`; loop and instruction profile tests cover `aggregate_scope.h`; those profile tests and `WaveAnalysisMfmaTest` cover `mfma_shadow`; loop and instruction report tests cover `report_document` and `report_writer.h`. `gtest_discover_tests` registers each test case with CTest, so `ctest --test-dir build --output-on-failure` runs them all (or run the binary directly, with `--gtest_filter` to pick cases).

Tests use module interfaces only, and no captured traces. Their shared inputs:

- `synthetic_wave.h` builds decoded waves, a table classifier and loop body listings for the wave analysis and profile tests.
- `report_test_util.h` shares CSV parsing helpers between the report tests.
- `tests/code_objects/*.s` are assembled at build time by `llvm-mc` into relocatable gfx942, gfx950 and gfx1200 objects, so `.text` starts at vaddr 0; `llvm-objcopy` gives `gfx942_two_texts` two `.text` sections, at 0x1000 and 0x2000.

The CLI's end-to-end output is not covered.
