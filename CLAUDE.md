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
- **LLVM** (CONFIG mode) — MC disassembler infrastructure; its `llvm-mc` (from `LLVM_TOOLS_BINARY_DIR`) assembles the test code objects
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

- **Disassembler / ObjectFileInfo** (`disassembler.h/cpp`) — LLVM MC-based AMDGPU disassembler. `Disassembler::add_code_object` loads an ELF code object under its id and returns an `llvm::Error` ("<path>: <reason>") if the file cannot be read, is not AMDGPU, or has an ISA (ELF machine flags) missing from `include/my_rocperf_tool/data/comgr-isa-metadata.def`; `find_code_object(id)` returns null for an id never loaded. An `ObjectFileInfo` answers questions about the instruction at an ELF vaddr — `size_at`, `kind_at` (`InstKind` from `inst_kind.h`: MFMA, classic s_barrier, other), `text_at` / `copy_text_at` (LLVM's printed text, which the trace decoder classifies, so it must stay byte-identical) — and lists a vaddr range's instructions with `inst_vaddrs`. Section layout, MC objects and the print scratch buffer are private. One MC stack and one lazily filled opcode→kind table per ISA; each `.text` section is decoded whole on first query. Needs no setup: the first construction registers the AMDGPU target (once per process). Not thread-safe (one per thread), and neither copyable nor movable because each MCContext points at the Disassembler's MC options.
- **WaveAnalysis** (`wave_analysis.h/cpp`) — Per-trace analysis of decoded waves, streamed in with `add_wave` one at a time (the decoder owns wave memory only during its callback; nothing is buffered). Inputs are plain `WaveAnalysisOptions` (SIMD filter, loop detection, MFMA co-exec, per-wave loop rows, `DurationCheck` off/warn/error; no absl) and an `InstClassifier` callable (PC → `InstKind`), called once per PC and cached; the CLI wraps `ObjectFileInfo::kind_at`, tests pass a table. Owns the per-wave order: null-PC skipping, SIMD filter, s_barrier MSG+IMMED coalescing (the classic-barrier query feeds only an assert), the duration check, `detect_loops` (whose header states the coalesced-input precondition), then the wave's records and loops into its `LoopProfile`, numbered by `WaveLoopInfo::wave_index` (waves reuse cu/simd/wave_id slots). `std::move(analysis).finish()` returns a `WaveAnalysisResult`: `WaveCounts`, the `LoopProfile`, per-wave loops, and diagnostics (duration < stall, MFMA shadow overlaps) as data, plus a `DurationCheckFailure` under `Error`. It prints nothing.
- **LoopProfile** (`loop_profile.h/cpp`) — The per-loop, per-PC metrics of one trace, as running sums: memory grows with loop bodies (and, with per-wave rows, the waves' loops), not records. `add_wave(wave, loops, kind_at)` takes one coalesced wave and its detected loops and walks each loop's body once, over dense per-body-dword slots, summing count, duration, stall, idle, issue (max(duration − stall, 0) per record, in int32), relative issue and, under `mfma_coexec`, MFMA-shadow bubbles; it returns shadow overlaps for the diagnostics. A wave without loops adds nothing. `metrics(LoopRecord, LoopBody)` answers for one loop record — a wave's (`LoopRecord::wave` = its `wave_index`) or the aggregate (`nullopt`), carrying its `LoopStats` — with a `LoopInstRow` per body instruction (vaddr, text and kind from the body, plus metrics) and a `LoopSummary` (per-iteration averages, `mfma_count` from the body's kinds, `mfma_time`, `mfma_util`). Every number counts the record's own executions: a wave's record, each execution in that wave of a PC in the body (the population `detect_loops` sums into its `LoopStats`); the aggregate, those of the waves where the loop was detected. A body PC the record never ran has an n=0 row, and avg_bubble is the record's bubble over its iteration_count. Constructed without `wave_rows` (the CLI's `--dump_loop_contents` off or `--aggregate_wave_only`), it keeps only a bubble total per wave and loop, and a wave's record gets its summary with no rows. The `LoopBody` is plain data, so the module needs no code object; the CLI lists it from `ObjectFileInfo::inst_vaddrs`. Also defines `kMfmaShadowCycles`, `kAttTimestampGranule` and `MfmaShadowOverlap`. `loop_detection.h` gives `BackEdge::contains(pc)`, the one body-range test, and `LoopStats::operator+=`, the one merge (a pass into its loop in `detect_loops`, a wave's loop into the CLI's aggregate).
- **RocPerfTool** (`rocperf_tool.h/cpp`) — Registers rocprofiler-sdk callbacks for code object load tracing; feeds loaded code objects to the Disassembler.
- **trace_pc.h** — Header-only `DenseMapInfo` for the decoder's `{vaddr, code object}` PC, the key of `WaveAnalysis`'s per-PC kind cache.
- **InitLLVM** (`init_llvm.h`) — Header-only, optional process setup for executables: `llvm::InitLLVM`'s crash handlers and stack traces, and `llvm_shutdown` on destruction. Registers no targets and parses no LLVM options.

**CLI Tool (`tools/load_issue_stall.cpp`):**

Reads an ATT output directory containing code object files (`*_code_object_id_*.out`) and ATT trace files (`*.att`). Decodes traces via `rocprof_trace_decoder_parse_data`: the SE-data callback hands over the trace, the ISA callback answers from the run's Disassembler, and the trace callback forwards each wave to a `WaveAnalysis`. It then renders each loop record (one per wave, plus the aggregate under `--aggregate_wave`) from `LoopProfile::metrics`, listing each loop's body once per trace (`LoopBodies`) from `ObjectFileInfo`. Uses `absl::flags` for CLI argument parsing and maps them to `WaveAnalysisOptions` once, in `main`. Traces decode on parallel workers after a serial warm-up (the decoder's trie fill races otherwise); each trace's stderr text (unreadable file, its run's code object load errors, analysis diagnostics, rejection) is buffered and printed by `run_main` in trace order, ahead of stdout, so stderr does not depend on `--jobs`.

The `*_results.db` SQLite databases rocprofv3 writes alongside these files are not read. Code object load addresses are not needed: the trace decoder resolves each hardware PC to a `{code_object_id, ELF vaddr}` pair before the tool sees it, and every address downstream stays in ELF vaddr space.

## Namespace

All project code lives in the `my_rocperf_tool` namespace. LLVM `DenseMapInfo` specializations are in the `llvm` namespace as required.

## Testing

`tests/` holds one googletest executable, `my_rocperf_tool_tests`, linked against the core library. `gtest_discover_tests` registers each test case with CTest, so `ctest --test-dir build --output-on-failure` runs them all (or run the binary directly, with `--gtest_filter` to pick cases).

Tests use module interfaces only:

- `wave_analysis_test.cpp` — `WaveAnalysis` on synthetic waves with a table classifier, reading per-PC numbers from the `LoopProfile` rows of each loop's aggregate record: s_barrier MSG+IMMED coalescing and its saturation, the SIMD filter leaving no trace, MFMA shadow overlap tolerance and reporting, the shadow end never rewinding, the bubble rule, relative issue resetting at the loop header, per-loop totals, one classification per PC, `wave_loop_rows` off leaving rows to the aggregate, and the off/warn/error duration-check outcomes (off and warn keep the offending records, issuing in no time).
- `loop_profile_test.cpp` — `LoopProfile` fed synthetic waves directly, with `detect_loops`' loops: row sums and the row of a body instruction never run, issue clamped at 0 where stall exceeds duration, no bubbles or classification without `mfma_coexec`, the summary's averages, avg_bubble and MFMA count from the body's kinds, each record's rows and averages counting only its own executions (per wave, and over the waves where the loop was detected, not a wave that ran the body once), a wave's record without `wave_rows`, and an outer loop's avg_bubble per outer iteration.
- `synthetic_wave.h` — the wave builder, table classifier and loop body listing those two share.
- `loop_detection_test.cpp` — `detect_loops` on synthetic waves built from plain `rocprofiler_thread_trace_decoder_*` structs: iteration counts, back-edges, per-loop totals, null-PC skipping.
- `disassembler_test.cpp` — `Disassembler` and `ObjectFileInfo` on gfx942 and gfx1200 code objects that the build assembles from `tests/code_objects/*.s` (relocatable, so `.text` starts at vaddr 0, except one that `llvm-objcopy` gives two `.text` sections at 0x1000 and 0x2000): size, kind and text per vaddr in either section, gfx12's split barrier not taken for the classic one, range walks across multi-dword instructions and none outside or across sections, load failures (unknown ISA, unreadable or non-ELF file), unknown-id lookups, no `InitLLVM`, concurrent first construction, and compile-time checks that it cannot be copied or moved.
- `att_output_dir_test.cpp` — `AttOutputDir` on temp directories of empty, rocprofv3-named files: prefix parsing, per-subdirectory runs, trace sort order, traces and code objects kept when their target cannot be stat'ed, and the "cannot read <path>: <reason>" failures.

The CLI's end-to-end output is not covered.
