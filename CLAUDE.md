# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

my_rocperf_tool is a C++17 performance analysis tool for AMD GPUs. It decodes Advanced Thread Trace (ATT) data from rocprofiler and correlates per-instruction execution statistics (stall cycles, duration, idle time) with LLVM MC-based disassembly of AMDGPU code objects.

## Build Commands

```bash
git submodule update --init --recursive
cmake -B build -G Ninja -DROCPROF_TRACE_DECODER_PATH=/path/to/decoder
cmake --build build
```

CMake minimum 3.28, Ninja generator, C++17, compiled with `-fPIC -fno-rtti`.

### Dependencies

- **rocprofiler-sdk** — AMD ROCm profiler SDK
- **LLVM** (CONFIG mode) — MC disassembler infrastructure
- **SQLite3** — reading profiling result databases
- **rocprof-trace-decoder** — ATT trace decoding (path configured via `ROCPROF_TRACE_DECODER_PATH`)
- **abseil-cpp** — git submodule at `3rd_party/abseil-cpp` (logging, flags)

### Build Outputs

- `build/lib/libmy_rocperf_tool.so` — shared library
- `build/tools/load_issue_stall` — CLI tool

## Code Style

Pre-commit hooks enforce clang-format and other checks. Set up with `pre-commit install`, run manually with `pre-commit run --all-files`.

## Architecture

**Library (`lib/` → `libmy_rocperf_tool.so`):**

- **Disassembler / ObjectFileInfo** (`disassembler.h/cpp`) — LLVM MC-based AMDGPU disassembler. Parses ELF code objects, identifies ISA from ELF machine flags (matched against `include/my_rocperf_tool/data/comgr-isa-metadata.def`), caches subtarget info, and provides on-demand instruction decoding via `decode_at(addr)`.
- **RocPerfTool** (`rocperf_tool.h/cpp`) — Registers rocprofiler-sdk callbacks for code object load tracing; feeds loaded code objects to the Disassembler.
- **InstStatistics** (`inst_statistics.h`) — Header-only; accumulates per-PC execution statistics (stall, duration, idle, category) from decoded thread trace data.
- **init_llvm** (`init_llvm.h/cpp`) — Initializes all LLVM targets, MC layers, disassemblers, and asm parsers.

**CLI Tool (`tools/load_issue_stall.cpp`):**

Reads an ATT output directory containing code object files (`*_code_object_id_*.out`), ATT trace files (`*.att`), and SQLite result databases (`*_results.db`). Decodes traces via `rocprof_trace_decoder_parse_data` with SE-data, ISA, and trace callbacks, then dumps annotated disassembly with per-instruction timing stats. Uses `absl::flags` for CLI argument parsing.

## Namespace

All project code lives in the `my_rocperf_tool` namespace. LLVM `DenseMapInfo` specializations are in the `llvm` namespace as required.

## Testing

No test suite currently exists.
