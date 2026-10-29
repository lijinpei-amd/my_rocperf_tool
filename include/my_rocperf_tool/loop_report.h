#pragma once

#include "my_rocperf_tool/loop_profile.h"
#include "my_rocperf_tool/wave_analysis.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include "fmt/format.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace my_rocperf_tool {

enum class LoopOutputFormat { Text, Csv, Json };

// The loop report's options. The formats differ on purpose, in field names,
// in which options they honour and in how they show a missing value (JSON
// omits the key, CSV leaves the field blank, text pads the column): scripts
// parse the CSV and JSON as they are.
struct LoopOutputConfig {
  LoopOutputFormat format = LoopOutputFormat::Text;
  bool show_contents = false;
  bool show_stats = true;
  bool show_addr = true;
  bool show_avg = false;
  bool show_mfma_coexec = false;
  bool color_mfma_bubble = false;
  // The report's groups are built to match: a group per wave unless
  // aggregate_wave_only, which implies aggregate_wave, then the aggregate
  // under aggregate_wave. Rendering reads only aggregate_wave, which adds
  // CSV's record_type and wave_count columns.
  bool aggregate_wave = false;
  bool aggregate_wave_only = false;
  // Set when the directory holds more than one run (output-file prefix):
  // dispatch ids repeat across runs, so every record also names its prefix.
  // Off for a single run, which keeps that output unchanged.
  bool show_prefix = false;
};

// One loop record as the report prints it.
struct ReportedLoop {
  LoopRecord record;
  // The profile's answer for the record. Its rows view the loop body's text,
  // which must outlive the report.
  LoopRecordMetrics metrics;
};

// The records printed under one heading: a wave's, or the aggregate across
// waves.
struct LoopGroup {
  bool is_aggregate = false;
  uint8_t cu = 0;        // valid only when !is_aggregate
  uint8_t simd = 0;      // valid only when !is_aggregate
  uint32_t wave_id = 0;  // valid only when !is_aggregate
  std::vector<ReportedLoop> loops;
};

// What the report prints for one trace, one shader engine of one dispatch,
// resolved up front: plain data, so rendering needs no code object, profile
// or cache, and a test can build one by hand.
struct TraceLoopReport {
  // The trace's run, printed under show_prefix.
  std::string prefix;
  uint64_t dispatch_id = 0;
  uint64_t se_id = 0;
  // What the decode found and the SIMD filter it ran under
  // (WaveAnalysisOptions::simd), for the text banner of a trace that prints
  // no group.
  WaveCounts counts;
  int32_t simd = -1;
  // In print order.
  std::vector<LoopGroup> groups;
};

// One trace's part of the document. A lower small-buffer size than
// fmt::memory_buffer's default (500): aggregate_wave_only output fits in
// <256 B, and the caller holds a chunk per trace until the document is
// written.
using LoopReportChunk = fmt::basic_memory_buffer<char, 256>;

// Appends `report` to `out` as its trace's chunk of the document. Depends on
// nothing but its arguments, so traces render in parallel, each into its own
// chunk. In text, a trace that prints no group gets a "no loops detected"
// banner instead, so that it does not read like a failed run; CSV and JSON
// get no banner, so they stay parseable, and leave the chunk as it was.
void render_loop_report(LoopReportChunk& out, const LoopOutputConfig& cfg,
                        const TraceLoopReport& report);

// Writes the document that `chunks`, in trace order, make: the format's
// prologue (CSV's header, JSON's opening bracket), every non-empty chunk,
// joined as the format requires, and the epilogue. `write` receives the
// document's bytes in order, a piece at a time.
void write_loop_report_document(
    const LoopOutputConfig& cfg, llvm::ArrayRef<LoopReportChunk> chunks,
    llvm::function_ref<void(std::string_view)> write);

}  // namespace my_rocperf_tool
