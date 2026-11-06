#pragma once

#include "my_rocperf_tool/loop_profile.h"
#include "my_rocperf_tool/report_config.h"
#include "my_rocperf_tool/wave_analysis.h"

#include <cstdint>
#include <string>
#include <vector>

namespace my_rocperf_tool {

// One loop record as the report prints it.
struct ReportedLoop {
  LoopRecord record;
  // The profile's answer for the record. Its rows view the loop body's text,
  // which must outlive the report.
  LoopRecordMetrics metrics;
};

// The records printed under one heading: a wave's, a slot's or the aggregate
// across waves.
struct LoopGroup {
  AggregateScope scope = AggregateScope::Wave;
  WaveSlot slot;  // unused under All
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
  // (WaveAnalysisOptions::simd, -1 for every SIMD), for the text banner of a
  // trace that prints no group.
  WaveCounts counts;
  int32_t simd_filter = -1;
  // In print order.
  std::vector<LoopGroup> groups;
};

// Appends `report` to `out` as its trace's chunk of the document. Depends on
// nothing but its arguments, so traces render in parallel, each into its own
// chunk. In text, a trace that prints no group gets a "no loops detected"
// banner instead, so that it does not read like a failed run; CSV and JSON
// get no banner, so they stay parseable, and leave the chunk as it was.
void render_loop_report(ReportChunk& out, const ReportConfig& cfg,
                        const TraceLoopReport& report);

void write_loop_csv_header(ReportChunk& out, const ReportConfig& cfg);

}  // namespace my_rocperf_tool
