#pragma once

#include "my_rocperf_tool/inst_profile.h"
#include "my_rocperf_tool/report_config.h"
#include "my_rocperf_tool/wave_analysis.h"

#include <cstdint>
#include <string>
#include <vector>

namespace my_rocperf_tool {

struct TraceInstReport {
  std::string prefix;
  uint64_t dispatch_id = 0;
  uint64_t se_id = 0;
  WaveCounts counts;
  int32_t simd_filter = -1;
  std::vector<InstGroup>
      groups;  // in print order; rows' text must outlive the report
};
// cfg.report is StaticInst or DynamicInst; DynamicInst adds the occurrence
// column/key.
void render_inst_report(ReportChunk& out, const ReportConfig& cfg,
                        const TraceInstReport& report);
void write_inst_csv_header(ReportChunk& out, const ReportConfig& cfg);

}  // namespace my_rocperf_tool
