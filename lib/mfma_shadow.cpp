#include "my_rocperf_tool/mfma_shadow.h"

#include <algorithm>

namespace my_rocperf_tool {

uint64_t MfmaShadow::step(const rocprofiler_thread_trace_decoder_inst_t& inst,
                          bool is_mfma,
                          std::vector<MfmaShadowOverlap>& overlaps) {
  if (is_mfma) {
    int64_t issue_point = inst.time + inst.stall;
    // Up to one timestamp granule of overlap is decoder slop.
    if (end > 0 && issue_point < end - kAttTimestampGranule)
      overlaps.push_back({inst.pc, issue_point, end});
    // Never rewind and uncover cycles the previous MFMA already covered.
    end = std::max(end, issue_point + kMfmaShadowCycles);
    return 0;
  }
  int64_t uncovered = inst.time + inst.duration - end;
  int64_t bubble = std::min(static_cast<int64_t>(inst.duration), uncovered);
  return bubble > 0 ? static_cast<uint64_t>(bubble) : 0;
}

}  // namespace my_rocperf_tool
