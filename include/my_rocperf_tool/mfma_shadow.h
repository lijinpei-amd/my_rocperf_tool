#pragma once

#include "trace_decoder_types.h"

#include <cstdint>
#include <vector>

namespace my_rocperf_tool {

// Cycles an MFMA is assumed to keep the MAI pipe busy, during which a
// co-issued non-MFMA instruction executes for free.
inline constexpr int64_t kMfmaShadowCycles = 16;

// ATT hardware timestamps land on a 4-cycle grid, and the decoder splits the
// cycles between two timestamp packets across the instructions it saw in
// between. A single instruction's duration can therefore be off by one
// granule.
inline constexpr int64_t kAttTimestampGranule = 4;

// An MFMA that issued inside the previous MFMA's shadow by more than one
// timestamp granule: more than decoder slop, so kMfmaShadowCycles is wrong for
// this ISA.
struct MfmaShadowOverlap {
  rocprofiler_thread_trace_decoder_pc_t pc;
  int64_t issue_point;
  int64_t shadow_end;
};

// A record kept after a failed duration check issues in zero cycles.
inline uint64_t issue_cycles(
    const rocprofiler_thread_trace_decoder_inst_t& inst) {
  int64_t issue = static_cast<int64_t>(inst.duration) - inst.stall;
  return issue > 0 ? static_cast<uint64_t>(issue) : 0;
}

// One forward-only MFMA shadow over a walk's records.
class MfmaShadow {
 public:
  uint64_t step(const rocprofiler_thread_trace_decoder_inst_t& inst,
                bool is_mfma, std::vector<MfmaShadowOverlap>& overlaps);

 private:
  int64_t end = 0;
};

}  // namespace my_rocperf_tool
