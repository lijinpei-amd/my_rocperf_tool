#pragma once

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"

#include "trace_decoder_types.h"

#include <cstdint>
#include <vector>

namespace my_rocperf_tool {

struct BackEdge {
  uint64_t code_object_id;
  uint64_t source_addr; // branch PC (end of loop body)
  uint64_t target_addr; // loop header PC (start of loop)

  bool operator==(const BackEdge &other) const {
    return code_object_id == other.code_object_id &&
           source_addr == other.source_addr &&
           target_addr == other.target_addr;
  }
};

struct LoopStats {
  uint32_t iteration_count = 0;
  uint64_t total_inst_count = 0;
  uint64_t total_duration = 0;
  uint64_t total_stall = 0;
  uint64_t total_idle = 0;
};

struct DetectedLoop {
  BackEdge back_edge;
  LoopStats stats;
};

struct WaveLoopInfo {
  uint8_t cu;
  uint8_t simd;
  uint8_t wave_id;
  std::vector<DetectedLoop> loops;
};

/// Compute idle time for an instruction given the previous instruction's
/// end time. Returns the idle value and updates last_time.
inline uint32_t compute_idle(const rocprofiler_thread_trace_decoder_inst_t &inst,
                             int64_t &last_time) {
  uint32_t idle = 0;
  if (inst.time >= last_time)
    idle = inst.time - last_time;
  last_time = inst.time + inst.duration;
  return idle;
}

/// Returns true if this instruction should be skipped (null PC).
inline bool is_null_pc(const rocprofiler_thread_trace_decoder_inst_t &inst) {
  return inst.pc.code_object_id == 0 && inst.pc.address == 0;
}

/// Detect loops in a wave's instruction trace via back-edge analysis.
WaveLoopInfo detect_loops(const rocprofiler_thread_trace_decoder_wave_t &wave);

} // namespace my_rocperf_tool

namespace llvm {
template <> struct DenseMapInfo<my_rocperf_tool::BackEdge> {
  using UInt64Info = DenseMapInfo<uint64_t>;

  static my_rocperf_tool::BackEdge getEmptyKey() {
    return {UInt64Info::getEmptyKey(), UInt64Info::getEmptyKey(),
            UInt64Info::getEmptyKey()};
  }

  static my_rocperf_tool::BackEdge getTombstoneKey() {
    return {UInt64Info::getTombstoneKey(), UInt64Info::getTombstoneKey(),
            UInt64Info::getTombstoneKey()};
  }

  static unsigned getHashValue(const my_rocperf_tool::BackEdge &val) {
    return hash_combine(UInt64Info::getHashValue(val.code_object_id),
                        UInt64Info::getHashValue(val.source_addr),
                        UInt64Info::getHashValue(val.target_addr));
  }

  static bool isEqual(const my_rocperf_tool::BackEdge &lhs,
                      const my_rocperf_tool::BackEdge &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm
