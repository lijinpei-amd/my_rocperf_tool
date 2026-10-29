#pragma once

#include "my_rocperf_tool/inst_kind.h"
#include "my_rocperf_tool/loop_detection.h"
#include "my_rocperf_tool/loop_profile.h"
#include "my_rocperf_tool/trace_pc.h"

#include "llvm/ADT/DenseMap.h"

#include "trace_decoder_types.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <variant>
#include <vector>

namespace my_rocperf_tool {

// The decoder defines a record's duration as its stall plus its issue (or
// execution) time, so a record whose stall exceeds its duration is
// inconsistent, and so is every number derived from it.
enum class DurationCheck {
  Off,
  // Each offending record becomes a diagnostic; the result stays usable.
  Warn,
  // The result carries a DurationCheckFailure, and its numbers are not to be
  // reported.
  Error,
};

struct WaveAnalysisOptions {
  // -1 analyzes every wave; otherwise only waves on this SIMD. Dropping the
  // waves at the source rather than filtering the reports keeps every
  // accumulator -- the loop profile, loops -- consistently scoped to the
  // selected SIMD.
  int32_t simd = -1;
  bool detect_loops = true;
  // Accumulate MFMA-shadow bubbles over each detected loop's body.
  bool mfma_coexec = false;
  // Keep each wave's per-instruction loop rows, not only the aggregate's.
  // Without, the profile answers a wave's loop record with its summary alone,
  // and keeps no per-wave slots.
  bool wave_loop_rows = true;
  DurationCheck duration_check = DurationCheck::Error;
};

// The kind of the instruction at an attributed PC. WaveAnalysis caches the
// answer, so it asks once per PC.
using InstClassifier =
    std::function<InstKind(const rocprofiler_thread_trace_decoder_pc_t&)>;

// Decode counters, reported in the run summary so a trace that yields no
// loops is distinguishable from one that failed to decode.
struct WaveCounts {
  // Waves analyzed, which excludes those the SIMD filter dropped.
  uint64_t waves = 0;
  // Their attributed records, an s_barrier's token pair counted once.
  uint64_t insts = 0;
  // Kept apart from waves so a trace whose waves all ran on other SIMDs is not
  // reported as an empty capture.
  uint64_t waves_filtered = 0;
  // Waves with at least one detected loop, and those loops summed over them.
  uint64_t loop_waves = 0;
  uint64_t loops = 0;
};

// A record whose stall exceeds its duration.
struct DurationBelowStall {
  rocprofiler_thread_trace_decoder_pc_t pc;
  int32_t duration;
  uint32_t stall;
};

using WaveDiagnostic = std::variant<DurationBelowStall, MfmaShadowOverlap>;

// What DurationCheck::Error found: how many records failed, and the first.
struct DurationCheckFailure {
  uint64_t records;
  DurationBelowStall first;
};

// What the waves of one trace add up to.
struct WaveAnalysisResult {
  WaveCounts counts;
  // Each wave's executions of its detected loops' bodies, as per-loop, per-PC
  // metrics.
  LoopProfile profile;
  // The detected loops of each wave that has any, in the order the waves were
  // added. Empty unless detect_loops.
  std::vector<WaveLoopInfo> wave_loops;
  // In the order they were found.
  std::vector<WaveDiagnostic> diagnostics;
  std::optional<DurationCheckFailure> duration_check_failure;
};

// Analyzes the decoded waves of one trace. The decoder owns a wave's memory
// only while it hands the wave over, so waves are streamed in and none is
// kept.
//
// Each wave goes through the same steps in a fixed order: the SIMD filter,
// then s_barrier token coalescing, then the duration check, then loop
// detection, then the loop profile, which takes the wave's records and loops.
// A filtered wave reaches no accumulator, and every accumulator sees the
// coalesced records.
class WaveAnalysis {
 public:
  WaveAnalysis(WaveAnalysisOptions options, InstClassifier classify);

  void add_wave(const rocprofiler_thread_trace_decoder_wave_t& wave);

  // The analysis of every wave added. Consumes this analysis.
  WaveAnalysisResult finish() &&;

 private:
  using pc_t = rocprofiler_thread_trace_decoder_pc_t;
  using wave_t = rocprofiler_thread_trace_decoder_wave_t;

  InstKind kind_at(const pc_t& pc);
  // Fills `coalesced` from the wave's records.
  void coalesce_barrier_tokens(const wave_t& wave);
  // Applies the duration-check policy to a record that failed the check.
  void add_duration_below_stall(
      const rocprofiler_thread_trace_decoder_inst_t& inst);

  WaveAnalysisOptions options;
  InstClassifier classify;
  llvm::DenseMap<pc_t, InstKind> kinds;
  // Reused across waves, so coalescing allocates only when a wave outgrows
  // every earlier one.
  std::vector<rocprofiler_thread_trace_decoder_inst_t> coalesced;
  WaveAnalysisResult result;
};

}  // namespace my_rocperf_tool
