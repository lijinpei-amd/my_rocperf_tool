#include "my_rocperf_tool/wave_analysis.h"

#include "my_rocperf_tool/inst_kind.h"
#include "my_rocperf_tool/loop_detection.h"
#include "my_rocperf_tool/loop_profile.h"

#include "trace_decoder_types.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace my_rocperf_tool {

WaveAnalysis::WaveAnalysis(WaveAnalysisOptions options, InstClassifier classify)
    : options(options), classify(std::move(classify)) {
  result.profile = LoopProfile(options.mfma_coexec);
}

void WaveAnalysis::add_wave(const wave_t& decoded) {
  if (options.simd >= 0 && decoded.simd != options.simd) {
    result.counts.waves_filtered++;
    return;
  }
  coalesce_barrier_tokens(decoded);
  wave_t wave = decoded;
  wave.instructions_array = coalesced.data();
  wave.instructions_size = coalesced.size();
  const uint64_t wave_index = result.counts.waves++;

  for (size_t j = 0; j < wave.instructions_size; j++) {
    const auto& inst = wave.instructions_array[j];
    if (is_null_pc(inst)) continue;
    result.counts.insts++;
    assert(inst.pc.code_object_id != 0);
    if (options.duration_check != DurationCheck::Off &&
        inst.duration < static_cast<int32_t>(inst.stall))
      add_duration_below_stall(inst);
  }

  WaveLoopInfo loop_info{};
  if (options.detect_loops) loop_info = detect_loops(wave);
  loop_info.wave_index = wave_index;
  for (const auto& overlap : result.profile.add_wave(
           wave, loop_info, [this](const pc_t& pc) { return kind_at(pc); }))
    result.diagnostics.push_back(overlap);
  if (loop_info.loops.empty()) return;
  result.counts.loop_waves++;
  result.counts.loops += loop_info.loops.size();
  result.wave_loops.push_back(std::move(loop_info));
}

WaveAnalysisResult WaveAnalysis::finish() && { return std::move(result); }

InstKind WaveAnalysis::kind_at(const pc_t& pc) {
  auto [it, inserted] = kinds.try_emplace(pc);
  if (inserted) it->second = classify(pc);
  return it->second;
}

// gfx9-style `s_barrier` is the only known instruction whose execution lands
// in the trace as multiple tokens at the same PC: a barrier_signal (MSG)
// entry token followed immediately by a barrier_wait (IMMED) exit token, with
// no scheduling gap between them. Coalesce that pair into one record so
// PC-keyed accumulation downstream counts a single execution. Any other
// consecutive same-PC pattern is unknown territory and asserts so the wave
// decoder's behavior is surfaced rather than silently double-counted.
void WaveAnalysis::coalesce_barrier_tokens(const wave_t& wave) {
  coalesced.clear();
  coalesced.reserve(wave.instructions_size);
  constexpr uint32_t kStallMax = (1u << 24) - 1;
  auto pc_eq = [](const auto& a, const auto& b) {
    return a.pc.address == b.pc.address &&
           a.pc.code_object_id == b.pc.code_object_id;
  };
  for (size_t i = 0; i < wave.instructions_size;) {
    const auto& cur = wave.instructions_array[i];
    if (i + 1 >= wave.instructions_size || is_null_pc(cur) ||
        !pc_eq(cur, wave.instructions_array[i + 1])) {
      coalesced.push_back(cur);
      ++i;
      continue;
    }
    const auto& nxt = wave.instructions_array[i + 1];
    assert(kind_at(cur.pc) == InstKind::ClassicBarrier &&
           "consecutive same-PC tokens at non-s_barrier PC");
    assert(cur.category == ROCPROFILER_THREAD_TRACE_DECODER_INST_MESSAGE &&
           "first s_barrier token must be MESSAGE");
    assert(nxt.category == ROCPROFILER_THREAD_TRACE_DECODER_INST_IMMED &&
           "second s_barrier token must be IMMED");
    assert(nxt.time == cur.time + cur.duration &&
           "no gap allowed between s_barrier MSG and IMMED tokens");
    assert((i + 2 >= wave.instructions_size ||
            !pc_eq(cur, wave.instructions_array[i + 2])) &&
           "expected exactly 2 consecutive same-PC tokens at s_barrier");
    auto merged = cur;
    uint64_t total_dur = static_cast<uint64_t>(cur.duration) +
                         static_cast<uint64_t>(nxt.duration);
    uint64_t total_stall =
        static_cast<uint64_t>(cur.stall) + static_cast<uint64_t>(nxt.stall);
    merged.duration =
        total_dur > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())
            ? std::numeric_limits<int32_t>::max()
            : static_cast<int32_t>(total_dur);
    merged.stall = total_stall > kStallMax ? kStallMax
                                           : static_cast<uint32_t>(total_stall);
    coalesced.push_back(merged);
    i += 2;
  }
}

void WaveAnalysis::add_duration_below_stall(
    const rocprofiler_thread_trace_decoder_inst_t& inst) {
  DurationBelowStall record{inst.pc, inst.duration, inst.stall};
  if (options.duration_check == DurationCheck::Warn)
    result.diagnostics.push_back(record);
  else if (result.duration_check_failure)
    result.duration_check_failure->records++;
  else
    result.duration_check_failure = DurationCheckFailure{1, record};
}

}  // namespace my_rocperf_tool
