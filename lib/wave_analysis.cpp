#include "my_rocperf_tool/wave_analysis.h"

#include "my_rocperf_tool/inst_kind.h"
#include "my_rocperf_tool/loop_detection.h"

#include "trace_decoder_types.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace my_rocperf_tool {

WaveAnalysis::WaveAnalysis(WaveAnalysisOptions options, InstClassifier classify)
    : options(options), classify(std::move(classify)) {}

void WaveAnalysis::add_wave(const wave_t& decoded) {
  if (options.simd >= 0 && decoded.simd != options.simd) {
    result.counts.waves_filtered++;
    return;
  }
  coalesce_barrier_tokens(decoded);
  wave_t wave = decoded;
  wave.instructions_array = coalesced.data();
  wave.instructions_size = coalesced.size();
  result.counts.waves++;

  int64_t last_time = wave.begin_time;
  for (size_t j = 0; j < wave.instructions_size; j++) {
    const auto& inst = wave.instructions_array[j];
    if (is_null_pc(inst)) continue;
    result.counts.insts++;
    assert(inst.pc.code_object_id != 0);
    uint32_t idle = compute_idle(inst, last_time);
    if (options.duration_check != DurationCheck::Off &&
        inst.duration < static_cast<int32_t>(inst.stall))
      add_duration_below_stall(inst);
    result.stats.add_inst(inst, idle);
  }

  if (!options.detect_loops) return;
  auto loop_info = detect_loops(wave);
  if (loop_info.loops.empty()) return;
  for (const auto& dl : loop_info.loops) {
    compute_relative_issue(wave, dl.back_edge);
    if (options.mfma_coexec) analyze_mfma_coexec(wave, dl.back_edge);
  }
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

void WaveAnalysis::compute_relative_issue(const wave_t& wave,
                                          const BackEdge& edge) {
  int64_t iter_start = -1;
  for (size_t i = 0; i < wave.instructions_size; i++) {
    const auto& inst = wave.instructions_array[i];
    if (is_null_pc(inst)) continue;
    if (inst.pc.code_object_id != edge.code_object_id ||
        inst.pc.address < edge.target_addr ||
        inst.pc.address > edge.source_addr)
      continue;
    // New iteration starts when we hit the loop header.
    if (inst.pc.address == edge.target_addr)
      iter_start = inst.time + inst.stall;
    if (iter_start >= 0) {
      int64_t rel = (inst.time + inst.stall) - iter_start;
      if (rel >= 0)
        result.rel_issue_totals[edge][inst.pc] += static_cast<uint64_t>(rel);
    }
  }
}

void WaveAnalysis::analyze_mfma_coexec(const wave_t& wave,
                                       const BackEdge& edge) {
  // Instructions are dword-aligned, so a body PC's slot is its dword offset.
  auto& body = body_kinds[edge];
  if (body.empty())
    body.assign((edge.source_addr - edge.target_addr) / 4 + 1, -1);
  int64_t mfma_shadow_end = 0;

  for (size_t i = 0; i < wave.instructions_size; i++) {
    const auto& inst = wave.instructions_array[i];
    if (is_null_pc(inst)) continue;
    if (inst.pc.code_object_id != edge.code_object_id ||
        inst.pc.address < edge.target_addr ||
        inst.pc.address > edge.source_addr)
      continue;

    auto& kind = body[(inst.pc.address - edge.target_addr) / 4];
    if (kind < 0) kind = static_cast<int8_t>(kind_at(inst.pc));
    if (static_cast<InstKind>(kind) == InstKind::Mfma) {
      // MFMA is productive work — update shadow, don't count as bubble.
      int64_t issue_point = inst.time + inst.stall;
      // Verify no two MFMAs issue within each other's shadow. An overlap of
      // up to one timestamp granule is decoder slop, not a pipe violation --
      // only a larger one means kMfmaShadowCycles is wrong for this ISA.
      if (mfma_shadow_end > 0 &&
          issue_point < mfma_shadow_end - kAttTimestampGranule)
        result.diagnostics.push_back(
            MfmaShadowOverlap{inst.pc, issue_point, mfma_shadow_end});
      // Never rewind: a slop-induced overlap must not un-cover cycles the
      // previous MFMA already accounted for, which would attribute a bubble
      // to whatever follows.
      mfma_shadow_end =
          std::max(mfma_shadow_end, issue_point + kMfmaShadowCycles);
    } else {
      // Non-MFMA: compute bubble (portion not covered by co-exec shadow).
      int64_t end_time = inst.time + inst.duration;
      if (end_time > mfma_shadow_end) {
        int64_t uncovered = end_time - mfma_shadow_end;
        int64_t bubble =
            std::min(static_cast<int64_t>(inst.duration), uncovered);
        if (bubble > 0)
          result.bubble_totals[edge][inst.pc] += static_cast<uint64_t>(bubble);
      }
    }
  }
}

}  // namespace my_rocperf_tool
