#include "my_rocperf_tool/loop_profile.h"

#include "my_rocperf_tool/inst_kind.h"
#include "my_rocperf_tool/loop_detection.h"

#include "llvm/ADT/ArrayRef.h"

#include "trace_decoder_types.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace my_rocperf_tool {

namespace {

// Instructions are dword-aligned, so a body PC's slot is its dword offset.
size_t body_slot(const BackEdge& edge, uint64_t addr) {
  return (addr - edge.target_addr) / 4;
}

}  // namespace

LoopProfile::LoopProfile(bool mfma_coexec) : mfma_coexec(mfma_coexec) {}

std::vector<MfmaShadowOverlap> LoopProfile::add_wave(const wave_t& wave,
                                                     const WaveLoopInfo& loops,
                                                     KindLookup kind_at) {
  int64_t last_time = wave.begin_time;
  for (size_t i = 0; i < wave.instructions_size; i++) {
    const auto& inst = wave.instructions_array[i];
    if (is_null_pc(inst)) continue;
    uint32_t idle = compute_idle(inst, last_time);
    auto& totals = pc_totals[inst.pc];
    totals.count++;
    totals.duration += inst.duration;
    totals.stall += inst.stall;
    totals.idle += idle;
    // A record whose stall exceeds its duration (kept unless the duration
    // check rejects the trace) issued in no time rather than negative time.
    int32_t issue = inst.duration - static_cast<int32_t>(inst.stall);
    totals.issue += issue > 0 ? static_cast<uint64_t>(issue) : 0;
  }

  std::vector<MfmaShadowOverlap> overlaps;
  for (const auto& loop : loops.loops)
    add_passes(wave, loop.back_edge, kind_at, overlaps);
  return overlaps;
}

void LoopProfile::add_passes(const wave_t& wave, const BackEdge& edge,
                             KindLookup kind_at,
                             std::vector<MfmaShadowOverlap>& overlaps) {
  auto& body = body_slots[edge];
  if (body.empty()) body.resize(body_slot(edge, edge.source_addr) + 1);
  int64_t iter_start = -1;
  int64_t mfma_shadow_end = 0;

  for (size_t i = 0; i < wave.instructions_size; i++) {
    const auto& inst = wave.instructions_array[i];
    if (is_null_pc(inst) || !edge.contains(inst.pc)) continue;
    auto& slot = body[body_slot(edge, inst.pc.address)];

    // New iteration starts when we hit the loop header.
    if (inst.pc.address == edge.target_addr)
      iter_start = inst.time + inst.stall;
    if (iter_start >= 0) {
      int64_t rel = (inst.time + inst.stall) - iter_start;
      if (rel >= 0) slot.rel_issue += static_cast<uint64_t>(rel);
    }

    if (!mfma_coexec) continue;
    if (!slot.kind) slot.kind = kind_at(inst.pc);
    if (*slot.kind == InstKind::Mfma) {
      // MFMA is productive work — update shadow, don't count as bubble.
      int64_t issue_point = inst.time + inst.stall;
      // Verify no two MFMAs issue within each other's shadow. An overlap of
      // up to one timestamp granule is decoder slop, not a pipe violation --
      // only a larger one means kMfmaShadowCycles is wrong for this ISA.
      if (mfma_shadow_end > 0 &&
          issue_point < mfma_shadow_end - kAttTimestampGranule)
        overlaps.push_back(
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
        if (bubble > 0) slot.bubble += static_cast<uint64_t>(bubble);
      }
    }
  }
}

LoopRecordMetrics LoopProfile::metrics(const LoopRecord& record,
                                       const LoopBody& body) const {
  const BackEdge& edge = record.back_edge;
  auto slots = body_slots.find(edge);
  LoopRecordMetrics result;
  result.rows.reserve(body.size());
  for (const auto& inst : body) {
    pc_t pc{inst.vaddr, edge.code_object_id};
    assert(edge.contains(pc) && "body instruction outside its loop");
    LoopInstRow row{inst.vaddr, inst.text, inst.kind, 0, 0,    0,
                    0,          0,         0,         0, false};
    auto totals = pc_totals.find(pc);
    if (totals != pc_totals.end()) {
      row.n = totals->second.count;
      row.dur = totals->second.duration;
      row.stall = totals->second.stall;
      row.idle = totals->second.idle;
      row.issue = totals->second.issue;
      row.has_stats = true;
      if (slots != body_slots.end()) {
        const auto& slot = slots->second[body_slot(edge, inst.vaddr)];
        row.bubble = slot.bubble;
        row.rel_issue = slot.rel_issue;
      }
    }
    result.rows.push_back(row);
  }
  result.summary = summarize(record.stats, result.rows);
  return result;
}

LoopSummary LoopProfile::summarize(const LoopStats& stats,
                                   llvm::ArrayRef<LoopInstRow> rows) {
  LoopSummary sum{};
  uint64_t total_bubble = 0;
  // Rows sum over the trace's waves rather than the record's passes, so the
  // per-iteration bubble divides by a row's n instead of the iteration count:
  // the largest, that of an instruction every pass runs.
  uint64_t total_n = 0;
  for (const auto& row : rows) {
    total_bubble += row.bubble;
    if (row.has_stats && row.n > total_n) total_n = row.n;
    if (row.kind == InstKind::Mfma) sum.mfma_count++;
  }
  if (stats.iteration_count > 0) {
    sum.avg_dur =
        static_cast<double>(stats.total_duration) / stats.iteration_count;
    sum.avg_stall =
        static_cast<double>(stats.total_stall) / stats.iteration_count;
    sum.avg_idle =
        static_cast<double>(stats.total_idle) / stats.iteration_count;
  }
  if (total_n > 0) sum.avg_bubble = static_cast<double>(total_bubble) / total_n;
  sum.mfma_time = sum.mfma_count * kMfmaShadowCycles;
  if (sum.avg_dur > 0)
    sum.mfma_util = static_cast<double>(sum.mfma_time) / sum.avg_dur;
  return sum;
}

}  // namespace my_rocperf_tool
