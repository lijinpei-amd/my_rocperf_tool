#include "my_rocperf_tool/loop_profile.h"

#include "my_rocperf_tool/inst_kind.h"
#include "my_rocperf_tool/loop_detection.h"

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

LoopSummary summarize(const LoopStats& stats, uint64_t bubble,
                      const LoopBody& body) {
  LoopSummary sum{};
  for (const auto& inst : body)
    if (inst.kind == InstKind::Mfma) sum.mfma_count++;
  if (stats.iteration_count > 0) {
    sum.avg_dur =
        static_cast<double>(stats.total_duration) / stats.iteration_count;
    sum.avg_stall =
        static_cast<double>(stats.total_stall) / stats.iteration_count;
    sum.avg_idle =
        static_cast<double>(stats.total_idle) / stats.iteration_count;
    sum.avg_bubble = static_cast<double>(bubble) / stats.iteration_count;
  }
  sum.mfma_time = sum.mfma_count * kMfmaShadowCycles;
  if (sum.avg_dur > 0)
    sum.mfma_util = static_cast<double>(sum.mfma_time) / sum.avg_dur;
  return sum;
}

}  // namespace

LoopProfile::LoopProfile(bool mfma_coexec, bool wave_rows)
    : mfma_coexec(mfma_coexec), wave_rows(wave_rows) {}

std::vector<MfmaShadowOverlap> LoopProfile::add_wave(const wave_t& wave,
                                                     const WaveLoopInfo& loops,
                                                     KindLookup kind_at) {
  std::vector<MfmaShadowOverlap> overlaps;
  for (const auto& loop : loops.loops)
    add_passes(wave, loops.wave_index, loop.back_edge, kind_at, overlaps);
  return overlaps;
}

void LoopProfile::add_passes(const wave_t& wave, uint64_t wave_index,
                             const BackEdge& edge, KindLookup kind_at,
                             std::vector<MfmaShadowOverlap>& overlaps) {
  const size_t body_size = body_slot(edge, edge.source_addr) + 1;
  auto& loop = loop_totals[edge];
  if (loop.aggregate.slots.empty()) {
    loop.aggregate.slots.resize(body_size);
    loop.kinds.resize(body_size);
  }
  [[maybe_unused]] auto [record_it, inserted] =
      wave_totals.try_emplace({wave_index, edge});
  assert(inserted && "a loop added twice for one wave");
  RecordTotals& record = record_it->second;
  std::vector<SlotTotals>& slots = wave_rows ? record.slots : scratch_slots;
  slots.assign(body_size, SlotTotals{});

  int64_t last_time = wave.begin_time;
  int64_t iter_start = -1;
  int64_t mfma_shadow_end = 0;
  for (size_t i = 0; i < wave.instructions_size; i++) {
    const auto& inst = wave.instructions_array[i];
    if (is_null_pc(inst)) continue;
    // Idle runs from the end of the previous record, in the body or not, as
    // detect_loops counts it.
    uint32_t idle = compute_idle(inst, last_time);
    if (!edge.contains(inst.pc)) continue;
    const size_t index = body_slot(edge, inst.pc.address);
    auto& slot = slots[index];
    slot.count++;
    slot.duration += inst.duration;
    slot.stall += inst.stall;
    slot.idle += idle;
    // A record whose stall exceeds its duration (kept unless the duration
    // check rejects the trace) issued in no time rather than negative time.
    int32_t issue = inst.duration - static_cast<int32_t>(inst.stall);
    slot.issue += issue > 0 ? static_cast<uint64_t>(issue) : 0;

    // New iteration starts when we hit the loop header.
    if (inst.pc.address == edge.target_addr)
      iter_start = inst.time + inst.stall;
    if (iter_start >= 0) {
      int64_t rel = (inst.time + inst.stall) - iter_start;
      if (rel >= 0) slot.rel_issue += static_cast<uint64_t>(rel);
    }

    if (!mfma_coexec) continue;
    auto& kind = loop.kinds[index];
    if (!kind) kind = kind_at(inst.pc);
    if (*kind == InstKind::Mfma) {
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

  for (size_t i = 0; i < body_size; i++) {
    loop.aggregate.slots[i] += slots[i];
    record.bubble += slots[i].bubble;
  }
  loop.aggregate.bubble += record.bubble;
}

LoopRecordMetrics LoopProfile::metrics(const LoopRecord& record,
                                       const LoopBody& body) const {
  const BackEdge& edge = record.back_edge;
  const RecordTotals* totals = nullptr;
  if (record.wave) {
    auto it = wave_totals.find({*record.wave, edge});
    assert(it != wave_totals.end() && "the record's wave lacks its loop");
    if (it != wave_totals.end()) totals = &it->second;
  } else if (auto it = loop_totals.find(edge); it != loop_totals.end()) {
    totals = &it->second.aggregate;
  }

  LoopRecordMetrics result;
  if (!record.wave || wave_rows) {
    result.rows.reserve(body.size());
    for (const auto& inst : body) {
      assert(edge.contains(pc_t{inst.vaddr, edge.code_object_id}) &&
             "body instruction outside its loop");
      LoopInstRow row{inst.vaddr, inst.text, inst.kind, 0, 0,    0,
                      0,          0,         0,         0, false};
      if (totals) {
        const auto& slot = totals->slots[body_slot(edge, inst.vaddr)];
        row.n = slot.count;
        row.dur = slot.duration;
        row.stall = slot.stall;
        row.idle = slot.idle;
        row.issue = slot.issue;
        row.bubble = slot.bubble;
        row.rel_issue = slot.rel_issue;
        row.has_stats = slot.count > 0;
      }
      result.rows.push_back(row);
    }
  }
  result.summary = summarize(record.stats, totals ? totals->bubble : 0, body);
  return result;
}

}  // namespace my_rocperf_tool
