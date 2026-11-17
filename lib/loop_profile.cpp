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

// Instructions are dword-aligned, so a body PC indexes its dword offset.
size_t body_dword(const BackEdge& edge, uint64_t addr) {
  return (addr - edge.target_addr) / 4;
}

LoopSummary summarize(const LoopStats& stats, uint64_t bubble,
                      const LoopBody& body) {
  LoopSummary sum{};
  for (const auto& inst : body) {
    if (inst.kind.cls == InstClass::Mfma) {
      sum.mfma_count++;
      sum.mfma_time += inst.kind.mfma_cycles;
    }
  }
  if (stats.iteration_count > 0) {
    sum.avg_dur =
        static_cast<double>(stats.total_duration) / stats.iteration_count;
    sum.avg_stall =
        static_cast<double>(stats.total_stall) / stats.iteration_count;
    sum.avg_idle =
        static_cast<double>(stats.total_idle) / stats.iteration_count;
    sum.avg_bubble = static_cast<double>(bubble) / stats.iteration_count;
  }
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
    add_passes(wave, loops.wave_index, {loops.cu, loops.simd, loops.wave_id},
               loop.back_edge, kind_at, overlaps);
  return overlaps;
}

void LoopProfile::add_passes(const wave_t& wave, uint64_t wave_index,
                             WaveSlot wave_slot, const BackEdge& edge,
                             KindLookup kind_at,
                             std::vector<MfmaShadowOverlap>& overlaps) {
  const size_t body_size = body_dword(edge, edge.source_addr) + 1;
  auto& loop = loop_totals[edge];
  if (loop.aggregate.dwords.empty()) {
    loop.aggregate.dwords.resize(body_size);
    loop.kinds.resize(body_size);
  }
  [[maybe_unused]] auto [record_it, inserted] =
      wave_totals.try_emplace({wave_index, edge});
  assert(inserted && "a loop added twice for one wave");
  RecordTotals& record = record_it->second;
  std::vector<DwordTotals>& dwords = wave_rows ? record.dwords : scratch_dwords;
  dwords.assign(body_size, DwordTotals{});

  int64_t last_time = wave.begin_time;
  int64_t iter_start = -1;
  MfmaShadow shadow;
  for (size_t i = 0; i < wave.instructions_size; i++) {
    const auto& inst = wave.instructions_array[i];
    if (is_null_pc(inst)) continue;
    // Idle runs from the end of the previous record, in the body or not, as
    // detect_loops counts it.
    uint32_t idle = compute_idle(inst, last_time);
    if (!edge.contains(inst.pc)) continue;
    const size_t index = body_dword(edge, inst.pc.address);
    auto& dword = dwords[index];
    dword.count++;
    dword.duration += inst.duration;
    dword.stall += inst.stall;
    dword.idle += idle;
    dword.issue += issue_cycles(inst);

    // New iteration starts when we hit the loop header.
    if (inst.pc.address == edge.target_addr)
      iter_start = inst.time + inst.stall;
    if (iter_start >= 0) {
      int64_t rel = (inst.time + inst.stall) - iter_start;
      if (rel >= 0) dword.rel_issue += static_cast<uint64_t>(rel);
    }

    if (!mfma_coexec) continue;
    auto& kind = loop.kinds[index];
    if (!kind) kind = kind_at(inst.pc);
    dword.bubble += shadow.step(inst, *kind, overlaps);
  }

  auto& slot_record = slot_totals[{wave_slot, edge}];
  slot_record.dwords.resize(body_size);
  for (size_t i = 0; i < body_size; i++) {
    slot_record.dwords[i] += dwords[i];
    loop.aggregate.dwords[i] += dwords[i];
    record.bubble += dwords[i].bubble;
  }
  loop.aggregate.bubble += record.bubble;
  slot_record.bubble += record.bubble;
}

LoopRecordMetrics LoopProfile::metrics(const LoopRecord& record,
                                       const LoopBody& body) const {
  assert(!(record.wave && record.slot) && "record has both wave and slot");
  const BackEdge& edge = record.back_edge;
  const RecordTotals* totals = nullptr;
  if (record.wave) {
    auto it = wave_totals.find({*record.wave, edge});
    assert(it != wave_totals.end() && "the record's wave lacks its loop");
    if (it != wave_totals.end()) totals = &it->second;
  } else if (record.slot) {
    auto it = slot_totals.find({*record.slot, edge});
    if (it != slot_totals.end()) totals = &it->second;
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
        const auto& dword = totals->dwords[body_dword(edge, inst.vaddr)];
        row.n = dword.count;
        row.dur = dword.duration;
        row.stall = dword.stall;
        row.idle = dword.idle;
        row.issue = dword.issue;
        row.bubble = dword.bubble;
        row.rel_issue = dword.rel_issue;
        row.has_stats = dword.count > 0;
      }
      result.rows.push_back(row);
    }
  }
  result.summary = summarize(record.stats, totals ? totals->bubble : 0, body);
  return result;
}

}  // namespace my_rocperf_tool
