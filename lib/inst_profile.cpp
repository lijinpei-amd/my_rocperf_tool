#include "my_rocperf_tool/inst_profile.h"

#include "my_rocperf_tool/loop_detection.h"

#include <algorithm>
#include <optional>
#include <tuple>

namespace my_rocperf_tool {
namespace {

void merge(InstSums& to, const InstSums& from) {
  to.n += from.n;
  to.exec_waves += from.exec_waves;
  to.dur += from.dur;
  to.stall += from.stall;
  to.idle += from.idle;
  to.issue += from.issue;
  to.bubble += from.bubble;
  to.rel_issue += from.rel_issue;
}

InstSummary summarize(uint64_t waves, const std::vector<InstRow>& rows) {
  InstSummary summary;
  summary.waves = waves;
  if (!waves) return summary;
  InstSums sums;
  uint64_t mfmas = 0;
  for (const auto& row : rows) {
    merge(sums, row.sums);
    if (row.kind == InstKind::Mfma) mfmas += row.sums.n;
  }
  summary.avg_records = static_cast<double>(sums.n) / waves;
  summary.avg_dur = static_cast<double>(sums.dur) / waves;
  summary.avg_stall = static_cast<double>(sums.stall) / waves;
  summary.avg_idle = static_cast<double>(sums.idle) / waves;
  summary.avg_bubble = static_cast<double>(sums.bubble) / waves;
  summary.avg_mfma = static_cast<double>(mfmas) / waves;
  if (summary.avg_dur > 0)
    summary.mfma_util = summary.avg_mfma * kMfmaShadowCycles / summary.avg_dur;
  return summary;
}

}  // namespace

InstProfile::InstProfile(InstProfileKind kind, AggregateScope scope,
                         bool mfma_coexec)
    : kind(kind), scope(scope), mfma_coexec(mfma_coexec) {}

std::vector<MfmaShadowOverlap> InstProfile::add_wave(const wave_t& wave,
                                                     uint64_t wave_index,
                                                     KindLookup kind_at) {
  llvm::DenseMap<Key, InstRow> rows;
  llvm::DenseMap<pc_t, uint32_t> occurrences;
  std::vector<MfmaShadowOverlap> overlaps;
  MfmaShadow shadow;
  int64_t last_time = wave.begin_time;
  std::optional<int64_t> first_issue;
  for (size_t i = 0; i < wave.instructions_size; i++) {
    const auto& inst = wave.instructions_array[i];
    if (is_null_pc(inst)) continue;
    uint32_t occurrence =
        kind == InstProfileKind::Dynamic ? ++occurrences[inst.pc] : 0;
    InstKind inst_kind = kind_at(inst.pc);
    auto [it, inserted] = rows.try_emplace({inst.pc, occurrence});
    auto& row = it->second;
    if (inserted) {
      row.pc = inst.pc;
      row.occurrence = occurrence;
      row.kind = inst_kind;
      row.sums.exec_waves = 1;
    }
    auto& sums = row.sums;
    sums.n++;
    sums.dur += inst.duration;
    sums.stall += inst.stall;
    sums.idle += compute_idle(inst, last_time);
    sums.issue += issue_cycles(inst);
    int64_t issue_point = inst.time + inst.stall;
    if (!first_issue) first_issue = issue_point;
    int64_t rel = issue_point - *first_issue;
    // Drop negative relative issue, as loop rows do for duration below stall.
    if (rel >= 0) sums.rel_issue += static_cast<uint64_t>(rel);
    if (mfma_coexec)
      sums.bubble += shadow.step(inst, inst_kind == InstKind::Mfma, overlaps);
  }

  WaveSlot slot{wave.cu, wave.simd, wave.wave_id};
  size_t group_index = 0;
  if (scope == AggregateScope::Wave) {
    group_index = totals.size();
    totals.push_back({slot, wave_index, 1, std::move(rows)});
    return overlaps;
  }
  if (scope == AggregateScope::SimdSlot) {
    auto [it, inserted] = slot_groups.try_emplace(slot, totals.size());
    group_index = it->second;
    if (inserted) totals.push_back({slot, 0});
  } else if (totals.empty()) {
    totals.push_back({{}, 0});
  }
  auto& group = totals[group_index];
  group.waves++;
  for (const auto& entry : rows) {
    auto [it, inserted] = group.rows.try_emplace(entry.first, entry.second);
    if (!inserted) merge(it->second.sums, entry.second.sums);
  }
  return overlaps;
}

std::vector<InstGroup> InstProfile::groups() && {
  std::vector<InstGroup> result;
  result.reserve(totals.size());
  for (auto& group : totals) {
    InstGroup g{scope, group.slot, group.wave_index, {}, {}};
    g.rows.reserve(group.rows.size());
    for (auto& entry : group.rows) g.rows.push_back(std::move(entry.second));
    // Release the map's buckets before converting the next group; clear()
    // may retain them even after the rows have been moved out.
    group.rows = llvm::DenseMap<Key, InstRow>();
    std::sort(
        g.rows.begin(), g.rows.end(),
        [this](const InstRow& a, const InstRow& b) {
          if (kind == InstProfileKind::Dynamic) {
            double avg_a = static_cast<double>(a.sums.rel_issue) / a.sums.n;
            double avg_b = static_cast<double>(b.sums.rel_issue) / b.sums.n;
            if (avg_a != avg_b) return avg_a < avg_b;
          }
          return std::tie(a.pc.code_object_id, a.pc.address, a.occurrence) <
                 std::tie(b.pc.code_object_id, b.pc.address, b.occurrence);
        });
    g.summary = summarize(group.waves, g.rows);
    result.push_back(std::move(g));
  }
  totals.clear();
  slot_groups.clear();
  std::sort(result.begin(), result.end(),
            [](const InstGroup& a, const InstGroup& b) {
              if (a.slot != b.slot) return a.slot < b.slot;
              return a.wave_index < b.wave_index;
            });
  return result;
}

}  // namespace my_rocperf_tool
