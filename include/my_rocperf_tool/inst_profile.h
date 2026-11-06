#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "my_rocperf_tool/aggregate_scope.h"
#include "my_rocperf_tool/inst_kind.h"
#include "my_rocperf_tool/mfma_shadow.h"
#include "my_rocperf_tool/trace_pc.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace my_rocperf_tool {

enum class InstProfileKind { Static, Dynamic };

// One key's sums over a group's executions of it.
struct InstSums {
  uint64_t n = 0;           // executions
  uint64_t exec_waves = 0;  // waves with at least one execution
  uint64_t dur = 0, stall = 0, idle = 0, issue = 0, bubble = 0, rel_issue = 0;
};

struct InstRow {
  rocprofiler_thread_trace_decoder_pc_t pc;
  uint32_t occurrence;  // 1-based in a dynamic profile, 0 in a static one
  InstKind kind;
  std::string_view text;  // empty from the profile; the CLI fills it in
  InstSums sums;
};

// Per-wave averages over a group's waves; all 0 for a group of no records.
struct InstSummary {
  uint64_t waves = 0;
  double avg_records = 0, avg_dur = 0, avg_stall = 0, avg_idle = 0,
         avg_bubble = 0;
  double avg_mfma = 0;  // MFMA executions per wave
  double mfma_util =
      0;  // avg_mfma * kMfmaShadowCycles / avg_dur, 0 if avg_dur is 0
};

struct InstGroup {
  AggregateScope scope;
  WaveSlot slot;        // unused under All
  uint64_t wave_index;  // the wave, under Wave only
  InstSummary summary;
  std::vector<InstRow> rows;
};

class InstProfile {
 public:
  using pc_t = rocprofiler_thread_trace_decoder_pc_t;
  using wave_t = rocprofiler_thread_trace_decoder_wave_t;
  using KindLookup = llvm::function_ref<InstKind(const pc_t&)>;

  InstProfile(InstProfileKind kind, AggregateScope scope, bool mfma_coexec);
  // `wave` has its s_barrier token pairs coalesced. kind_at is asked for
  // every attributed record (WaveAnalysis caches it per PC). Returns the
  // wave's MFMA shadow overlaps, under mfma_coexec.
  std::vector<MfmaShadowOverlap> add_wave(const wave_t& wave,
                                          uint64_t wave_index,
                                          KindLookup kind_at);
  // Consumes the profile. In report order: Wave groups by (slot, wave_index);
  // SimdSlot groups by slot; one All group, or none if no wave was added.
  std::vector<InstGroup> groups() &&;

 private:
  using Key = std::pair<pc_t, uint32_t>;
  struct GroupTotals {
    WaveSlot slot;
    uint64_t wave_index;
    uint64_t waves = 0;
    llvm::DenseMap<Key, InstRow> rows = llvm::DenseMap<Key, InstRow>();
  };

  InstProfileKind kind;
  AggregateScope scope;
  bool mfma_coexec;
  std::vector<GroupTotals> totals;
  llvm::DenseMap<WaveSlot, size_t> slot_groups;
};

}  // namespace my_rocperf_tool
