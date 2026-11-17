#pragma once

#include "my_rocperf_tool/aggregate_scope.h"
#include "my_rocperf_tool/inst_kind.h"
#include "my_rocperf_tool/loop_detection.h"
#include "my_rocperf_tool/mfma_shadow.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include "trace_decoder_types.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace my_rocperf_tool {

// One instruction of a loop body, as the instruction source lists it.
struct LoopBodyInst {
  uint64_t vaddr;
  std::string text;
  InstKind kind;
};

// A loop's instructions in address order, header to back-edge branch. Plain
// data, so the profile needs no code object; the CLI lists it from
// ObjectFileInfo.
using LoopBody = std::vector<LoopBodyInst>;

// Names one record of a loop: the passes of one wave, the waves of one slot,
// or every wave in which detect_loops found it.
struct LoopRecord {
  BackEdge back_edge;
  // detect_loops' totals over the record's passes.
  LoopStats stats;
  // The waves the record covers, 1 for a per-wave record. Only the report
  // prints it: LoopProfile::metrics does not read it.
  uint32_t wave_count;
  // A per-wave record's wave, by WaveLoopInfo::wave_index; nullopt for a
  // per-slot or aggregate record.
  std::optional<uint64_t> wave;
  // Set only for a per-slot record; wave and slot are mutually exclusive.
  std::optional<WaveSlot> slot;
};

// One body instruction's metrics in a loop record.
struct LoopInstRow {
  uint64_t vaddr;
  // Views the body's text, so it is valid while the body is.
  std::string_view text;
  InstKind kind;
  // Executions, and their sums: issue is max(duration - stall, 0) per
  // execution.
  uint64_t n;
  uint64_t dur, stall, idle, issue;
  // Cycles no MFMA shadow covered; 0 unless the profile analyzes MFMA
  // co-execution.
  uint64_t bubble;
  // Issue times relative to the issue of the pass's loop header.
  uint64_t rel_issue;
  // n > 0. A row without stats has every metric 0.
  bool has_stats;
};

// A loop record's per-iteration averages and MFMA occupancy.
struct LoopSummary {
  double avg_dur, avg_stall, avg_idle, avg_bubble;
  double mfma_util;  // mfma_time / avg_dur
  uint32_t mfma_count;
  uint32_t mfma_time;  // Sum of the body MFMAs' InstKind::mfma_cycles
};

// What the profile answers for one loop record: a row per body instruction,
// in body order, and the summary. A wave's record has no rows when the
// profile was built without wave_rows.
struct LoopRecordMetrics {
  std::vector<LoopInstRow> rows;
  LoopSummary summary;
};

// The per-loop, per-PC metrics of one trace's waves. Waves are added one at a
// time, each with the loops detect_loops found in it, and nothing of a wave is
// kept but running sums: a bubble total per loop record, rows for per-slot
// and aggregate records, and per-wave rows when requested. Memory grows with
// the loop bodies and the waves' loops, not with the records.
//
// A query names a per-wave (wave set), per-slot (slot set) or aggregate
// (neither set) record. Every number counts that record's executions, so
// averages per execution and per iteration divide within one population:
// - a wave's record: each execution, in that wave, of a PC in the loop's
//   body, which is what detect_loops sums into that wave's LoopStats;
// - a slot's record: those of its waves in which detect_loops found the loop,
//   the waves whose LoopStats the per-slot record sums;
// - the aggregate: those of the waves in which detect_loops found the loop,
//   the waves whose LoopStats the aggregate record sums.
// A wave that runs a body without detect_loops finding the loop adds to no
// record of it. avg_bubble is the record's bubble per iteration, like the
// dur, stall and idle averages beside it.
class LoopProfile {
 public:
  using pc_t = rocprofiler_thread_trace_decoder_pc_t;
  using wave_t = rocprofiler_thread_trace_decoder_wave_t;
  // The kind of the instruction at an attributed PC.
  using KindLookup = llvm::function_ref<InstKind(const pc_t&)>;

  LoopProfile() = default;
  // With mfma_coexec, bubbles are accumulated over each loop's body. Without
  // wave_rows, a wave's record keeps only what its summary needs, and its
  // metrics have no rows.
  explicit LoopProfile(bool mfma_coexec, bool wave_rows = true);

  // Adds a wave's records, with each s_barrier's MSG+IMMED token pair
  // coalesced, and `loops`, what detect_loops found in them; a loop's records
  // are filed under loops.wave_index, which must differ from every earlier
  // wave's. Only an assert checks that: under NDEBUG a repeated index would
  // re-zero that wave's rows while its record's bubble kept adding. kind_at
  // is asked only under mfma_coexec, at most once per PC of each loop body.
  // Returns the MFMA shadow overlaps the wave's loops show, in the order
  // found.
  std::vector<MfmaShadowOverlap> add_wave(const wave_t& wave,
                                          const WaveLoopInfo& loops,
                                          KindLookup kind_at);

  // `body` is the listing of record.back_edge's body. A wave's record must
  // name a wave added with that loop; the aggregate of a loop never added has
  // every row 0.
  LoopRecordMetrics metrics(const LoopRecord& record,
                            const LoopBody& body) const;

 private:
  // A body dword's sums over a record's executions.
  struct DwordTotals {
    uint64_t count = 0;
    uint64_t duration = 0;
    uint64_t stall = 0;
    uint64_t idle = 0;
    uint64_t issue = 0;
    uint64_t bubble = 0;
    uint64_t rel_issue = 0;

    DwordTotals& operator+=(const DwordTotals& other) {
      count += other.count;
      duration += other.duration;
      stall += other.stall;
      idle += other.idle;
      issue += other.issue;
      bubble += other.bubble;
      rel_issue += other.rel_issue;
      return *this;
    }
  };
  // One record's sums. The bubble total is kept apart from the dwords, so a
  // wave's summary needs no rows.
  struct RecordTotals {
    // Sums per body dword: the walk touches every record of every pass, so
    // its lookup must be an index rather than a hash. Empty for a wave's
    // record without wave_rows.
    std::vector<DwordTotals> dwords;
    uint64_t bubble = 0;
  };
  // A loop's aggregate record, and the kind of the instruction at each body
  // dword once the MFMA analysis has asked.
  struct LoopTotals {
    RecordTotals aggregate;
    std::vector<std::optional<InstKind>> kinds;
  };

  // Sums the wave's executions of `edge`'s body into the wave's, slot's and
  // aggregate records, relative issue and, under mfma_coexec, MFMA-shadow
  // bubbles included, in one walk.
  void add_passes(const wave_t& wave, uint64_t wave_index, WaveSlot wave_slot,
                  const BackEdge& edge, KindLookup kind_at,
                  std::vector<MfmaShadowOverlap>& overlaps);

  bool mfma_coexec = false;
  bool wave_rows = true;
  llvm::DenseMap<BackEdge, LoopTotals> loop_totals;
  // Keyed by wave_index and loop.
  llvm::DenseMap<std::pair<uint64_t, BackEdge>, RecordTotals> wave_totals;
  llvm::DenseMap<std::pair<WaveSlot, BackEdge>, RecordTotals> slot_totals;
  // The walk's dwords when the wave's are not kept, reused across waves so
  // that only kept rows allocate.
  std::vector<DwordTotals> scratch_dwords;
};

}  // namespace my_rocperf_tool
