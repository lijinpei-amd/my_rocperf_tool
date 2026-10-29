#pragma once

#include "my_rocperf_tool/inst_kind.h"
#include "my_rocperf_tool/loop_detection.h"

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

// Cycles an MFMA is assumed to keep the MAI pipe busy, during which a
// co-issued non-MFMA instruction executes for free.
inline constexpr int64_t kMfmaShadowCycles = 16;

// ATT hardware timestamps land on a 4-cycle grid, and the decoder splits the
// cycles between two timestamp packets across the instructions it saw in
// between. A single instruction's duration can therefore be off by one
// granule.
inline constexpr int64_t kAttTimestampGranule = 4;

// An MFMA that issued inside the previous MFMA's shadow by more than one
// timestamp granule: more than decoder slop, so kMfmaShadowCycles is wrong for
// this ISA.
struct MfmaShadowOverlap {
  rocprofiler_thread_trace_decoder_pc_t pc;
  int64_t issue_point;
  int64_t shadow_end;
};

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

// Names one record of a loop: the passes one wave made through its body, or
// those of every wave in which detect_loops found it.
struct LoopRecord {
  BackEdge back_edge;
  // detect_loops' totals over the record's passes.
  LoopStats stats;
  // The waves the record covers, 1 for a per-wave record. Only the report
  // prints it: LoopProfile::metrics does not read it.
  uint32_t wave_count;
  // A per-wave record's wave, by WaveLoopInfo::wave_index; nullopt for the
  // aggregate.
  std::optional<uint64_t> wave;
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
  uint32_t mfma_time;  // mfma_count * kMfmaShadowCycles
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
// kept but running sums: a bubble total per loop it ran, and that loop's rows
// only where a wave's rows are kept. Memory grows with the loop bodies and the
// waves' loops, not with the records.
//
// A query names the record it is for, a wave's or the aggregate, and every
// number in the answer counts that record's executions, so averages per
// execution and per iteration divide within one population:
// - a wave's record: each execution, in that wave, of a PC in the loop's
//   body, which is what detect_loops sums into that wave's LoopStats;
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
  struct SlotTotals {
    uint64_t count = 0;
    uint64_t duration = 0;
    uint64_t stall = 0;
    uint64_t idle = 0;
    uint64_t issue = 0;
    uint64_t bubble = 0;
    uint64_t rel_issue = 0;

    SlotTotals& operator+=(const SlotTotals& other) {
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
  // One record's sums. The bubble total is kept apart from the slots, so a
  // wave's summary needs no rows.
  struct RecordTotals {
    // A slot per body dword: the walk touches every record of every pass, so
    // its lookup must be an index rather than a hash. Empty for a wave's
    // record without wave_rows.
    std::vector<SlotTotals> slots;
    uint64_t bubble = 0;
  };
  // A loop's aggregate record, and the kind of the instruction at each body
  // dword once the MFMA analysis has asked.
  struct LoopTotals {
    RecordTotals aggregate;
    std::vector<std::optional<InstKind>> kinds;
  };

  // Sums the wave's executions of `edge`'s body into the wave's record and the
  // loop's aggregate, relative issue and, under mfma_coexec, MFMA-shadow
  // bubbles included, in one walk.
  void add_passes(const wave_t& wave, uint64_t wave_index, const BackEdge& edge,
                  KindLookup kind_at, std::vector<MfmaShadowOverlap>& overlaps);

  bool mfma_coexec = false;
  bool wave_rows = true;
  llvm::DenseMap<BackEdge, LoopTotals> loop_totals;
  // Keyed by wave_index and loop.
  llvm::DenseMap<std::pair<uint64_t, BackEdge>, RecordTotals> wave_totals;
  // The walk's slots when the wave's are not kept, reused across waves so
  // that only kept rows allocate.
  std::vector<SlotTotals> scratch_slots;
};

}  // namespace my_rocperf_tool
