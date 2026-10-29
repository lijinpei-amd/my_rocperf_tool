#pragma once

#include "my_rocperf_tool/inst_kind.h"
#include "my_rocperf_tool/loop_detection.h"
#include "my_rocperf_tool/trace_pc.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include "trace_decoder_types.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
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
  // 1 for a per-wave record.
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
// in body order, and the summary.
struct LoopRecordMetrics {
  std::vector<LoopInstRow> rows;
  LoopSummary summary;
};

// The per-loop, per-PC metrics of one trace's waves. Waves are added one at a
// time, each with the loops detect_loops found in it, and nothing of a wave is
// kept but running sums: memory grows with the PCs and loop bodies seen, not
// with the records.
//
// A query names the record it is for, a wave's or the aggregate, so that what
// each number covers is decided here rather than by the callers. Only the
// summary's dur, stall and idle averages follow the record, through its
// LoopStats; every record of a loop gets the same trace-wide rows. Their n,
// dur, stall, idle and issue count each execution of the PC in every wave
// added, and their bubble and rel_issue the passes of the waves in which
// detect_loops found the loop. As n counts over the trace rather than the
// record's passes, avg_bubble divides the loop's bubble by the largest n among
// its rows with stats.
class LoopProfile {
 public:
  using pc_t = rocprofiler_thread_trace_decoder_pc_t;
  using wave_t = rocprofiler_thread_trace_decoder_wave_t;
  // The kind of the instruction at an attributed PC.
  using KindLookup = llvm::function_ref<InstKind(const pc_t&)>;

  LoopProfile() = default;
  // With mfma_coexec, bubbles are accumulated over each loop's body.
  explicit LoopProfile(bool mfma_coexec);

  // Adds a wave's records, with each s_barrier's MSG+IMMED token pair
  // coalesced, and `loops`, what detect_loops found in them. kind_at is asked
  // only under mfma_coexec, at most once per PC of each loop body. Returns the
  // MFMA shadow overlaps the wave's loops show, in the order found.
  std::vector<MfmaShadowOverlap> add_wave(const wave_t& wave,
                                          const WaveLoopInfo& loops,
                                          KindLookup kind_at);

  // `body` is the listing of record.back_edge's body.
  LoopRecordMetrics metrics(const LoopRecord& record,
                            const LoopBody& body) const;

 private:
  struct PcTotals {
    uint64_t count = 0;
    uint64_t duration = 0;
    uint64_t stall = 0;
    uint64_t idle = 0;
    uint64_t issue = 0;
  };
  // A loop body dword's totals, and the kind of the instruction there once
  // the MFMA analysis has asked.
  struct BodySlot {
    uint64_t bubble = 0;
    uint64_t rel_issue = 0;
    std::optional<InstKind> kind;
  };

  // Relative issue and, under mfma_coexec, MFMA-shadow bubbles over the
  // wave's passes through `edge`'s body, in one walk.
  void add_passes(const wave_t& wave, const BackEdge& edge, KindLookup kind_at,
                  std::vector<MfmaShadowOverlap>& overlaps);
  static LoopSummary summarize(const LoopStats& stats,
                               llvm::ArrayRef<LoopInstRow> rows);

  bool mfma_coexec = false;
  llvm::DenseMap<pc_t, PcTotals> pc_totals;
  // Per loop, a slot per dword of its body: the analyses touch every record
  // of every pass, so their lookup must be an index rather than a hash.
  llvm::DenseMap<BackEdge, std::vector<BodySlot>> body_slots;
};

}  // namespace my_rocperf_tool
