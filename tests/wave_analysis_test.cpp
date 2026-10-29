#include "my_rocperf_tool/wave_analysis.h"

#include "my_rocperf_tool/inst_kind.h"
#include "my_rocperf_tool/loop_detection.h"
#include "my_rocperf_tool/loop_profile.h"

#include "synthetic_wave.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <variant>

namespace my_rocperf_tool {
namespace {

WaveAnalysisOptions options_with(DurationCheck duration_check) {
  WaveAnalysisOptions options;
  options.duration_check = duration_check;
  return options;
}

// The rows and summary of `edge` aggregated over the waves of `result` that
// have it. The rows view `body`, which must outlive them.
LoopRecordMetrics loop_metrics(const WaveAnalysisResult& result,
                               const LoopBody& body,
                               const BackEdge& edge = kLoop) {
  LoopRecord record{edge, {}, 0, std::nullopt};
  for (const auto& wave : result.wave_loops) {
    for (const auto& loop : wave.loops) {
      if (loop.back_edge == edge) {
        record.stats += loop.stats;
        record.wave_count++;
      }
    }
  }
  return result.profile.metrics(record, body);
}

TEST(WaveAnalysisTest, CoalescesBarrierTokenPairIntoOneRecord) {
  KindTable table{{{kLoopHeader, InstKind::ClassicBarrier}}};
  WaveBuilder builder;
  builder.at(kPrologue, kSalu, /*time=*/0, /*duration=*/4);
  for (int pass = 0; pass < 2; pass++) {
    const int64_t start = 50 * pass;
    builder.at(kLoopHeader, kMessage, start + 10, /*duration=*/5, /*stall=*/3)
        .at(kLoopHeader, kImmed, start + 15, /*duration=*/7, /*stall=*/2)
        .at(kLoopHeader + 4, kSalu, start + 30, /*duration=*/4)
        .at(kLoopHeader + 8, kSalu, start + 40, /*duration=*/4)
        .at(kLoopBranch, pass == 0 ? kJump : kNext, start + 50,
            /*duration=*/4);
  }
  WaveAnalysis analysis(WaveAnalysisOptions{}, table.classifier());

  analysis.add_wave(builder.wave());
  auto result = std::move(analysis).finish();

  EXPECT_EQ(result.counts.insts, 1 + 2 * 4u);
  auto body = list_body(kLoop);
  auto metrics = loop_metrics(result, body);
  auto barrier = row_at(metrics, kLoopHeader);
  EXPECT_EQ(barrier.n, 2u);
  EXPECT_EQ(barrier.dur, 2 * 12u);
  EXPECT_EQ(barrier.stall, 2 * 5u);
  // Each pass's barrier starts 6 cycles after the end of the record before
  // it.
  EXPECT_EQ(barrier.idle, 2 * 6u);
  // Idle after the barrier runs from the end of the merged record.
  EXPECT_EQ(row_at(metrics, kLoopHeader + 4).idle, 2 * 8u);
}

TEST(WaveAnalysisTest, CoalescedBarrierSaturates) {
  constexpr int32_t kDurationMax = std::numeric_limits<int32_t>::max();
  constexpr uint32_t kStallMax = (1u << 24) - 1;
  KindTable table{{{kLoopHeader, InstKind::ClassicBarrier}}};
  WaveBuilder builder;
  for (int pass = 0; pass < 2; pass++) {
    const int64_t start = 2 * pass * int64_t{kDurationMax};
    builder.at(kLoopHeader, kMessage, start, kDurationMax - 1, kStallMax - 1)
        .at(kLoopHeader, kImmed, start + kDurationMax - 1, /*duration=*/5,
            /*stall=*/5)
        .at(kLoopBranch, pass == 0 ? kJump : kNext, start + kDurationMax + 10,
            /*duration=*/1);
  }
  WaveAnalysis analysis(WaveAnalysisOptions{}, table.classifier());

  analysis.add_wave(builder.wave());
  auto result = std::move(analysis).finish();

  auto body = list_body(kLoop);
  auto barrier = row_at(loop_metrics(result, body), kLoopHeader);
  EXPECT_EQ(barrier.n, 2u);
  EXPECT_EQ(barrier.dur, 2 * uint64_t{kDurationMax});
  EXPECT_EQ(barrier.stall, 2 * uint64_t{kStallMax});
}

TEST(WaveAnalysisTest, SimdFilterDropsOtherWavesBeforeAnyAccumulator) {
  KindTable table;
  WaveBuilder kept(/*simd=*/1);
  kept.iteration(/*branch_taken=*/true)
      .iteration(/*branch_taken=*/true)
      .iteration(/*branch_taken=*/false);
  // Another SIMD runs a loop elsewhere, whose branch records would fail the
  // duration check.
  constexpr uint64_t kOtherHeader = 0x200;
  constexpr BackEdge kOtherLoop{kCodeObject, kOtherHeader + 4, kOtherHeader};
  WaveBuilder dropped(/*simd=*/2);
  for (int pass = 0; pass < 3; pass++)
    dropped.inst(kOtherHeader, kValu)
        .at(kOtherHeader + 4, pass < 2 ? kJump : kNext, 100 * pass + 50,
            /*duration=*/1, /*stall=*/9);
  WaveAnalysisOptions options;
  options.simd = 1;
  options.mfma_coexec = true;
  WaveAnalysis analysis(options, table.classifier());

  analysis.add_wave(dropped.wave());
  analysis.add_wave(kept.wave());
  auto result = std::move(analysis).finish();

  EXPECT_EQ(result.counts.waves, 1u);
  EXPECT_EQ(result.counts.waves_filtered, 1u);
  EXPECT_EQ(result.counts.insts, 12u);
  EXPECT_EQ(result.counts.loop_waves, 1u);
  EXPECT_EQ(result.counts.loops, 1u);
  ASSERT_EQ(result.wave_loops.size(), 1u);
  EXPECT_EQ(result.wave_loops[0].simd, 1);
  // The dropped wave is not numbered either.
  EXPECT_EQ(result.wave_loops[0].wave_index, 0u);
  auto body = list_body(kLoop);
  for (const auto& row : loop_metrics(result, body).rows) EXPECT_EQ(row.n, 3u);
  auto other_body = list_body(kOtherLoop);
  for (const auto& row : loop_metrics(result, other_body, kOtherLoop).rows) {
    EXPECT_FALSE(row.has_stats);
    EXPECT_EQ(row.rel_issue, 0u);
    EXPECT_EQ(row.bubble, 0u);
  }
  EXPECT_TRUE(result.diagnostics.empty());
  EXPECT_FALSE(result.duration_check_failure);
}

TEST(WaveAnalysisTest, WithoutWaveLoopRowsOnlyTheAggregateHasRows) {
  KindTable table;
  WaveBuilder builder;
  builder.iteration(/*branch_taken=*/true).iteration(/*branch_taken=*/false);
  WaveAnalysisOptions options;
  options.wave_loop_rows = false;
  WaveAnalysis analysis(options, table.classifier());

  analysis.add_wave(builder.wave());
  auto result = std::move(analysis).finish();

  ASSERT_EQ(result.wave_loops.size(), 1u);
  const auto& wave = result.wave_loops[0];
  ASSERT_EQ(wave.loops.size(), 1u);
  auto body = list_body(kLoop);
  LoopRecord wave_record{kLoop, wave.loops[0].stats, 1, wave.wave_index};
  EXPECT_TRUE(result.profile.metrics(wave_record, body).rows.empty());
  EXPECT_EQ(loop_metrics(result, body).rows.size(), body.size());
}

// MFMA tests: a loop body of two MFMAs, a VALU and the branch, run twice with
// passes 100 cycles apart so no shadow reaches into the next pass.
constexpr uint64_t kMfma1 = kLoopHeader;
constexpr uint64_t kMfma2 = kLoopHeader + 4;
constexpr uint64_t kValuInLoop = kLoopHeader + 8;
constexpr int64_t kPassCycles = 100;

KindTable mfma_table() {
  return KindTable{{{kMfma1, InstKind::Mfma}, {kMfma2, InstKind::Mfma}}};
}

WaveAnalysisResult analyze_mfma_loop(KindTable& table, WaveBuilder& builder) {
  WaveAnalysisOptions options;
  options.mfma_coexec = true;
  WaveAnalysis analysis(options, table.classifier());
  analysis.add_wave(builder.wave());
  auto result = std::move(analysis).finish();
  EXPECT_EQ(result.counts.loops, 1u);
  return result;
}

TEST(WaveAnalysisMfmaTest, OverlapWithinOneGranuleIsNotReported) {
  static_assert(kMfmaShadowCycles - kAttTimestampGranule == 12);
  auto table = mfma_table();
  WaveBuilder builder;
  for (int pass = 0; pass < 2; pass++) {
    const int64_t start = pass * kPassCycles;
    builder.at(kMfma1, kValu, start, /*duration=*/4)
        .at(kMfma2, kValu, start + 12, /*duration=*/4)
        .at(kValuInLoop, kValu, start + 20, /*duration=*/4)
        .at(kLoopBranch, pass == 0 ? kJump : kNext, start + 30,
            /*duration=*/2);
  }

  auto result = analyze_mfma_loop(table, builder);

  EXPECT_TRUE(result.diagnostics.empty());
}

TEST(WaveAnalysisMfmaTest, LargerOverlapIsReported) {
  auto table = mfma_table();
  WaveBuilder builder;
  for (int pass = 0; pass < 2; pass++) {
    const int64_t start = pass * kPassCycles;
    builder.at(kMfma1, kValu, start, /*duration=*/4)
        .at(kMfma2, kValu, start + 11, /*duration=*/4)
        .at(kLoopBranch, pass == 0 ? kJump : kNext, start + 30,
            /*duration=*/2);
  }

  auto result = analyze_mfma_loop(table, builder);

  ASSERT_EQ(result.diagnostics.size(), 2u);
  for (int pass = 0; pass < 2; pass++) {
    const auto* overlap =
        std::get_if<MfmaShadowOverlap>(&result.diagnostics[pass]);
    ASSERT_NE(overlap, nullptr);
    EXPECT_EQ(overlap->pc.address, kMfma2);
    EXPECT_EQ(overlap->pc.code_object_id, kCodeObject);
    EXPECT_EQ(overlap->issue_point, pass * kPassCycles + 11);
    EXPECT_EQ(overlap->shadow_end, pass * kPassCycles + kMfmaShadowCycles);
  }
}

TEST(WaveAnalysisMfmaTest, ShadowEndNeverRewinds) {
  auto table = mfma_table();
  WaveBuilder builder;
  for (int pass = 0; pass < 2; pass++) {
    const int64_t start = pass * kPassCycles;
    // The first MFMA stalls, so the second issues before it: the shadow end
    // stays at the first's start + 10 + 16 = 26, and does not drop to the
    // second's 2 + 16 = 18.
    builder.at(kMfma1, kValu, start, /*duration=*/12, /*stall=*/10)
        .at(kMfma2, kValu, start + 2, /*duration=*/4)
        .at(kValuInLoop, kValu, start + 20, /*duration=*/6)
        .at(kLoopBranch, pass == 0 ? kJump : kNext, start + 40,
            /*duration=*/1);
  }

  auto result = analyze_mfma_loop(table, builder);

  auto body = list_body(kLoop, table.kinds);
  auto metrics = loop_metrics(result, body);
  EXPECT_EQ(row_at(metrics, kValuInLoop).bubble, 0u);
  EXPECT_EQ(row_at(metrics, kLoopBranch).bubble, 2u);
}

TEST(WaveAnalysisMfmaTest, BubbleIsTheUncoveredPartOfANonMfma) {
  KindTable table{{{kMfma1, InstKind::Mfma}}};
  WaveBuilder builder;
  for (int pass = 0; pass < 2; pass++) {
    const int64_t start = pass * kPassCycles;
    // The shadow covers [start, start + 16).
    builder
        .at(kMfma1, kValu, start, /*duration=*/4)
        // Ends at 8: covered.
        .at(kLoopHeader + 4, kValu, start + 4, /*duration=*/4)
        // Ends at 20: its last 4 cycles are uncovered.
        .at(kLoopHeader + 8, kValu, start + 14, /*duration=*/6)
        // Ends at 33, 17 cycles past the shadow, so all 3 of its cycles are.
        .at(kLoopBranch, pass == 0 ? kJump : kNext, start + 30,
            /*duration=*/3);
  }

  auto result = analyze_mfma_loop(table, builder);

  auto body = list_body(kLoop, table.kinds);
  auto metrics = loop_metrics(result, body);
  EXPECT_EQ(row_at(metrics, kLoopHeader + 4).bubble, 0u);
  EXPECT_EQ(row_at(metrics, kLoopHeader + 8).bubble, 2 * 4u);
  EXPECT_EQ(row_at(metrics, kLoopBranch).bubble, 2 * 3u);
  // Once per PC, not once per record.
  EXPECT_EQ(table.calls, 4);
}

TEST(WaveAnalysisTest, RelativeIssueResetsAtLoopHeader) {
  KindTable table;
  WaveBuilder builder;
  builder.inst(kPrologue, kSalu)
      .iteration(/*branch_taken=*/true)
      .wait(1000)
      .iteration(/*branch_taken=*/true)
      .wait(77)
      .iteration(/*branch_taken=*/false)
      .inst(kEpilogue, kSalu);
  WaveAnalysis analysis(WaveAnalysisOptions{}, table.classifier());

  analysis.add_wave(builder.wave());
  auto result = std::move(analysis).finish();

  // Each pass's records issue 0, 10, 20 and 30 cycles after its header,
  // however long the waits between passes.
  auto body = list_body(kLoop);
  auto metrics = loop_metrics(result, body);
  for (uint64_t i = 0; i < 4; i++)
    EXPECT_EQ(row_at(metrics, kLoopHeader + 4 * i).rel_issue, 3 * i * kSpacing);
}

TEST(WaveAnalysisTest, ThreeIterationsOfAFourInstructionLoop) {
  KindTable table;
  WaveBuilder builder;
  builder.inst(kPrologue, kSalu)
      .iteration(/*branch_taken=*/true)
      .iteration(/*branch_taken=*/true)
      .iteration(/*branch_taken=*/false)
      .inst(kEpilogue, kSalu);
  WaveAnalysis analysis(WaveAnalysisOptions{}, table.classifier());

  analysis.add_wave(builder.wave());
  auto result = std::move(analysis).finish();

  EXPECT_EQ(result.counts.waves, 1u);
  EXPECT_EQ(result.counts.insts, 14u);
  EXPECT_EQ(result.counts.loop_waves, 1u);
  EXPECT_EQ(result.counts.loops, 1u);
  ASSERT_EQ(result.wave_loops.size(), 1u);
  ASSERT_EQ(result.wave_loops[0].loops.size(), 1u);
  const auto& loop = result.wave_loops[0].loops[0];
  EXPECT_EQ(loop.back_edge, kLoop);
  EXPECT_EQ(loop.stats.iteration_count, 3u);
  EXPECT_EQ(loop.stats.total_inst_count, 12u);
  EXPECT_EQ(loop.stats.total_duration, 12u * kDuration);
  EXPECT_EQ(loop.stats.total_stall, 12u * kStall);
  EXPECT_EQ(loop.stats.total_idle, 12u * (kSpacing - kDuration));
  auto body = list_body(kLoop);
  for (const auto& row : loop_metrics(result, body).rows) EXPECT_EQ(row.n, 3u);
  // Guards lazy classification on purpose: only the per-loop MFMA analysis,
  // off here, and an assert on s_barrier token pairs ask for kinds.
  EXPECT_EQ(table.calls, 0);
}

// A record without a PC is not an instruction of the wave: it is neither
// counted nor checked, and adds nothing to a loop's stats or rows, nor to the
// idle time of the record after it.
TEST(WaveAnalysisTest, SkipsNullPcRecords) {
  KindTable table;
  WaveBuilder builder;
  builder.inst(kPrologue, kSalu);
  for (int pass = 0; pass < 3; pass++)
    builder.inst(kLoopHeader, kValu)
        .null_pc()
        .inst(kLoopHeader + 4, kValu)
        .inst(kLoopHeader + 8, kSalu)
        .null_pc()
        .inst(kLoopBranch, pass < 2 ? kJump : kNext);
  builder.inst(kEpilogue, kSalu);
  WaveAnalysis analysis(options_with(DurationCheck::Error), table.classifier());

  analysis.add_wave(builder.wave());
  auto result = std::move(analysis).finish();

  EXPECT_EQ(result.counts.insts, 14u);
  EXPECT_TRUE(result.diagnostics.empty());
  EXPECT_FALSE(result.duration_check_failure);
  ASSERT_EQ(result.wave_loops.size(), 1u);
  ASSERT_EQ(result.wave_loops[0].loops.size(), 1u);
  const auto& stats = result.wave_loops[0].loops[0].stats;
  EXPECT_EQ(stats.iteration_count, 3u);
  EXPECT_EQ(stats.total_inst_count, 12u);
  EXPECT_EQ(stats.total_duration, 12u * kDuration);
  EXPECT_EQ(stats.total_stall, 12u * kStall);
  EXPECT_EQ(stats.total_idle, 12u * (kSpacing - kDuration));
  auto body = list_body(kLoop);
  for (const auto& row : loop_metrics(result, body).rows) {
    EXPECT_EQ(row.n, 3u);
    EXPECT_EQ(row.dur, 3u * kDuration);
    EXPECT_EQ(row.stall, 3u * kStall);
    EXPECT_EQ(row.idle, 3u * (kSpacing - kDuration));
  }
}

// Coalescing runs before loop detection and the per-loop analyses, so each
// pass's barrier reaches them as one record.
TEST(WaveAnalysisTest, LoopAnalysesSeeTheCoalescedBarrier) {
  constexpr uint64_t kBarrier = kLoopHeader + 4;
  KindTable table{{{kBarrier, InstKind::ClassicBarrier}}};
  WaveBuilder builder;
  for (int pass = 0; pass < 3; pass++) {
    const int64_t start = pass * kPassCycles;
    builder.at(kLoopHeader, kValu, start, /*duration=*/4)
        .at(kBarrier, kMessage, start + 10, /*duration=*/5, /*stall=*/3)
        .at(kBarrier, kImmed, start + 15, /*duration=*/7, /*stall=*/2)
        .at(kLoopHeader + 8, kSalu, start + 30, /*duration=*/4)
        .at(kLoopBranch, pass < 2 ? kJump : kNext, start + 40,
            /*duration=*/2);
  }
  WaveAnalysisOptions options;
  options.mfma_coexec = true;
  WaveAnalysis analysis(options, table.classifier());

  analysis.add_wave(builder.wave());
  auto result = std::move(analysis).finish();

  ASSERT_EQ(result.wave_loops.size(), 1u);
  ASSERT_EQ(result.wave_loops[0].loops.size(), 1u);
  const auto& loop = result.wave_loops[0].loops[0];
  EXPECT_EQ(loop.back_edge, kLoop);
  EXPECT_EQ(loop.stats.iteration_count, 3u);
  EXPECT_EQ(loop.stats.total_inst_count, 3 * 4u);
  auto body = list_body(kLoop);
  auto barrier = row_at(loop_metrics(result, body), kBarrier);
  EXPECT_EQ(barrier.n, 3u);
  // The merged record issues after both tokens' stalls: 10 + 3 + 2 cycles
  // after the header.
  EXPECT_EQ(barrier.rel_issue, 3 * 15u);
  // No MFMA shadow covers any of its 5 + 7 cycles.
  EXPECT_EQ(barrier.bubble, 3 * 12u);
}

// Two passes through the loop body, in each of which records 0x104 and 0x108
// stall longer than they last.
WaveAnalysisResult analyze_with_duration_check(DurationCheck duration_check) {
  KindTable table;
  WaveBuilder builder;
  for (int pass = 0; pass < 2; pass++) {
    const int64_t start = 40 * pass;
    builder.at(0x100, kValu, start, /*duration=*/4, /*stall=*/1)
        .at(0x104, kValu, start + 10, /*duration=*/2, /*stall=*/5)
        .at(0x108, kValu, start + 20, /*duration=*/1, /*stall=*/3)
        .at(kLoopBranch, pass == 0 ? kJump : kNext, start + 30,
            /*duration=*/2);
  }
  WaveAnalysis analysis(options_with(duration_check), table.classifier());
  analysis.add_wave(builder.wave());
  return std::move(analysis).finish();
}

// The offending records reach the loop profile, issuing in no time.
void expect_records_kept(const WaveAnalysisResult& result) {
  auto body = list_body(kLoop);
  auto metrics = loop_metrics(result, body);
  EXPECT_EQ(row_at(metrics, 0x100).issue, 2 * 3u);
  auto first = row_at(metrics, 0x104);
  EXPECT_EQ(first.n, 2u);
  EXPECT_EQ(first.dur, 2 * 2u);
  EXPECT_EQ(first.stall, 2 * 5u);
  EXPECT_EQ(first.issue, 0u);
  auto second = row_at(metrics, 0x108);
  EXPECT_EQ(second.n, 2u);
  EXPECT_EQ(second.issue, 0u);
}

TEST(WaveAnalysisDurationCheckTest, OffChecksNothing) {
  auto result = analyze_with_duration_check(DurationCheck::Off);

  EXPECT_TRUE(result.diagnostics.empty());
  EXPECT_FALSE(result.duration_check_failure);
  expect_records_kept(result);
}

TEST(WaveAnalysisDurationCheckTest, WarnReportsEachRecordAndKeepsThem) {
  auto result = analyze_with_duration_check(DurationCheck::Warn);

  ASSERT_EQ(result.diagnostics.size(), 4u);
  const auto* first = std::get_if<DurationBelowStall>(&result.diagnostics[0]);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first->pc.address, 0x104u);
  EXPECT_EQ(first->pc.code_object_id, kCodeObject);
  EXPECT_EQ(first->duration, 2);
  EXPECT_EQ(first->stall, 5u);
  for (int i = 1; i < 4; i++) {
    const auto* next = std::get_if<DurationBelowStall>(&result.diagnostics[i]);
    ASSERT_NE(next, nullptr);
    EXPECT_EQ(next->pc.address, i % 2 ? 0x108u : 0x104u);
  }
  EXPECT_FALSE(result.duration_check_failure);
  expect_records_kept(result);
}

TEST(WaveAnalysisDurationCheckTest, ErrorCountsRecordsAndKeepsTheFirst) {
  auto result = analyze_with_duration_check(DurationCheck::Error);

  EXPECT_TRUE(result.diagnostics.empty());
  ASSERT_TRUE(result.duration_check_failure);
  EXPECT_EQ(result.duration_check_failure->records, 4u);
  const auto& first = result.duration_check_failure->first;
  EXPECT_EQ(first.pc.address, 0x104u);
  EXPECT_EQ(first.pc.code_object_id, kCodeObject);
  EXPECT_EQ(first.duration, 2);
  EXPECT_EQ(first.stall, 5u);
}

}  // namespace
}  // namespace my_rocperf_tool
