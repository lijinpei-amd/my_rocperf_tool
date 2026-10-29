#include "my_rocperf_tool/loop_profile.h"

#include "my_rocperf_tool/inst_kind.h"
#include "my_rocperf_tool/loop_detection.h"

#include "synthetic_wave.h"

#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace my_rocperf_tool {
namespace {

// Adds `builder`'s wave to `profile` as the trace's wave_index'th wave, with
// the loops detect_loops finds in it, and returns those loops.
WaveLoopInfo add_wave(LoopProfile& profile, WaveBuilder& builder,
                      KindTable& table, uint64_t wave_index = 0) {
  auto wave = builder.wave();
  auto loops = detect_loops(wave);
  loops.wave_index = wave_index;
  EXPECT_TRUE(profile.add_wave(wave, loops, table.classifier()).empty());
  return loops;
}

// The record of the one loop of a wave.
LoopRecord wave_record(const WaveLoopInfo& loops) {
  EXPECT_EQ(loops.loops.size(), 1u);
  const auto& loop = loops.loops.at(0);
  return {loop.back_edge, loop.stats, 1, loops.wave_index};
}

TEST(LoopProfileTest, RowsSumEachExecutionOfTheBody) {
  // The waves never run the body's third instruction.
  constexpr uint64_t kSkipped = kLoopHeader + 8;
  KindTable table;
  WaveBuilder builder;
  builder.inst(kPrologue, kSalu);
  for (int pass = 0; pass < 3; pass++)
    builder.inst(kLoopHeader, kValu)
        .inst(kLoopHeader + 4, kValu)
        .inst(kLoopBranch, pass < 2 ? kJump : kNext);
  builder.inst(kEpilogue, kSalu);
  LoopProfile profile;

  auto loops = add_wave(profile, builder, table);
  auto body = list_body(kLoop, {{kLoopHeader + 4, InstKind::Mfma}});
  auto metrics = profile.metrics(wave_record(loops), body);

  ASSERT_EQ(metrics.rows.size(), body.size());
  for (size_t i = 0; i < body.size(); i++) {
    EXPECT_EQ(metrics.rows[i].vaddr, body[i].vaddr);
    EXPECT_EQ(metrics.rows[i].text, body[i].text);
    EXPECT_EQ(metrics.rows[i].kind, body[i].kind);
  }
  for (uint64_t addr : {kLoopHeader, kLoopHeader + 4, kLoopBranch}) {
    auto row = row_at(metrics, addr);
    EXPECT_TRUE(row.has_stats);
    EXPECT_EQ(row.n, 3u);
    EXPECT_EQ(row.dur, 3u * kDuration);
    EXPECT_EQ(row.stall, 3u * kStall);
    EXPECT_EQ(row.idle, 3u * (kSpacing - kDuration));
    EXPECT_EQ(row.issue, 3u * (kDuration - kStall));
  }
  auto skipped = row_at(metrics, kSkipped);
  EXPECT_FALSE(skipped.has_stats);
  EXPECT_EQ(skipped.n, 0u);
  EXPECT_EQ(skipped.dur, 0u);
  EXPECT_EQ(skipped.idle, 0u);
  EXPECT_EQ(skipped.rel_issue, 0u);
}

TEST(LoopProfileTest, IssueIsZeroWhereStallExceedsDuration) {
  KindTable table;
  WaveBuilder builder;
  for (int pass = 0; pass < 2; pass++) {
    const int64_t start = 40 * pass;
    builder.at(kLoopHeader, kValu, start, /*duration=*/4, /*stall=*/1)
        .at(kLoopHeader + 4, kValu, start + 10, /*duration=*/2, /*stall=*/5)
        .at(kLoopHeader + 8, kValu, start + 20, /*duration=*/3, /*stall=*/3)
        .at(kLoopBranch, pass == 0 ? kJump : kNext, start + 30,
            /*duration=*/2);
  }
  LoopProfile profile;

  auto loops = add_wave(profile, builder, table);
  auto body = list_body(kLoop);
  auto metrics = profile.metrics(wave_record(loops), body);

  EXPECT_EQ(row_at(metrics, kLoopHeader).issue, 2 * 3u);
  // Its duration and stall still count as recorded.
  auto below = row_at(metrics, kLoopHeader + 4);
  EXPECT_EQ(below.dur, 2 * 2u);
  EXPECT_EQ(below.stall, 2 * 5u);
  EXPECT_EQ(below.issue, 0u);
  EXPECT_EQ(row_at(metrics, kLoopHeader + 8).issue, 0u);
  EXPECT_EQ(row_at(metrics, kLoopBranch).issue, 2 * 2u);
}

TEST(LoopProfileTest, NoBubblesWithoutMfmaCoexec) {
  KindTable table{{{kLoopHeader, InstKind::Mfma}}};
  WaveBuilder builder;
  builder.iteration(/*branch_taken=*/true).iteration(/*branch_taken=*/false);
  LoopProfile profile;

  auto loops = add_wave(profile, builder, table);
  auto body = list_body(kLoop, table.kinds);
  auto metrics = profile.metrics(wave_record(loops), body);

  for (const auto& row : metrics.rows) EXPECT_EQ(row.bubble, 0u);
  EXPECT_EQ(row_at(metrics, kLoopBranch).rel_issue, 2 * 3 * kSpacing);
  EXPECT_EQ(metrics.summary.avg_bubble, 0.0);
  EXPECT_EQ(table.calls, 0);
}

TEST(LoopProfileTest, SummaryAveragesTheRecordsIterations) {
  KindTable table{{{kLoopHeader, InstKind::Mfma}}};
  // The MFMA's shadow covers each pass's first 16 cycles. The first pass's
  // second record has its last 4 cycles uncovered; each branch all 3 of its.
  WaveBuilder builder;
  builder.at(kLoopHeader, kValu, /*time=*/0, /*duration=*/4)
      .at(kLoopHeader + 4, kValu, /*time=*/14, /*duration=*/6)
      .at(kLoopBranch, kJump, /*time=*/30, /*duration=*/3)
      .at(kLoopHeader, kValu, /*time=*/100, /*duration=*/4)
      .at(kLoopBranch, kNext, /*time=*/130, /*duration=*/3);
  LoopProfile profile(/*mfma_coexec=*/true);
  add_wave(profile, builder, table);
  // The listing's kinds count, executed or not.
  auto body = list_body(kLoop, {{kLoopHeader, InstKind::Mfma},
                                {kLoopHeader + 8, InstKind::Mfma}});
  LoopRecord record{kLoop,
                    {/*iteration_count=*/4, /*total_inst_count=*/12,
                     /*total_duration=*/100, /*total_stall=*/20,
                     /*total_idle=*/40},
                    1,
                    0};

  auto summary = profile.metrics(record, body).summary;

  EXPECT_DOUBLE_EQ(summary.avg_dur, 25.0);
  EXPECT_DOUBLE_EQ(summary.avg_stall, 5.0);
  EXPECT_DOUBLE_EQ(summary.avg_idle, 10.0);
  // The bubble of both passes, 4 + 2 * 3, per iteration of the record.
  EXPECT_DOUBLE_EQ(summary.avg_bubble, 2.5);
  EXPECT_EQ(summary.mfma_count, 2u);
  EXPECT_EQ(summary.mfma_time, 2 * kMfmaShadowCycles);
  EXPECT_DOUBLE_EQ(summary.mfma_util, 2.0 * kMfmaShadowCycles / 25.0);
}

// The scope tests' loop: an MFMA header, whose shadow covers what ends in the
// 16 cycles after it issues.
KindTable mfma_header_table() {
  return KindTable{{{kLoopHeader, InstKind::Mfma}}};
}

// Two passes of WaveBuilder::inst records, 10 cycles apart. The shadow ends
// 17 cycles into each pass, so all 4 cycles of each of the last two records
// are bubble.
WaveBuilder two_passes() {
  WaveBuilder builder;
  builder.iteration(/*branch_taken=*/true).iteration(/*branch_taken=*/false);
  return builder;
}

// Three passes 100 cycles apart, timed unlike two_passes'. The shadow ends 18
// cycles into each pass: 4 of the third record's 10 cycles are bubble, and
// both of the branch's.
WaveBuilder three_passes() {
  WaveBuilder builder;
  for (int pass = 0; pass < 3; pass++) {
    const int64_t start = 100 * pass;
    builder.at(kLoopHeader, kValu, start, /*duration=*/4, /*stall=*/2)
        .at(kLoopHeader + 4, kValu, start + 6, /*duration=*/4)
        .at(kLoopHeader + 8, kSalu, start + 12, /*duration=*/10, /*stall=*/3)
        .at(kLoopBranch, pass < 2 ? kJump : kNext, start + 30,
            /*duration=*/2);
  }
  return builder;
}

// What a body instruction's row sums to over one record's executions.
struct RowSums {
  uint64_t n, dur, stall, idle, issue, bubble, rel_issue;
};
// kLoop's rows, in body order.
using BodySums = std::array<RowSums, 4>;

constexpr BodySums kTwoPassSums{{
    {2, 8, 2, 6, 6, 0, 0},
    {2, 8, 2, 12, 6, 0, 20},
    {2, 8, 2, 12, 6, 8, 40},
    {2, 8, 2, 12, 6, 8, 60},
}};
constexpr BodySums kThreePassSums{{
    {3, 12, 6, 136, 6, 0, 0},
    {3, 12, 0, 6, 12, 0, 12},
    {3, 30, 9, 6, 21, 12, 39},
    {3, 6, 0, 24, 6, 6, 84},
}};
// The two waves' sums added.
constexpr BodySums kBothWavesSums{{
    {5, 20, 8, 142, 12, 0, 0},
    {5, 20, 2, 18, 18, 0, 32},
    {5, 38, 11, 18, 27, 20, 79},
    {5, 14, 2, 36, 12, 14, 144},
}};

void expect_rows(const LoopRecordMetrics& metrics, const BodySums& sums) {
  ASSERT_EQ(metrics.rows.size(), sums.size());
  for (size_t i = 0; i < sums.size(); i++) {
    SCOPED_TRACE(i);
    const auto& row = metrics.rows[i];
    EXPECT_TRUE(row.has_stats);
    EXPECT_EQ(row.n, sums[i].n);
    EXPECT_EQ(row.dur, sums[i].dur);
    EXPECT_EQ(row.stall, sums[i].stall);
    EXPECT_EQ(row.idle, sums[i].idle);
    EXPECT_EQ(row.issue, sums[i].issue);
    EXPECT_EQ(row.bubble, sums[i].bubble);
    EXPECT_EQ(row.rel_issue, sums[i].rel_issue);
  }
}

// A record's rows count the executions detect_loops summed into its
// LoopStats.
void expect_rows_sum_to_stats(const LoopRecordMetrics& metrics,
                              const LoopStats& stats) {
  LoopStats sums;
  for (const auto& row : metrics.rows) {
    sums.total_inst_count += row.n;
    sums.total_duration += row.dur;
    sums.total_stall += row.stall;
    sums.total_idle += row.idle;
  }
  EXPECT_EQ(sums.total_inst_count, stats.total_inst_count);
  EXPECT_EQ(sums.total_duration, stats.total_duration);
  EXPECT_EQ(sums.total_stall, stats.total_stall);
  EXPECT_EQ(sums.total_idle, stats.total_idle);
}

// Each record's numbers count its own executions: a wave's, its own; the
// aggregate's, those of the waves in which detect_loops found the loop, and
// not those of a wave that ran the body once.
TEST(LoopProfileTest, EachRecordCountsItsOwnExecutions) {
  auto table = mfma_header_table();
  LoopProfile profile(/*mfma_coexec=*/true);
  auto twice = two_passes();
  WaveBuilder once;
  once.iteration(/*branch_taken=*/false);
  auto thrice = three_passes();

  auto first = wave_record(add_wave(profile, twice, table, /*wave_index=*/0));
  EXPECT_TRUE(add_wave(profile, once, table, /*wave_index=*/1).loops.empty());
  auto third = wave_record(add_wave(profile, thrice, table, /*wave_index=*/2));
  LoopRecord aggregate{kLoop, first.stats, 2, std::nullopt};
  aggregate.stats += third.stats;
  auto body = list_body(kLoop, table.kinds);

  auto first_metrics = profile.metrics(first, body);
  auto third_metrics = profile.metrics(third, body);
  auto aggregate_metrics = profile.metrics(aggregate, body);

  expect_rows(first_metrics, kTwoPassSums);
  expect_rows(third_metrics, kThreePassSums);
  expect_rows(aggregate_metrics, kBothWavesSums);
  expect_rows_sum_to_stats(first_metrics, first.stats);
  expect_rows_sum_to_stats(third_metrics, third.stats);
  expect_rows_sum_to_stats(aggregate_metrics, aggregate.stats);
  // Per execution, the renderers' avg_cycle and avg_bubble divide within the
  // record.
  auto branch_average = [](const LoopRecordMetrics& metrics,
                           uint64_t LoopInstRow::* sum) {
    auto row = row_at(metrics, kLoopBranch);
    return static_cast<double>(row.*sum) / row.n;
  };
  EXPECT_DOUBLE_EQ(branch_average(first_metrics, &LoopInstRow::rel_issue),
                   30.0);
  EXPECT_DOUBLE_EQ(branch_average(third_metrics, &LoopInstRow::rel_issue),
                   28.0);
  EXPECT_DOUBLE_EQ(branch_average(aggregate_metrics, &LoopInstRow::rel_issue),
                   28.8);
  EXPECT_DOUBLE_EQ(branch_average(first_metrics, &LoopInstRow::bubble), 4.0);
  EXPECT_DOUBLE_EQ(branch_average(third_metrics, &LoopInstRow::bubble), 2.0);
  EXPECT_DOUBLE_EQ(branch_average(aggregate_metrics, &LoopInstRow::bubble),
                   2.8);
  // Per iteration, the record's bubble over its iterations: 16 over 2, 18
  // over 3, and 34 over 5.
  EXPECT_DOUBLE_EQ(first_metrics.summary.avg_bubble, 8.0);
  EXPECT_DOUBLE_EQ(third_metrics.summary.avg_bubble, 6.0);
  EXPECT_DOUBLE_EQ(aggregate_metrics.summary.avg_bubble, 6.8);
}

TEST(LoopProfileTest, WithoutWaveRowsAWavesRecordHasOnlyItsSummary) {
  auto table = mfma_header_table();
  LoopProfile profile(/*mfma_coexec=*/true, /*wave_rows=*/false);
  auto twice = two_passes();
  auto thrice = three_passes();

  auto first = wave_record(add_wave(profile, twice, table, /*wave_index=*/0));
  auto second = wave_record(add_wave(profile, thrice, table, /*wave_index=*/1));
  LoopRecord aggregate{kLoop, first.stats, 2, std::nullopt};
  aggregate.stats += second.stats;
  auto body = list_body(kLoop, table.kinds);

  auto first_metrics = profile.metrics(first, body);
  auto second_metrics = profile.metrics(second, body);
  EXPECT_TRUE(first_metrics.rows.empty());
  EXPECT_TRUE(second_metrics.rows.empty());
  EXPECT_DOUBLE_EQ(first_metrics.summary.avg_bubble, 8.0);
  EXPECT_DOUBLE_EQ(second_metrics.summary.avg_bubble, 6.0);
  EXPECT_EQ(first_metrics.summary.mfma_count, 1u);
  auto aggregate_metrics = profile.metrics(aggregate, body);
  expect_rows(aggregate_metrics, kBothWavesSums);
  EXPECT_DOUBLE_EQ(aggregate_metrics.summary.avg_bubble, 6.8);
}

// An outer loop's body holds an inner loop's, whose PCs run more often than
// the outer loop iterates. The outer loop's bubble is still per outer
// iteration.
TEST(LoopProfileTest, OuterLoopsBubbleIsPerOuterIteration) {
  constexpr uint64_t kOuterHeader = 0x200;
  constexpr uint64_t kInnerHeader = 0x204;
  constexpr uint64_t kInnerBranch = 0x20c;
  constexpr uint64_t kOuterBranch = 0x210;
  constexpr BackEdge kOuter{kCodeObject, kOuterBranch, kOuterHeader};
  constexpr BackEdge kInner{kCodeObject, kInnerBranch, kInnerHeader};
  // No MFMA casts a shadow, so every record's whole duration is bubble.
  KindTable table;
  WaveBuilder builder;
  for (int outer = 0; outer < 2; outer++) {
    builder.inst(kOuterHeader, kValu);
    for (int inner = 0; inner < 3; inner++)
      builder.inst(kInnerHeader, kValu)
          .inst(kInnerHeader + 4, kValu)
          .inst(kInnerBranch, inner < 2 ? kJump : kNext);
    builder.inst(kOuterBranch, outer < 1 ? kJump : kNext);
  }
  LoopProfile profile(/*mfma_coexec=*/true);

  auto loops = add_wave(profile, builder, table);

  ASSERT_EQ(loops.loops.size(), 2u);
  for (const auto& loop : loops.loops) {
    ASSERT_TRUE(loop.back_edge == kOuter || loop.back_edge == kInner);
    for (const auto& record :
         {LoopRecord{loop.back_edge, loop.stats, 1, loops.wave_index},
          LoopRecord{loop.back_edge, loop.stats, 1, std::nullopt}}) {
      auto metrics = profile.metrics(record, list_body(loop.back_edge));
      EXPECT_DOUBLE_EQ(metrics.summary.avg_bubble, metrics.summary.avg_dur);
      if (loop.back_edge == kOuter) {
        EXPECT_EQ(loop.stats.iteration_count, 2u);
        EXPECT_EQ(row_at(metrics, kOuterHeader).n, 2u);
        EXPECT_EQ(row_at(metrics, kInnerHeader).n, 2 * 3u);
        // 2 passes of 11 records, over 2 iterations rather than the inner
        // body's n of 6.
        EXPECT_DOUBLE_EQ(metrics.summary.avg_bubble, 11.0 * kDuration);
      }
    }
  }
}

}  // namespace
}  // namespace my_rocperf_tool
