#include "my_rocperf_tool/loop_profile.h"

#include "my_rocperf_tool/inst_kind.h"
#include "my_rocperf_tool/loop_detection.h"

#include "synthetic_wave.h"

#include "gtest/gtest.h"

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
  // The bubble of both passes, 4 + 2 * 3, over the largest n.
  EXPECT_DOUBLE_EQ(summary.avg_bubble, 5.0);
  EXPECT_EQ(summary.mfma_count, 2u);
  EXPECT_EQ(summary.mfma_time, 2 * kMfmaShadowCycles);
  EXPECT_DOUBLE_EQ(summary.mfma_util, 2.0 * kMfmaShadowCycles / 25.0);
}

// Rows do not follow the record: n, dur, stall, idle and issue count every
// execution in every wave, bubble and rel_issue the passes of the waves in
// which detect_loops found the loop. Only the summary's dur, stall and idle
// averages are the record's own.
TEST(LoopProfileTest, EveryRecordOfALoopGetsTheTraceWideRows) {
  KindTable table;
  LoopProfile profile;
  WaveBuilder twice;
  twice.iteration(/*branch_taken=*/true).iteration(/*branch_taken=*/false);
  WaveBuilder once;
  once.iteration(/*branch_taken=*/false);
  WaveBuilder thrice;
  thrice.iteration(/*branch_taken=*/true)
      .wait(100)
      .iteration(/*branch_taken=*/true)
      .wait(100)
      .iteration(/*branch_taken=*/false);

  auto first = add_wave(profile, twice, table, /*wave_index=*/0);
  EXPECT_TRUE(add_wave(profile, once, table, /*wave_index=*/1).loops.empty());
  auto third = add_wave(profile, thrice, table, /*wave_index=*/2);
  auto body = list_body(kLoop);
  auto first_record = wave_record(first);
  auto third_record = wave_record(third);
  LoopRecord aggregate{kLoop, first_record.stats, 2, std::nullopt};
  aggregate.stats += third_record.stats;

  for (const auto& record : {first_record, third_record, aggregate}) {
    auto metrics = profile.metrics(record, body);
    for (size_t i = 0; i < metrics.rows.size(); i++) {
      EXPECT_EQ(metrics.rows[i].n, 2 + 1 + 3u);
      EXPECT_EQ(metrics.rows[i].rel_issue, (2 + 3) * i * kSpacing);
    }
    EXPECT_DOUBLE_EQ(metrics.summary.avg_idle,
                     static_cast<double>(record.stats.total_idle) /
                         record.stats.iteration_count);
  }
  EXPECT_NE(profile.metrics(first_record, body).summary.avg_idle,
            profile.metrics(third_record, body).summary.avg_idle);
}

}  // namespace
}  // namespace my_rocperf_tool
