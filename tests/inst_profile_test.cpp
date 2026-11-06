#include "my_rocperf_tool/inst_profile.h"
#include "gtest/gtest.h"
#include "synthetic_wave.h"

#include <array>
#include <tuple>

namespace my_rocperf_tool {
namespace {

const InstRow& row_at(const InstGroup& g, uint64_t addr,
                      uint32_t occurrence = 0) {
  for (const auto& row : g.rows)
    if (row.pc.address == addr && row.occurrence == occurrence) return row;
  ADD_FAILURE() << "no row at " << addr << " occurrence " << occurrence;
  static const InstRow missing{};
  return missing;
}

TEST(InstProfileTest, StaticRowsSumEachPcOverWaves) {
  KindTable table;
  InstProfile profile(InstProfileKind::Static, AggregateScope::All, false);
  for (int i = 0; i < 2; i++) {
    WaveBuilder b;
    b.inst(0xf0, kSalu)
        .inst(0x100, kValu)
        .inst(0x104, kValu)
        .inst(0x110, kSalu);
    EXPECT_TRUE(profile.add_wave(b.wave(), i, table.classifier()).empty());
  }
  auto groups = profile.groups();
  ASSERT_EQ(groups.size(), 1u);
  const auto& g = groups[0];
  ASSERT_EQ(g.rows.size(), 4u);
  const std::array<uint64_t, 4> addresses{0xf0, 0x100, 0x104, 0x110};
  for (size_t i = 0; i < 4; i++) {
    const auto& row = g.rows[i];
    EXPECT_EQ(row.pc.address, addresses[i]);
    EXPECT_TRUE(row.text.empty());
    EXPECT_EQ(row.sums.n, 2u);
    EXPECT_EQ(row.sums.exec_waves, 2u);
    EXPECT_EQ(row.sums.dur, 8u);
    EXPECT_EQ(row.sums.stall, 2u);
    EXPECT_EQ(row.sums.issue, 6u);
  }
  EXPECT_EQ(row_at(g, 0xf0).sums.idle, 0u);
  EXPECT_EQ(row_at(g, 0x100).sums.idle, 12u);
  EXPECT_EQ(row_at(g, 0xf0).sums.rel_issue, 0u);
  EXPECT_EQ(row_at(g, 0x100).sums.rel_issue, 20u);
  EXPECT_EQ(row_at(g, 0x104).sums.rel_issue, 40u);
  EXPECT_EQ(row_at(g, 0x110).sums.rel_issue, 60u);
  EXPECT_EQ(g.summary.waves, 2u);
  EXPECT_DOUBLE_EQ(g.summary.avg_records, 4);
  EXPECT_DOUBLE_EQ(g.summary.avg_dur, 16);
  EXPECT_DOUBLE_EQ(g.summary.avg_stall, 4);
  EXPECT_DOUBLE_EQ(g.summary.avg_idle, 18);
}

TEST(InstProfileTest, RelIsAnchoredAtTheFirstRecordAndIdleAtBeginTime) {
  KindTable table;
  WaveBuilder b;
  b.wait(100).inst(0xf0, kSalu).inst(0x100, kValu);
  InstProfile profile(InstProfileKind::Static, AggregateScope::All, false);
  profile.add_wave(b.wave(), 0, table.classifier());
  auto groups = profile.groups();
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(row_at(groups[0], 0xf0).sums.idle, 100u);
  EXPECT_EQ(row_at(groups[0], 0xf0).sums.rel_issue, 0u);
  EXPECT_EQ(row_at(groups[0], 0x100).sums.idle, 6u);
  EXPECT_EQ(row_at(groups[0], 0x100).sums.rel_issue, 10u);
}

WaveBuilder shadow_wave() {
  WaveBuilder b;
  b.at(0xf0, kValu, 0, 4).at(0x100, kValu, 10, 4).at(0x104, kValu, 20, 4);
  return b;
}

TEST(InstProfileTest, MfmaShadowRunsAcrossTheWholeWave) {
  KindTable table{{{0xf0, InstKind::Mfma}}};
  auto b = shadow_wave();
  InstProfile profile(InstProfileKind::Static, AggregateScope::All, true);
  EXPECT_TRUE(profile.add_wave(b.wave(), 0, table.classifier()).empty());
  auto groups = profile.groups();
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(row_at(groups[0], 0xf0).sums.bubble, 0u);
  EXPECT_EQ(row_at(groups[0], 0x100).sums.bubble, 0u);
  EXPECT_EQ(row_at(groups[0], 0x104).sums.bubble, 4u);
  EXPECT_DOUBLE_EQ(groups[0].summary.avg_bubble, 4);
}

TEST(InstProfileTest, WithoutMfmaCoexecNoBubblesButMfmasCount) {
  KindTable table{{{0xf0, InstKind::Mfma}}};
  auto b = shadow_wave();
  InstProfile profile(InstProfileKind::Static, AggregateScope::All, false);
  EXPECT_TRUE(profile.add_wave(b.wave(), 0, table.classifier()).empty());
  auto groups = profile.groups();
  ASSERT_EQ(groups.size(), 1u);
  for (const auto& r : groups[0].rows) EXPECT_EQ(r.sums.bubble, 0u);
  EXPECT_EQ(row_at(groups[0], 0xf0).kind, InstKind::Mfma);
  EXPECT_DOUBLE_EQ(groups[0].summary.avg_mfma, 1);
  EXPECT_DOUBLE_EQ(groups[0].summary.mfma_util, 16.0 / 12.0);
}

TEST(InstProfileTest, SkipsNullPcRecords) {
  KindTable table;
  WaveBuilder b;
  b.null_pc().inst(0xf0, kSalu).null_pc().inst(0x100, kValu);
  InstProfile profile(InstProfileKind::Static, AggregateScope::All, false);
  profile.add_wave(b.wave(), 0, table.classifier());
  auto groups = profile.groups();
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(groups[0].rows.size(), 2u);
  EXPECT_DOUBLE_EQ(groups[0].summary.avg_records, 2);
  EXPECT_EQ(row_at(groups[0], 0xf0).sums.rel_issue, 0u);
  EXPECT_EQ(row_at(groups[0], 0xf0).sums.stall, 1u);
  EXPECT_EQ(row_at(groups[0], 0x100).sums.idle, 6u);
}

TEST(InstProfileTest, IssueIsZeroWhereStallExceedsDuration) {
  KindTable table;
  WaveBuilder b;
  b.at(0xf0, kValu, 0, 2, 5).at(0x100, kValu, 10, 4, 1);
  InstProfile profile(InstProfileKind::Static, AggregateScope::All, false);
  profile.add_wave(b.wave(), 0, table.classifier());
  auto groups = profile.groups();
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(row_at(groups[0], 0xf0).sums.issue, 0u);
  EXPECT_EQ(row_at(groups[0], 0x100).sums.rel_issue, 6u);
}

void branching_waves(InstProfile& profile, KindTable& table) {
  WaveBuilder a, b;
  a.inst(0x100, kValu).inst(0x104, kValu).inst(0x100, kValu).inst(0x108, kValu);
  b.inst(0x100, kValu).inst(0x108, kValu);
  profile.add_wave(a.wave(), 0, table.classifier());
  profile.add_wave(b.wave(), 1, table.classifier());
}

TEST(InstProfileTest, DynamicRowsKeyByPcAndOccurrence) {
  KindTable table;
  InstProfile profile(InstProfileKind::Dynamic, AggregateScope::All, false);
  branching_waves(profile, table);
  auto groups = profile.groups();
  ASSERT_EQ(groups.size(), 1u);
  const auto& rows = groups[0].rows;
  ASSERT_EQ(rows.size(), 4u);
  const std::array<std::tuple<uint64_t, uint32_t, uint64_t>, 4> expected{
      {{0x100, 1, 2}, {0x104, 1, 1}, {0x100, 2, 1}, {0x108, 1, 2}}};
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(
        std::make_tuple(rows[i].pc.address, rows[i].occurrence, rows[i].sums.n),
        expected[i]);
    EXPECT_EQ(rows[i].sums.exec_waves, rows[i].sums.n);
  }
  EXPECT_EQ(row_at(groups[0], 0x100, 2).sums.rel_issue, 20u);
  EXPECT_EQ(row_at(groups[0], 0x108, 1).sums.rel_issue, 40u);
}

TEST(InstProfileTest, StaticRowsCountEveryExecution) {
  KindTable table;
  InstProfile profile(InstProfileKind::Static, AggregateScope::All, false);
  branching_waves(profile, table);
  auto groups = profile.groups();
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(groups[0].rows.size(), 3u);
  EXPECT_EQ(row_at(groups[0], 0x100).sums.n, 3u);
  EXPECT_EQ(row_at(groups[0], 0x100).sums.exec_waves, 2u);
  EXPECT_EQ(row_at(groups[0], 0x104).sums.n, 1u);
  EXPECT_EQ(row_at(groups[0], 0x104).sums.exec_waves, 1u);
  EXPECT_EQ(row_at(groups[0], 0x108).sums.n, 2u);
  EXPECT_EQ(row_at(groups[0], 0x108).sums.exec_waves, 2u);
  for (const auto& row : groups[0].rows) EXPECT_EQ(row.occurrence, 0u);
}

TEST(InstProfileTest, GroupsFollowTheScope) {
  for (auto kind : {InstProfileKind::Static, InstProfileKind::Dynamic})
    for (auto scope : {AggregateScope::Wave, AggregateScope::SimdSlot,
                       AggregateScope::All}) {
      KindTable table;
      InstProfile profile(kind, scope, false);
      std::array<WaveBuilder, 4> waves{WaveBuilder(0, 1), WaveBuilder(1, 0),
                                       WaveBuilder(0, 1), WaveBuilder(0, 0)};
      for (size_t i = 0; i < 4; i++) {
        waves[i].inst(0xf0, kValu);
        profile.add_wave(waves[i].wave(), i, table.classifier());
      }
      auto groups = profile.groups();
      if (scope == AggregateScope::Wave) {
        ASSERT_EQ(groups.size(), 4u);
        const std::array<uint64_t, 4> order{3, 0, 2, 1};
        for (size_t i = 0; i < 4; i++) {
          EXPECT_EQ(groups[i].wave_index, order[i]);
          EXPECT_EQ(groups[i].summary.waves, 1u);
        }
      } else if (scope == AggregateScope::SimdSlot) {
        ASSERT_EQ(groups.size(), 3u);
        EXPECT_TRUE((groups[0].slot == WaveSlot{0, 0, 0}));
        EXPECT_TRUE((groups[1].slot == WaveSlot{0, 0, 1}));
        EXPECT_TRUE((groups[2].slot == WaveSlot{0, 1, 0}));
        EXPECT_EQ(groups[0].summary.waves, 1u);
        EXPECT_EQ(groups[1].summary.waves, 2u);
        EXPECT_EQ(groups[2].summary.waves, 1u);
        EXPECT_EQ(groups[1].rows[0].sums.n, 2u);
      } else {
        ASSERT_EQ(groups.size(), 1u);
        EXPECT_EQ(groups[0].summary.waves, 4u);
        EXPECT_EQ(groups[0].rows[0].sums.n, 4u);
      }
      for (const auto& g : groups) EXPECT_EQ(g.scope, scope);
    }
}

TEST(InstProfileTest, NoWaveNoGroup) {
  for (auto kind : {InstProfileKind::Static, InstProfileKind::Dynamic})
    for (auto scope :
         {AggregateScope::Wave, AggregateScope::SimdSlot, AggregateScope::All})
      EXPECT_TRUE(InstProfile(kind, scope, false).groups().empty());
}

TEST(InstProfileTest, WaveWithoutAttributedRecordsHasAnEmptyGroup) {
  for (auto kind : {InstProfileKind::Static, InstProfileKind::Dynamic})
    for (auto scope : {AggregateScope::Wave, AggregateScope::SimdSlot,
                       AggregateScope::All}) {
      KindTable table;
      WaveBuilder b;
      b.null_pc();
      InstProfile profile(kind, scope, true);
      profile.add_wave(b.wave(), 0, table.classifier());
      auto groups = profile.groups();
      ASSERT_EQ(groups.size(), 1u);
      EXPECT_TRUE(groups[0].rows.empty());
      const auto& s = groups[0].summary;
      EXPECT_EQ(s.waves, 1u);
      EXPECT_DOUBLE_EQ(s.avg_records, 0);
      EXPECT_DOUBLE_EQ(s.avg_dur, 0);
      EXPECT_DOUBLE_EQ(s.avg_stall, 0);
      EXPECT_DOUBLE_EQ(s.avg_idle, 0);
      EXPECT_DOUBLE_EQ(s.avg_bubble, 0);
      EXPECT_DOUBLE_EQ(s.avg_mfma, 0);
      EXPECT_DOUBLE_EQ(s.mfma_util, 0);
      EXPECT_EQ(table.calls, 0);
    }
}

TEST(InstProfileTest, ReportsMfmaShadowOverlaps) {
  KindTable table{{{0xf0, InstKind::Mfma}, {0x100, InstKind::Mfma}}};
  WaveBuilder b;
  b.at(0xf0, kValu, 0, 4).at(0x100, kValu, 8, 4);
  InstProfile profile(InstProfileKind::Static, AggregateScope::All, true);
  auto overlaps = profile.add_wave(b.wave(), 0, table.classifier());
  ASSERT_EQ(overlaps.size(), 1u);
  EXPECT_EQ(overlaps[0].pc.address, 0x100u);
  EXPECT_EQ(overlaps[0].issue_point, 8);
  EXPECT_EQ(overlaps[0].shadow_end, 16);
}

}  // namespace
}  // namespace my_rocperf_tool
