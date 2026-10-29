#include "my_rocperf_tool/wave_analysis.h"

#include "my_rocperf_tool/inst_kind.h"
#include "my_rocperf_tool/loop_detection.h"

#include "gtest/gtest.h"

#include "trace_decoder_types.h"

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace my_rocperf_tool {
namespace {

using Category = rocprofiler_thread_trace_decoder_inst_category_t;

constexpr auto kSalu = ROCPROFILER_THREAD_TRACE_DECODER_INST_SALU;
constexpr auto kValu = ROCPROFILER_THREAD_TRACE_DECODER_INST_VALU;
constexpr auto kJump = ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP;
constexpr auto kNext = ROCPROFILER_THREAD_TRACE_DECODER_INST_NEXT;
constexpr auto kMessage = ROCPROFILER_THREAD_TRACE_DECODER_INST_MESSAGE;
constexpr auto kImmed = ROCPROFILER_THREAD_TRACE_DECODER_INST_IMMED;

constexpr uint64_t kCodeObject = 1;

// A 4-instruction loop body closed by a conditional branch back to its
// header, and the instructions either side of it.
constexpr uint64_t kPrologue = 0xf0;
constexpr uint64_t kLoopHeader = 0x100;
constexpr uint64_t kLoopBranch = 0x10c;
constexpr uint64_t kEpilogue = 0x110;
constexpr BackEdge kLoop{kCodeObject, kLoopBranch, kLoopHeader};

// What WaveBuilder::inst records look like: each starts kSpacing cycles after
// the previous one and lasts kDuration cycles, kStall of them stalled.
constexpr int64_t kSpacing = 10;
constexpr int32_t kDuration = 4;
constexpr uint32_t kStall = 1;

rocprofiler_thread_trace_decoder_pc_t pc(uint64_t addr) {
  return {addr, kCodeObject};
}

class WaveBuilder {
 public:
  explicit WaveBuilder(uint8_t simd = 0) : simd_(simd) {}

  WaveBuilder& at(uint64_t addr, Category category, int64_t time,
                  int32_t duration, uint32_t stall = 0) {
    rocprofiler_thread_trace_decoder_inst_t inst{};
    inst.category = category;
    inst.stall = stall;
    inst.duration = duration;
    inst.time = time;
    inst.pc = pc(addr);
    insts_.push_back(inst);
    next_time_ = time + kSpacing;
    return *this;
  }

  WaveBuilder& inst(uint64_t addr, Category category) {
    return at(addr, category, next_time_, kDuration, kStall);
  }

  WaveBuilder& wait(int64_t cycles) {
    next_time_ += cycles;
    return *this;
  }

  // One pass through the loop body. A taken branch is the back-edge.
  WaveBuilder& iteration(bool branch_taken) {
    return inst(kLoopHeader, kValu)
        .inst(kLoopHeader + 4, kValu)
        .inst(kLoopHeader + 8, kSalu)
        .inst(kLoopBranch, branch_taken ? kJump : kNext);
  }

  // The wave points into this builder, which must outlive it.
  rocprofiler_thread_trace_decoder_wave_t wave() {
    rocprofiler_thread_trace_decoder_wave_t wave{};
    wave.simd = simd_;
    wave.instructions_size = insts_.size();
    wave.instructions_array = insts_.data();
    return wave;
  }

 private:
  uint8_t simd_;
  std::vector<rocprofiler_thread_trace_decoder_inst_t> insts_;
  int64_t next_time_ = 0;
};

// Classifies the PCs of kCodeObject it lists, every other PC as Other, and
// counts the calls.
struct KindTable {
  std::map<uint64_t, InstKind> kinds;
  int calls = 0;

  // Refers to this table, which must outlive the classifier.
  InstClassifier classifier() {
    return [this](const rocprofiler_thread_trace_decoder_pc_t& pc) {
      calls++;
      auto it = kinds.find(pc.address);
      return it == kinds.end() ? InstKind::Other : it->second;
    };
  }
};

WaveAnalysisOptions options_with(DurationCheck duration_check) {
  WaveAnalysisOptions options;
  options.duration_check = duration_check;
  return options;
}

// The total `totals` holds for a PC of `edge`'s body, or nullopt if none.
std::optional<uint64_t> total_at(const PerLoopMap& totals, const BackEdge& edge,
                                 uint64_t addr) {
  auto loop = totals.find(edge);
  if (loop == totals.end()) return std::nullopt;
  auto it = loop->second.find(pc(addr));
  if (it == loop->second.end()) return std::nullopt;
  return it->second;
}

TEST(WaveAnalysisTest, CoalescesBarrierTokenPairIntoOneRecord) {
  KindTable table{{{0x100, InstKind::ClassicBarrier}}};
  WaveBuilder builder;
  builder.at(kPrologue, kSalu, /*time=*/0, /*duration=*/4)
      .at(0x100, kMessage, /*time=*/10, /*duration=*/5, /*stall=*/3)
      .at(0x100, kImmed, /*time=*/15, /*duration=*/7, /*stall=*/2)
      .at(0x104, kSalu, /*time=*/30, /*duration=*/4);
  WaveAnalysis analysis(WaveAnalysisOptions{}, table.classifier());

  analysis.add_wave(builder.wave());
  auto result = std::move(analysis).finish();

  EXPECT_EQ(result.counts.insts, 3u);
  auto barrier = result.stats.get_inst_at(pc(0x100));
  ASSERT_EQ(barrier.size(), 1u);
  EXPECT_EQ(barrier[0].category, static_cast<uint32_t>(kMessage));
  EXPECT_EQ(barrier[0].duration, 12);
  EXPECT_EQ(barrier[0].stall, 5u);
  EXPECT_EQ(barrier[0].idle, 6u);
  // Idle after the barrier runs from the end of the merged record.
  auto next = result.stats.get_inst_at(pc(0x104));
  ASSERT_EQ(next.size(), 1u);
  EXPECT_EQ(next[0].idle, 8u);
}

TEST(WaveAnalysisTest, CoalescedBarrierSaturates) {
  constexpr int32_t kDurationMax = std::numeric_limits<int32_t>::max();
  constexpr uint32_t kStallMax = (1u << 24) - 1;
  KindTable table{{{0x100, InstKind::ClassicBarrier}}};
  WaveBuilder builder;
  builder.at(0x100, kMessage, /*time=*/0, kDurationMax - 1, kStallMax - 1)
      .at(0x100, kImmed, /*time=*/kDurationMax - 1, /*duration=*/5,
          /*stall=*/5);
  WaveAnalysis analysis(WaveAnalysisOptions{}, table.classifier());

  analysis.add_wave(builder.wave());
  auto result = std::move(analysis).finish();

  auto barrier = result.stats.get_inst_at(pc(0x100));
  ASSERT_EQ(barrier.size(), 1u);
  EXPECT_EQ(barrier[0].duration, kDurationMax);
  EXPECT_EQ(barrier[0].stall, kStallMax);
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
  EXPECT_TRUE(result.stats.get_inst_at(pc(kOtherHeader)).empty());
  EXPECT_TRUE(result.stats.get_inst_at(pc(kOtherHeader + 4)).empty());
  EXPECT_EQ(result.stats.size(), 4u);
  ASSERT_EQ(result.wave_loops.size(), 1u);
  EXPECT_EQ(result.wave_loops[0].simd, 1);
  EXPECT_FALSE(result.rel_issue_totals.count(kOtherLoop));
  EXPECT_FALSE(result.bubble_totals.count(kOtherLoop));
  EXPECT_TRUE(result.diagnostics.empty());
  EXPECT_FALSE(result.duration_check_failure);
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

  EXPECT_EQ(total_at(result.bubble_totals, kLoop, kValuInLoop), std::nullopt);
  EXPECT_EQ(total_at(result.bubble_totals, kLoop, kLoopBranch), 2u);
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

  EXPECT_EQ(total_at(result.bubble_totals, kLoop, kLoopHeader + 4),
            std::nullopt);
  EXPECT_EQ(total_at(result.bubble_totals, kLoop, kLoopHeader + 8), 2 * 4u);
  EXPECT_EQ(total_at(result.bubble_totals, kLoop, kLoopBranch), 2 * 3u);
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
  for (uint64_t i = 0; i < 4; i++)
    EXPECT_EQ(total_at(result.rel_issue_totals, kLoop, kLoopHeader + 4 * i),
              3 * i * kSpacing);
  EXPECT_EQ(total_at(result.rel_issue_totals, kLoop, kPrologue), std::nullopt);
  EXPECT_EQ(total_at(result.rel_issue_totals, kLoop, kEpilogue), std::nullopt);
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
  for (uint64_t i = 0; i < 4; i++) {
    auto records = result.stats.get_inst_at(pc(kLoopHeader + 4 * i));
    EXPECT_EQ(records.size(), 3u);
  }
  // Classified only for the per-loop MFMA analysis, which is off.
  EXPECT_EQ(table.calls, 0);
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
  // The merged record issues after both tokens' stalls: 10 + 3 + 2 cycles
  // after the header.
  EXPECT_EQ(total_at(result.rel_issue_totals, kLoop, kBarrier), 3 * 15u);
  // No MFMA shadow covers any of its 5 + 7 cycles.
  EXPECT_EQ(total_at(result.bubble_totals, kLoop, kBarrier), 3 * 12u);
}

// Records 0x104 and 0x108 stall longer than they last.
WaveAnalysisResult analyze_with_duration_check(DurationCheck duration_check) {
  KindTable table;
  WaveBuilder builder;
  builder.at(0x100, kValu, /*time=*/0, /*duration=*/4, /*stall=*/1)
      .at(0x104, kValu, /*time=*/10, /*duration=*/2, /*stall=*/5)
      .at(0x108, kValu, /*time=*/20, /*duration=*/1, /*stall=*/3);
  WaveAnalysis analysis(options_with(duration_check), table.classifier());
  analysis.add_wave(builder.wave());
  return std::move(analysis).finish();
}

TEST(WaveAnalysisDurationCheckTest, OffChecksNothing) {
  auto result = analyze_with_duration_check(DurationCheck::Off);

  EXPECT_TRUE(result.diagnostics.empty());
  EXPECT_FALSE(result.duration_check_failure);
  EXPECT_EQ(result.stats.size(), 3u);
}

TEST(WaveAnalysisDurationCheckTest, WarnReportsEachRecordAndKeepsThem) {
  auto result = analyze_with_duration_check(DurationCheck::Warn);

  ASSERT_EQ(result.diagnostics.size(), 2u);
  const auto* first = std::get_if<DurationBelowStall>(&result.diagnostics[0]);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first->pc.address, 0x104u);
  EXPECT_EQ(first->pc.code_object_id, kCodeObject);
  EXPECT_EQ(first->duration, 2);
  EXPECT_EQ(first->stall, 5u);
  const auto* second = std::get_if<DurationBelowStall>(&result.diagnostics[1]);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(second->pc.address, 0x108u);
  EXPECT_FALSE(result.duration_check_failure);
  EXPECT_EQ(result.stats.size(), 3u);
}

TEST(WaveAnalysisDurationCheckTest, ErrorCountsRecordsAndKeepsTheFirst) {
  auto result = analyze_with_duration_check(DurationCheck::Error);

  EXPECT_TRUE(result.diagnostics.empty());
  ASSERT_TRUE(result.duration_check_failure);
  EXPECT_EQ(result.duration_check_failure->records, 2u);
  const auto& first = result.duration_check_failure->first;
  EXPECT_EQ(first.pc.address, 0x104u);
  EXPECT_EQ(first.pc.code_object_id, kCodeObject);
  EXPECT_EQ(first.duration, 2);
  EXPECT_EQ(first.stall, 5u);
}

}  // namespace
}  // namespace my_rocperf_tool
