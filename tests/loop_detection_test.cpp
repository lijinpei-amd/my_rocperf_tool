#include "my_rocperf_tool/loop_detection.h"

#include "gtest/gtest.h"

#include "trace_decoder_types.h"

#include <cstdint>
#include <vector>

namespace my_rocperf_tool {
namespace {

constexpr auto kSalu = ROCPROFILER_THREAD_TRACE_DECODER_INST_SALU;
constexpr auto kValu = ROCPROFILER_THREAD_TRACE_DECODER_INST_VALU;
constexpr auto kJump = ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP;
constexpr auto kNext = ROCPROFILER_THREAD_TRACE_DECODER_INST_NEXT;

constexpr uint64_t kCodeObject = 1;

// Each attributed record starts kSpacing cycles after the previous one and
// lasts kDuration cycles, kStall of them stalled, so every record but the first
// is preceded by kIdle idle cycles.
constexpr int64_t kSpacing = 10;
constexpr int32_t kDuration = 4;
constexpr uint32_t kStall = 1;
constexpr uint64_t kIdle = kSpacing - kDuration;

// A straight-line prologue, a 4-instruction loop body closed by a conditional
// branch back to its header, and the instruction after the loop.
constexpr uint64_t kPrologue = 0xf0;
constexpr uint64_t kLoopHeader = 0x100;
constexpr uint64_t kLoopBranch = 0x10c;
constexpr uint64_t kEpilogue = 0x110;
constexpr uint64_t kBodySize = 4;

class WaveBuilder {
 public:
  WaveBuilder& inst(uint64_t addr,
                    rocprofiler_thread_trace_decoder_inst_category_t category) {
    rocprofiler_thread_trace_decoder_inst_t inst{};
    inst.category = category;
    inst.stall = kStall;
    inst.duration = kDuration;
    inst.time = next_time_;
    inst.pc = {addr, kCodeObject};
    insts_.push_back(inst);
    next_time_ += kSpacing;
    return *this;
  }

  // One pass through the loop body. A taken branch is the back-edge.
  WaveBuilder& iteration(bool branch_taken) {
    return inst(kLoopHeader, kValu)
        .inst(kLoopHeader + 4, kValu)
        .inst(kLoopHeader + 8, kSalu)
        .inst(kLoopBranch, branch_taken ? kJump : kNext);
  }

  // A record the decoder could not attribute to a code object. It overlaps
  // the next record's start, so timing it would zero that record's idle.
  WaveBuilder& null_pc() {
    rocprofiler_thread_trace_decoder_inst_t inst{};
    inst.category = kValu;
    inst.duration = 1000;
    inst.time = next_time_ - kSpacing / 2;
    insts_.push_back(inst);
    return *this;
  }

  // The wave points into this builder, which must outlive it.
  rocprofiler_thread_trace_decoder_wave_t wave() {
    rocprofiler_thread_trace_decoder_wave_t wave{};
    wave.cu = 1;
    wave.simd = 2;
    wave.wave_id = 3;
    wave.begin_time = 0;
    wave.instructions_size = insts_.size();
    wave.instructions_array = insts_.data();
    return wave;
  }

 private:
  std::vector<rocprofiler_thread_trace_decoder_inst_t> insts_;
  int64_t next_time_ = 0;
};

// Checks that `loop` is the body's back-edge, run `iterations` times.
void expect_body_loop(const DetectedLoop& loop, uint32_t iterations) {
  EXPECT_EQ(loop.back_edge.code_object_id, kCodeObject);
  EXPECT_EQ(loop.back_edge.source_addr, kLoopBranch);
  EXPECT_EQ(loop.back_edge.target_addr, kLoopHeader);
  EXPECT_EQ(loop.stats.iteration_count, iterations);
  const uint64_t insts = kBodySize * iterations;
  EXPECT_EQ(loop.stats.total_inst_count, insts);
  EXPECT_EQ(loop.stats.total_duration, insts * kDuration);
  EXPECT_EQ(loop.stats.total_stall, insts * kStall);
  EXPECT_EQ(loop.stats.total_idle, insts * kIdle);
}

TEST(DetectLoopsTest, CountsEachPassThroughTheBody) {
  WaveBuilder builder;
  builder.inst(kPrologue, kSalu)
      .iteration(/*branch_taken=*/true)
      .iteration(/*branch_taken=*/true)
      .iteration(/*branch_taken=*/false)
      .inst(kEpilogue, kSalu);

  auto info = detect_loops(builder.wave());

  EXPECT_EQ(info.cu, 1);
  EXPECT_EQ(info.simd, 2);
  EXPECT_EQ(info.wave_id, 3);
  ASSERT_EQ(info.loops.size(), 1u);
  expect_body_loop(info.loops[0], 3);
}

TEST(DetectLoopsTest, IgnoresBodyRunOnce) {
  WaveBuilder builder;
  builder.inst(kPrologue, kSalu)
      .iteration(/*branch_taken=*/false)
      .inst(kEpilogue, kSalu);

  EXPECT_TRUE(detect_loops(builder.wave()).loops.empty());
}

// The smallest loop reported: the pass after the one back-edge counts too.
TEST(DetectLoopsTest, BackEdgeTakenOnceIsTwoIterations) {
  WaveBuilder builder;
  builder.inst(kPrologue, kSalu)
      .iteration(/*branch_taken=*/true)
      .iteration(/*branch_taken=*/false)
      .inst(kEpilogue, kSalu);

  auto info = detect_loops(builder.wave());

  ASSERT_EQ(info.loops.size(), 1u);
  expect_body_loop(info.loops[0], 2);
}

TEST(DetectLoopsTest, SkipsNullPcRecords) {
  WaveBuilder builder;
  builder.inst(kPrologue, kSalu);
  for (int i = 0; i < 3; i++) {
    builder.inst(kLoopHeader, kValu)
        .null_pc()
        .inst(kLoopHeader + 4, kValu)
        .inst(kLoopHeader + 8, kSalu)
        .null_pc()
        .inst(kLoopBranch, i < 2 ? kJump : kNext);
  }
  builder.inst(kEpilogue, kSalu);

  auto info = detect_loops(builder.wave());

  ASSERT_EQ(info.loops.size(), 1u);
  expect_body_loop(info.loops[0], 3);
}

}  // namespace
}  // namespace my_rocperf_tool
