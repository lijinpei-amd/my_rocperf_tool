#pragma once

// Synthetic decoded waves, and the instruction kinds and loop body listings
// that go with them, for the tests of the modules that analyze waves.

#include "my_rocperf_tool/inst_kind.h"
#include "my_rocperf_tool/loop_detection.h"
#include "my_rocperf_tool/loop_profile.h"
#include "my_rocperf_tool/wave_analysis.h"

#include "gtest/gtest.h"

#include "trace_decoder_types.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace my_rocperf_tool {

using Category = rocprofiler_thread_trace_decoder_inst_category_t;

inline constexpr auto kSalu = ROCPROFILER_THREAD_TRACE_DECODER_INST_SALU;
inline constexpr auto kValu = ROCPROFILER_THREAD_TRACE_DECODER_INST_VALU;
inline constexpr auto kJump = ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP;
inline constexpr auto kNext = ROCPROFILER_THREAD_TRACE_DECODER_INST_NEXT;
inline constexpr auto kMessage = ROCPROFILER_THREAD_TRACE_DECODER_INST_MESSAGE;
inline constexpr auto kImmed = ROCPROFILER_THREAD_TRACE_DECODER_INST_IMMED;

inline constexpr uint64_t kCodeObject = 1;

// A 4-instruction loop body closed by a conditional branch back to its
// header, and the instructions either side of it.
inline constexpr uint64_t kPrologue = 0xf0;
inline constexpr uint64_t kLoopHeader = 0x100;
inline constexpr uint64_t kLoopBranch = 0x10c;
inline constexpr uint64_t kEpilogue = 0x110;
inline constexpr BackEdge kLoop{kCodeObject, kLoopBranch, kLoopHeader};

// What WaveBuilder::inst records look like: each starts kSpacing cycles after
// the previous one and lasts kDuration cycles, kStall of them stalled.
inline constexpr int64_t kSpacing = 10;
inline constexpr int32_t kDuration = 4;
inline constexpr uint32_t kStall = 1;

inline rocprofiler_thread_trace_decoder_pc_t pc(uint64_t addr) {
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

// The listing of `edge`'s body as the CLI would pass it to LoopProfile: an
// instruction per dword, its text naming its address, of the kind `kinds`
// lists for it.
inline LoopBody list_body(const BackEdge& edge,
                          const std::map<uint64_t, InstKind>& kinds = {}) {
  LoopBody body;
  for (uint64_t addr = edge.target_addr; addr <= edge.source_addr; addr += 4) {
    auto it = kinds.find(addr);
    body.push_back({addr, "inst_" + std::to_string(addr),
                    it == kinds.end() ? InstKind::Other : it->second});
  }
  return body;
}

// The row of `metrics` for the instruction at `addr`.
inline LoopInstRow row_at(const LoopRecordMetrics& metrics, uint64_t addr) {
  for (const auto& row : metrics.rows)
    if (row.vaddr == addr) return row;
  ADD_FAILURE() << "no row at 0x" << std::hex << addr;
  return {};
}

}  // namespace my_rocperf_tool
