#include "my_rocperf_tool/loop_detection.h"

#include "llvm/ADT/DenseMap.h"

#include <algorithm>

namespace my_rocperf_tool {

WaveLoopInfo
detect_loops(const rocprofiler_thread_trace_decoder_wave_t &wave) {
  WaveLoopInfo result;
  result.cu = wave.cu;
  result.simd = wave.simd;
  result.wave_id = wave.wave_id;

  // Pass 1: discover all back-edges.
  llvm::DenseMap<BackEdge, LoopStats> loop_map;
  for (size_t i = 0; i + 1 < wave.instructions_size; i++) {
    const auto &inst = wave.instructions_array[i];
    if (is_null_pc(inst))
      continue;
    if (inst.category == ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP) {
      const auto &next = wave.instructions_array[i + 1];
      if (next.pc.code_object_id == inst.pc.code_object_id &&
          next.pc.address <= inst.pc.address) {
        BackEdge edge{inst.pc.code_object_id, inst.pc.address,
                      next.pc.address};
        loop_map.try_emplace(edge, LoopStats{});
      }
    }
  }

  if (loop_map.empty())
    return result;

  // Per-iteration accumulator keyed by back-edge.
  struct IterAccum {
    uint64_t inst_count = 0;
    uint64_t duration = 0;
    uint64_t stall = 0;
    uint64_t idle = 0;
  };
  llvm::DenseMap<BackEdge, IterAccum> active_iters;
  for (const auto &[edge, _] : loop_map)
    active_iters[edge] = IterAccum{};

  // Pass 2: accumulate stats, flushing on each back-edge hit.
  int64_t last_time = wave.begin_time;

  for (size_t i = 0; i < wave.instructions_size; i++) {
    const auto &inst = wave.instructions_array[i];
    if (is_null_pc(inst))
      continue;

    uint32_t idle = compute_idle(inst, last_time);

    // Accumulate this instruction into all active loop iterations whose
    // address range contains this instruction's PC.
    for (auto &[edge, accum] : active_iters) {
      if (inst.pc.code_object_id == edge.code_object_id &&
          inst.pc.address >= edge.target_addr &&
          inst.pc.address <= edge.source_addr) {
        accum.inst_count++;
        accum.duration += inst.duration;
        accum.stall += inst.stall;
        accum.idle += idle;
      }
    }

    // On a back-edge, flush the current iteration and start the next.
    if (inst.category == ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP &&
        i + 1 < wave.instructions_size) {
      const auto &next = wave.instructions_array[i + 1];
      if (next.pc.code_object_id == inst.pc.code_object_id &&
          next.pc.address <= inst.pc.address) {
        BackEdge edge{inst.pc.code_object_id, inst.pc.address,
                      next.pc.address};

        auto iter_it = active_iters.find(edge);
        if (iter_it != active_iters.end()) {
          auto &stats = loop_map[edge];
          stats.iteration_count++;
          stats.total_inst_count += iter_it->second.inst_count;
          stats.total_duration += iter_it->second.duration;
          stats.total_stall += iter_it->second.stall;
          stats.total_idle += iter_it->second.idle;
          iter_it->second = IterAccum{};
        }
      }
    }
  }

  // Flush any remaining active iterations (the last iteration that didn't
  // end with a back-edge jump).
  for (auto &[edge, accum] : active_iters) {
    auto it = loop_map.find(edge);
    if (it != loop_map.end() && accum.inst_count > 0) {
      it->second.iteration_count++;
      it->second.total_inst_count += accum.inst_count;
      it->second.total_duration += accum.duration;
      it->second.total_stall += accum.stall;
      it->second.total_idle += accum.idle;
    }
  }

  // Collect loops with >= 2 iterations
  for (auto &[edge, stats] : loop_map) {
    if (stats.iteration_count >= 2) {
      result.loops.push_back({edge, stats});
    }
  }

  // Sort by iteration count descending
  std::sort(result.loops.begin(), result.loops.end(),
            [](const DetectedLoop &a, const DetectedLoop &b) {
              return a.stats.iteration_count > b.stats.iteration_count;
            });

  return result;
}

} // namespace my_rocperf_tool
