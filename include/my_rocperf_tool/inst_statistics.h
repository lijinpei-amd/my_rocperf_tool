#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"

#include "trace_decoder_types.h"

#include <iostream>
#include <vector>

namespace llvm {
template <> struct DenseMapInfo<rocprofiler_thread_trace_decoder_pc_t> {
  using UInt64Info = DenseMapInfo<uint64_t>;
  static constexpr rocprofiler_thread_trace_decoder_pc_t getEmptyKey() {
    return {UInt64Info::getEmptyKey(), UInt64Info::getEmptyKey()};
  }

  static constexpr rocprofiler_thread_trace_decoder_pc_t getTombstoneKey() {
    return {UInt64Info::getTombstoneKey(), UInt64Info::getTombstoneKey()};
  }

  static unsigned
  getHashValue(const rocprofiler_thread_trace_decoder_pc_t &val) {
    return hash_combine(UInt64Info::getHashValue(val.address),
                        UInt64Info::getHashValue(val.code_object_id));
  }

  static bool isEqual(const rocprofiler_thread_trace_decoder_pc_t &lhs,
                      const rocprofiler_thread_trace_decoder_pc_t &rhs) {
    return lhs.address == rhs.address &&
           lhs.code_object_id == rhs.code_object_id;
  }
};
} // namespace llvm

namespace my_rocperf_tool {
class InstStatistics {
  using pc_t = rocprofiler_thread_trace_decoder_pc_t;
  using rocprof_inst_t = rocprofiler_thread_trace_decoder_inst_t;
  struct Inst : public rocprof_inst_t {
    Inst(rocprof_inst_t inst, uint32_t idle)
        : rocprof_inst_t(inst), idle(idle) {}
    uint32_t idle;
  };
  llvm::DenseMap<pc_t, std::vector<Inst>> insts;

public:
  bool add_inst(const rocprofiler_thread_trace_decoder_inst_t &inst,
                uint32_t idle) {
    const auto &pc = inst.pc;
    auto &states = insts[pc];
    bool res = states.empty();
    if (!res) {
      auto old_category = states[0].category;
      auto new_category = inst.category;
      auto is_branch = [](auto category) {
        return category == ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP ||
               category == ROCPROFILER_THREAD_TRACE_DECODER_INST_NEXT;
      };
      res = old_category == new_category ||
            is_branch(old_category) && is_branch(new_category);
    }
    states.push_back({inst, idle});
    return res;
  }
  llvm::ArrayRef<Inst> get_inst_at(const pc_t &pc) const {
    auto iter = insts.find(pc);
    if (iter == insts.end()) {
      return {};
    }
    return iter->second;
  }
  size_t size() const { return insts.size(); }
  const llvm::DenseMap<pc_t, std::vector<Inst>> &getInsts() const {
    return insts;
  }
};
} // namespace my_rocperf_tool
