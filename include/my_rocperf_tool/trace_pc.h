#pragma once

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"

#include "trace_decoder_types.h"

#include <cstdint>

// The decoder's {ELF vaddr, code object} PC as a DenseMap key, for the
// per-PC caches and totals.
namespace llvm {
template <>
struct DenseMapInfo<rocprofiler_thread_trace_decoder_pc_t> {
  using UInt64Info = DenseMapInfo<uint64_t>;
  static constexpr rocprofiler_thread_trace_decoder_pc_t getEmptyKey() {
    return {UInt64Info::getEmptyKey(), UInt64Info::getEmptyKey()};
  }

  static constexpr rocprofiler_thread_trace_decoder_pc_t getTombstoneKey() {
    return {UInt64Info::getTombstoneKey(), UInt64Info::getTombstoneKey()};
  }

  static unsigned getHashValue(
      const rocprofiler_thread_trace_decoder_pc_t& val) {
    return hash_combine(UInt64Info::getHashValue(val.address),
                        UInt64Info::getHashValue(val.code_object_id));
  }

  static bool isEqual(const rocprofiler_thread_trace_decoder_pc_t& lhs,
                      const rocprofiler_thread_trace_decoder_pc_t& rhs) {
    return lhs.address == rhs.address &&
           lhs.code_object_id == rhs.code_object_id;
  }
};
}  // namespace llvm
