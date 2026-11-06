#pragma once

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"

#include <cstdint>
#include <tuple>

namespace my_rocperf_tool {

enum class AggregateScope { Wave, SimdSlot, All };

struct WaveSlot {
  uint8_t cu = 0;
  uint8_t simd = 0;
  uint8_t wave_id = 0;

  bool operator==(const WaveSlot& other) const {
    return std::tie(cu, simd, wave_id) ==
           std::tie(other.cu, other.simd, other.wave_id);
  }
  bool operator<(const WaveSlot& other) const {
    return std::tie(cu, simd, wave_id) <
           std::tie(other.cu, other.simd, other.wave_id);
  }
};

}  // namespace my_rocperf_tool

namespace llvm {
template <>
struct DenseMapInfo<my_rocperf_tool::WaveSlot> {
  using Slot = my_rocperf_tool::WaveSlot;
  using UInt8Info = DenseMapInfo<uint8_t>;
  static Slot getEmptyKey() {
    return {UInt8Info::getEmptyKey(), UInt8Info::getEmptyKey(),
            UInt8Info::getEmptyKey()};
  }
  static Slot getTombstoneKey() {
    return {UInt8Info::getTombstoneKey(), UInt8Info::getTombstoneKey(),
            UInt8Info::getTombstoneKey()};
  }
  static unsigned getHashValue(const Slot& slot) {
    return hash_combine(slot.cu, slot.simd, slot.wave_id);
  }
  static bool isEqual(const Slot& lhs, const Slot& rhs) { return lhs == rhs; }
};
}  // namespace llvm
