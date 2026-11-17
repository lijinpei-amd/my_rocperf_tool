#pragma once

#include <cstdint>

namespace my_rocperf_tool {

// The instruction classes the analyses tell apart. The rules match LLVM
// AMDGPU opcode names, so an LLVM bump that renames opcodes is fixed where
// they are defined, in disassembler.cpp.
enum class InstClass : uint8_t {
  Other,
  // Opcode name starts with V_MFMA: every v_mfma_* but v_mfma_ld_scale_b32,
  // and not the sparse v_smfmac_* family.
  Mfma,
  // The classic s_barrier, not gfx12's split s_barrier_signal /
  // s_barrier_wait family.
  ClassicBarrier,
};

// What the analyses need to know about an instruction. A value-initialized
// InstKind is Other.
struct InstKind {
  InstClass cls = InstClass::Other;
  // An MFMA's cycles on its pipe, during which a co-issued non-MFMA executes
  // for free: LLVM's scheduling-model reciprocal throughput for the ISA, in
  // 4-cycle passes, times 4. Nonzero exactly when cls is Mfma.
  uint16_t mfma_cycles = 0;

  static constexpr InstKind mfma(uint16_t cycles) {
    return {InstClass::Mfma, cycles};
  }
  static constexpr InstKind classic_barrier() {
    return {InstClass::ClassicBarrier, 0};
  }

  friend constexpr bool operator==(InstKind a, InstKind b) {
    return a.cls == b.cls && a.mfma_cycles == b.mfma_cycles;
  }
  friend constexpr bool operator!=(InstKind a, InstKind b) { return !(a == b); }
};

}  // namespace my_rocperf_tool
