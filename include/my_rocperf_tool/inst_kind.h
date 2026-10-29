#pragma once

#include <cstdint>

namespace my_rocperf_tool {

// What the analyses need to know about an instruction. The rules match LLVM
// AMDGPU opcode names, so an LLVM bump that renames opcodes is fixed where
// they are defined, in disassembler.cpp.
enum class InstKind : uint8_t {
  Other,
  // Opcode name starts with V_MFMA: every v_mfma_*, v_mfma_ld_scale_b32
  // included, but not the sparse v_smfmac_* family.
  Mfma,
  // The classic s_barrier, not gfx12's split s_barrier_signal /
  // s_barrier_wait family.
  ClassicBarrier,
};

}  // namespace my_rocperf_tool
