#pragma once

#include "llvm/Support/InitLLVM.h"

namespace my_rocperf_tool {

// Process setup for an executable that uses LLVM: crash signal handlers and
// pretty stack traces for the scope's lifetime, llvm_shutdown on destruction.
// Optional -- a Disassembler registers the target it needs itself. Parses no
// LLVM command-line options, so argv is used only to describe the process in
// crash reports.
class InitLLVM : public llvm::InitLLVM {
 public:
  using llvm::InitLLVM::InitLLVM;
};

}  // namespace my_rocperf_tool
