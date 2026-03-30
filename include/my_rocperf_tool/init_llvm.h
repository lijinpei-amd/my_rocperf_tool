#pragma once

#include "llvm/Support/InitLLVM.h"

namespace my_rocperf_tool {

class InitLLVM : public llvm::InitLLVM {
public:
  InitLLVM(int &Argc, const char **&Argv);
};

} // namespace my_rocperf_tool
