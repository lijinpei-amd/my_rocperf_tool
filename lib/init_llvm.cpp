#include "my_rocperf_tool/init_llvm.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"

namespace my_rocperf_tool {

InitLLVM::InitLLVM(int &Argc, const char **&Argv) : llvm::InitLLVM(Argc, Argv) {
  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllDisassemblers();

  // Register the target printer for --version.
  llvm::cl::AddExtraVersionPrinter(
      llvm::TargetRegistry::printRegisteredTargetsForVersion);

  llvm::cl::ParseCommandLineOptions(Argc, Argv, "my rocperf tool\n");
}

} // namespace my_rocperf_tool
