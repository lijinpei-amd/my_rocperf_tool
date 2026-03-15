#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

namespace my_rocperf_tool {

static union alignas(llvm::InitLLVM) {
  char storage[sizeof(llvm::InitLLVM)];
} X;

void init_llvm(int argc, const char *argv[]) {
  new (&X.storage) llvm::InitLLVM(argc, argv);

  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllDisassemblers();

  // Register the target printer for --version.
  llvm::cl::AddExtraVersionPrinter(
      llvm::TargetRegistry::printRegisteredTargetsForVersion);

  llvm::cl::ParseCommandLineOptions(argc, argv, "my rocperf tool\n");
}

void fini_llvm() {
  static_cast<llvm::InitLLVM *>((void *)&X.storage[0])->~InitLLVM();
}

} // namespace my_rocperf_tool
