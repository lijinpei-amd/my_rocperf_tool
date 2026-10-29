#pragma once

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace my_rocperf_tool {

// The leading field of a rocprofv3 output file name is the `--output-file`
// prefix, which defaults to the profiled process' pid but can be any string.
// `prefix` also carries the file's directory relative to the scanned root
// (e.g. "host1/1234"). One prefix is one profiled run: code object ids and
// dispatch ids are only unique within it.
struct CodeObjectPath {
  std::string path;
  std::string prefix;
  std::string arch;
  uint64_t id;
};

struct AttPath {
  std::string path;
  std::string prefix;
  uint64_t agent_id;
  uint64_t se_id;
  uint64_t dispatch_id;
};

// Both reads fail with "cannot read <path>: <OS reason>", so a caller only
// adds its own framing.
class AttOutputDir {
 public:
  // Fails if `path` or any directory under it cannot be listed: a partial
  // scan would silently drop whole runs.
  static llvm::Expected<AttOutputDir> scan(const std::string& path);

  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> read_att_data(
      const AttPath& path) const;

  std::vector<CodeObjectPath> code_objects;
  std::vector<AttPath> att_paths;

 private:
  AttOutputDir() = default;
};

}  // namespace my_rocperf_tool
