#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace my_rocperf_tool {

struct CodeObjectPath {
  std::string path;
  uint64_t proc_id;
  std::string arch;
  uint64_t id;
};

struct AttPath {
  std::string path;
  uint64_t proc_id;
  uint64_t agent_id;
  uint64_t se_id;
  uint64_t dispatch_id;
};

class AttOutputDir {
 public:
  explicit AttOutputDir(const std::string& path);

  std::pair<std::unique_ptr<char[]>, size_t> read_att_data(
      const AttPath& path) const;

  std::vector<CodeObjectPath> code_objects;
  std::vector<AttPath> att_paths;
};

}  // namespace my_rocperf_tool
