#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
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

class AttOutputDir {
 public:
  explicit AttOutputDir(const std::string& path);

  std::pair<std::unique_ptr<char[]>, size_t> read_att_data(
      const AttPath& path) const;

  std::vector<CodeObjectPath> code_objects;
  std::vector<AttPath> att_paths;
};

}  // namespace my_rocperf_tool
