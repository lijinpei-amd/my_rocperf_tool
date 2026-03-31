#pragma once

#include <memory>
#include <string>
#include <unordered_map>
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

struct DBPath {
  std::string path;
  uint64_t proc_id;
};

class AttOutputDir {
public:
  explicit AttOutputDir(const std::string &path);

  std::unordered_map<int, uint64_t> read_load_bases() const;
  std::pair<std::unique_ptr<char[]>, size_t> read_att_data() const;

  std::vector<CodeObjectPath> code_objects;
  AttPath att_path;
  DBPath db_path;
};

} // namespace my_rocperf_tool
