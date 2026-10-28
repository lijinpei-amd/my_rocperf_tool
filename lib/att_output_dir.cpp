#include "my_rocperf_tool/att_output_dir.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

namespace my_rocperf_tool {

namespace {

// The output-file prefix may contain '_', so it is matched greedily and the
// fields that follow it are constrained instead: the ISA name has no '_', and
// the ids are numeric.
static std::regex code_object_re =
    std::regex("(.*)_([^_]+)_code_object_id_(\\d+)\\.out");
static std::regex att_re =
    std::regex("(.*)_(\\d+)_shader_engine_(\\d+)_(\\d+)\\.att");

template <typename T>
T parse_string(const std::string& str) {
  std::istringstream iss(str);
  T result;
  iss >> result;
  return result;
}

}  // namespace

AttOutputDir::AttOutputDir(const std::string& path) {
  // `--output-file` may contain directories, and a scanned tree can hold
  // several runs side by side (e.g. one subdirectory per host), so the same
  // file-name prefix in two subdirectories is two different runs.
  auto run_prefix = [&](const std::filesystem::path& file,
                        const std::string& name_prefix) {
    auto dir = std::filesystem::relative(file.parent_path(), path);
    if (dir.empty() || dir == ".") return name_prefix;
    return (dir / name_prefix).generic_string();
  };
  for (const auto& dir_ent :
       std::filesystem::recursive_directory_iterator(path)) {
    if (!dir_ent.is_regular_file()) {
      continue;
    }
    auto& ent_path = dir_ent.path();
    auto filename = ent_path.filename().string();
    {
      std::smatch mat;
      if (std::regex_match(filename, mat, code_object_re)) {
        std::string prefix = run_prefix(ent_path, mat[1]);
        std::string arch = mat[2];
        auto id = parse_string<uint64_t>(mat[3]);
        code_objects.push_back(
            CodeObjectPath{ent_path.string(), std::move(prefix), arch, id});
        continue;
      }
    }
    {
      std::smatch mat;
      if (std::regex_match(filename, mat, att_re)) {
        std::string prefix = run_prefix(ent_path, mat[1]);
        auto agent_id = parse_string<uint64_t>(mat[2]);
        auto se_id = parse_string<uint64_t>(mat[3]);
        auto dispatch_id = parse_string<uint64_t>(mat[4]);
        att_paths.push_back(AttPath{ent_path.string(), std::move(prefix),
                                    agent_id, se_id, dispatch_id});
        continue;
      }
    }
  }
  // Grouping by prefix keeps each run's traces contiguous, so a decode worker
  // walking them in order loads one run's code objects at a time.
  std::sort(att_paths.begin(), att_paths.end(),
            [](const AttPath& a, const AttPath& b) {
              return std::tie(a.prefix, a.dispatch_id, a.se_id, a.path) <
                     std::tie(b.prefix, b.dispatch_id, b.se_id, b.path);
            });
}

std::pair<std::unique_ptr<char[]>, size_t> AttOutputDir::read_att_data(
    const AttPath& path) const {
  std::ifstream att_file(path.path, std::ios::binary);
  att_file.seekg(0, std::ios::end);
  size_t att_file_size = att_file.tellg();
  att_file.seekg(0, std::ios::beg);
  auto att_file_content = std::make_unique<char[]>(att_file_size);
  att_file.read(att_file_content.get(), att_file_size);
  return {std::move(att_file_content), att_file_size};
}

}  // namespace my_rocperf_tool
