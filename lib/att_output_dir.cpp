#include "my_rocperf_tool/att_output_dir.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

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

llvm::Error cannot_read(const std::filesystem::path& path, std::error_code ec) {
  return llvm::createStringError(
      ec, "cannot read " + path.string() + ": " + ec.message());
}

}  // namespace

llvm::Expected<AttOutputDir> AttOutputDir::scan(const std::string& path) {
  AttOutputDir out_dir;
  const std::filesystem::path root(path);
  // `--output-file` may contain directories, and a scanned tree can hold
  // several runs side by side (e.g. one subdirectory per host), so the same
  // file-name prefix in two subdirectories is two different runs.
  auto run_prefix = [&](const std::filesystem::path& dir,
                        const std::string& name_prefix) {
    // Lexical is exact here, since every dir is root with names appended;
    // std::filesystem::relative would stat both paths and can throw.
    auto rel_dir = dir.lexically_relative(root);
    if (rel_dir.empty() || rel_dir == ".") return name_prefix;
    return (rel_dir / name_prefix).generic_string();
  };
  // Walked one directory at a time: recursive_directory_iterator, on failing
  // to enter a subdirectory, does not say which one.
  std::vector<std::filesystem::path> dirs{root};
  while (!dirs.empty()) {
    auto dir = std::move(dirs.back());
    dirs.pop_back();
    std::error_code ec;
    for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end;
         it.increment(ec)) {
      const auto& dir_ent = *it;
      std::error_code type_ec;
      // Symlinked directories are not descended, which also rules out cycles.
      if (!dir_ent.is_symlink(type_ec) && dir_ent.is_directory(type_ec)) {
        dirs.push_back(dir_ent.path());
        continue;
      }
      // Follows symlinks, so type_ec is set when the target cannot be stat'ed.
      // Such a file (a dangling symlink, say) is kept: reading it reports why,
      // where skipping it here would silently shrink the capture.
      const bool keep = dir_ent.is_regular_file(type_ec) || type_ec;
      auto filename = dir_ent.path().filename().string();
      std::smatch mat;
      if (std::regex_match(filename, mat, code_object_re)) {
        if (!keep) continue;
        std::string prefix = run_prefix(dir, mat[1]);
        std::string arch = mat[2];
        auto id = parse_string<uint64_t>(mat[3]);
        out_dir.code_objects.push_back(CodeObjectPath{
            dir_ent.path().string(), std::move(prefix), arch, id});
      } else if (std::regex_match(filename, mat, att_re)) {
        if (!keep) continue;
        std::string prefix = run_prefix(dir, mat[1]);
        auto agent_id = parse_string<uint64_t>(mat[2]);
        auto se_id = parse_string<uint64_t>(mat[3]);
        auto dispatch_id = parse_string<uint64_t>(mat[4]);
        out_dir.att_paths.push_back(AttPath{dir_ent.path().string(),
                                            std::move(prefix), agent_id, se_id,
                                            dispatch_id});
      }
    }
    if (ec) return cannot_read(dir, ec);
  }
  // Grouping by prefix keeps each run's traces contiguous, so a decode worker
  // walking them in order loads one run's code objects at a time.
  std::sort(out_dir.att_paths.begin(), out_dir.att_paths.end(),
            [](const AttPath& a, const AttPath& b) {
              return std::tie(a.prefix, a.dispatch_id, a.se_id, a.path) <
                     std::tie(b.prefix, b.dispatch_id, b.se_id, b.path);
            });
  return out_dir;
}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> AttOutputDir::read_att_data(
    const AttPath& path) const {
  auto buf = llvm::MemoryBuffer::getFile(path.path, /*IsText=*/false,
                                         /*RequiresNullTerminator=*/false);
  if (!buf) return cannot_read(path.path, buf.getError());
  return std::move(*buf);
}

}  // namespace my_rocperf_tool
