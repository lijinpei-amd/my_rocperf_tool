#include "my_rocperf_tool/att_output_dir.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

namespace my_rocperf_tool {
namespace {

namespace fs = std::filesystem;

using CodeObjectFields =
    std::tuple<std::string, std::string, std::string, uint64_t>;
using AttFields =
    std::tuple<std::string, std::string, uint64_t, uint64_t, uint64_t>;

std::vector<CodeObjectFields> code_object_fields(const AttOutputDir& dir) {
  std::vector<CodeObjectFields> fields;
  for (const auto& co : dir.code_objects)
    fields.emplace_back(co.path, co.prefix, co.arch, co.id);
  return fields;
}

std::vector<AttFields> att_fields(const AttOutputDir& dir) {
  std::vector<AttFields> fields;
  for (const auto& att : dir.att_paths)
    fields.emplace_back(att.path, att.prefix, att.agent_id, att.se_id,
                        att.dispatch_id);
  return fields;
}

std::string cannot_read(const std::string& path, std::errc reason) {
  return "cannot read " + path + ": " + std::make_error_code(reason).message();
}

// Each test scans its own fresh directory of empty, rocprofv3-named files.
class AttOutputDirTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string dir = ::testing::TempDir() + "att_output_dir_test.XXXXXX";
    ASSERT_NE(mkdtemp(dir.data()), nullptr) << std::strerror(errno);
    root_ = dir;
  }

  void TearDown() override {
    if (root_.empty()) return;
    // remove_all cannot descend into a directory a test made unreadable.
    if (!locked_dir_.empty())
      fs::permissions(locked_dir_, fs::perms::owner_all);
    fs::remove_all(root_);
  }

  std::string path(const std::string& rel) const {
    return (root_ / rel).string();
  }

  void touch(const std::string& rel) {
    fs::create_directories((root_ / rel).parent_path());
    std::ofstream file(root_ / rel);
    ASSERT_TRUE(file.is_open()) << path(rel);
  }

  fs::path root_;
  fs::path locked_dir_;
};

TEST_F(AttOutputDirTest, ParsesPrefixesContainingUnderscores) {
  touch("1234_gfx942_code_object_id_1.out");
  touch("1234_37690_shader_engine_0_1.att");
  touch("my_run_v2_gfx950_code_object_id_2.out");
  touch("my_run_v2_7_shader_engine_3_4.att");
  touch("my_run_v2_results.db");

  auto dir = AttOutputDir::scan(root_.string());
  ASSERT_TRUE(static_cast<bool>(dir)) << llvm::toString(dir.takeError());

  EXPECT_EQ(code_object_fields(*dir),
            (std::vector<CodeObjectFields>{
                {path("1234_gfx942_code_object_id_1.out"), "1234", "gfx942", 1},
                {path("my_run_v2_gfx950_code_object_id_2.out"), "my_run_v2",
                 "gfx950", 2},
            }));
  EXPECT_EQ(
      att_fields(*dir),
      (std::vector<AttFields>{
          {path("1234_37690_shader_engine_0_1.att"), "1234", 37690, 0, 1},
          {path("my_run_v2_7_shader_engine_3_4.att"), "my_run_v2", 7, 3, 4},
      }));
}

TEST_F(AttOutputDirTest, SameNameInTwoSubdirectoriesIsTwoRuns) {
  for (const char* host : {"host1", "host2"}) {
    touch(std::string(host) + "/1234_gfx942_code_object_id_1.out");
    touch(std::string(host) + "/1234_1_shader_engine_0_1.att");
  }

  auto dir = AttOutputDir::scan(root_.string());
  ASSERT_TRUE(static_cast<bool>(dir)) << llvm::toString(dir.takeError());

  EXPECT_EQ(code_object_fields(*dir),
            (std::vector<CodeObjectFields>{
                {path("host1/1234_gfx942_code_object_id_1.out"), "host1/1234",
                 "gfx942", 1},
                {path("host2/1234_gfx942_code_object_id_1.out"), "host2/1234",
                 "gfx942", 1},
            }));
  EXPECT_EQ(
      att_fields(*dir),
      (std::vector<AttFields>{
          {path("host1/1234_1_shader_engine_0_1.att"), "host1/1234", 1, 0, 1},
          {path("host2/1234_1_shader_engine_0_1.att"), "host2/1234", 1, 0, 1},
      }));
}

TEST_F(AttOutputDirTest, SortsTracesByPrefixDispatchSeThenPath) {
  // Ids compare as numbers, not strings. The eight traces tied on prefix,
  // dispatch and SE fall back to path order, which is not agent-id order
  // (run_10_ sorts before run_1_); with eight ties, directory order matching
  // path order by chance is negligible.
  const std::vector<std::string> sorted = {
      "a_1_shader_engine_3_20.att",  // prefix a
      // The eight tied on prefix run, dispatch 9 and SE 0, by path.
      "run_10_shader_engine_0_9.att",
      "run_1_shader_engine_0_9.att",
      "run_2_shader_engine_0_9.att",
      "run_3_shader_engine_0_9.att",
      "run_4_shader_engine_0_9.att",
      "run_5_shader_engine_0_9.att",
      "run_6_shader_engine_0_9.att",
      "run_7_shader_engine_0_9.att",
      "run_1_shader_engine_2_9.att",   // prefix run, dispatch 9, SE 2
      "run_1_shader_engine_10_9.att",  // prefix run, dispatch 9, SE 10
      "run_1_shader_engine_0_10.att",  // prefix run, dispatch 10
  };
  // Some filesystems list a directory in creation order and some in reverse,
  // so create the files in neither: stride 5 visits each of the 12 once.
  for (size_t i = 0; i < sorted.size(); ++i)
    touch(sorted[i * 5 % sorted.size()]);

  auto dir = AttOutputDir::scan(root_.string());
  ASSERT_TRUE(static_cast<bool>(dir)) << llvm::toString(dir.takeError());

  std::vector<std::string> paths;
  for (const auto& att : dir->att_paths) paths.push_back(att.path);
  std::vector<std::string> expected;
  for (const auto& name : sorted) expected.push_back(path(name));
  EXPECT_EQ(paths, expected);
}

// A run loads its code objects in this order, so its load errors come out in
// it too.
TEST_F(AttOutputDirTest, SortsCodeObjectsByPrefixIdThenPath) {
  // Ids compare as numbers, not strings, and the three tied on prefix and id
  // fall back to path order, so neither path nor directory order gives this.
  const std::vector<std::string> sorted = {
      "a_gfx942_code_object_id_2.out",  // prefix a
      // prefix run, id 1, then by path
      "run_gfx90a_code_object_id_1.out",
      "run_gfx942_code_object_id_1.out",
      "run_gfx950_code_object_id_1.out",
      "run_gfx942_code_object_id_2.out",    // prefix run, id 2
      "run_gfx942_code_object_id_10.out",   // prefix run, id 10
      "run_x_gfx942_code_object_id_1.out",  // prefix run_x
  };
  // Neither creation order nor its reverse: stride 3 visits each of the 7
  // once.
  for (size_t i = 0; i < sorted.size(); ++i)
    touch(sorted[i * 3 % sorted.size()]);

  auto dir = AttOutputDir::scan(root_.string());
  ASSERT_TRUE(static_cast<bool>(dir)) << llvm::toString(dir.takeError());

  std::vector<std::string> paths;
  for (const auto& co : dir->code_objects) paths.push_back(co.path);
  std::vector<std::string> expected;
  for (const auto& name : sorted) expected.push_back(path(name));
  EXPECT_EQ(paths, expected);
}

TEST_F(AttOutputDirTest, KeepsMissingTraceAndReportsItOnRead) {
  touch("run_1_shader_engine_0_1.att");
  fs::create_symlink(root_ / "gone.att", root_ / "run_1_shader_engine_1_1.att");

  auto dir = AttOutputDir::scan(root_.string());
  ASSERT_TRUE(static_cast<bool>(dir)) << llvm::toString(dir.takeError());
  ASSERT_EQ(dir->att_paths.size(), 2u);

  auto present = dir->read_att_data(dir->att_paths[0]);
  ASSERT_TRUE(static_cast<bool>(present))
      << llvm::toString(present.takeError());
  EXPECT_EQ((*present)->getBufferSize(), 0u);

  auto missing = dir->read_att_data(dir->att_paths[1]);
  ASSERT_FALSE(static_cast<bool>(missing));
  EXPECT_EQ(llvm::toString(missing.takeError()),
            cannot_read(path("run_1_shader_engine_1_1.att"),
                        std::errc::no_such_file_or_directory));
}

// The Disassembler reports why it cannot load the code object; dropping it
// here would instead leave the traces that need it to fail on a missing id.
TEST_F(AttOutputDirTest, KeepsMissingCodeObject) {
  touch("run_gfx942_code_object_id_1.out");
  fs::create_symlink(root_ / "gone.out",
                     root_ / "run_gfx942_code_object_id_2.out");

  auto dir = AttOutputDir::scan(root_.string());
  ASSERT_TRUE(static_cast<bool>(dir)) << llvm::toString(dir.takeError());
  EXPECT_EQ(code_object_fields(*dir),
            (std::vector<CodeObjectFields>{
                {path("run_gfx942_code_object_id_1.out"), "run", "gfx942", 1},
                {path("run_gfx942_code_object_id_2.out"), "run", "gfx942", 2},
            }));
}

TEST_F(AttOutputDirTest, ScanReportsMissingRoot) {
  auto dir = AttOutputDir::scan(path("missing"));
  ASSERT_FALSE(static_cast<bool>(dir));
  EXPECT_EQ(llvm::toString(dir.takeError()),
            cannot_read(path("missing"), std::errc::no_such_file_or_directory));
}

TEST_F(AttOutputDirTest, ScanReportsUnreadableSubdirectory) {
  touch("run_gfx942_code_object_id_1.out");
  touch("locked/run_1_shader_engine_0_1.att");
  locked_dir_ = root_ / "locked";
  fs::permissions(locked_dir_, fs::perms::none);
  std::error_code probe_ec;
  fs::directory_iterator probe(locked_dir_, probe_ec);
  if (!probe_ec) GTEST_SKIP() << "permissions not enforced (running as root?)";

  auto dir = AttOutputDir::scan(root_.string());
  ASSERT_FALSE(static_cast<bool>(dir));
  EXPECT_EQ(llvm::toString(dir.takeError()),
            cannot_read(path("locked"), std::errc::permission_denied));
}

}  // namespace
}  // namespace my_rocperf_tool
