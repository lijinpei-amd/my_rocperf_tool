#include "my_rocperf_tool/att_output_dir.h"

#include "sqlite3.h"

#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

namespace my_rocperf_tool {

namespace {

static std::regex code_object_re =
    std::regex("(\\d+)_(.+)_code_object_id_(\\d+)\\.out");
static std::regex att_re =
    std::regex("(\\d+)_(\\d+)_shader_engine_(\\d+)_(\\d+)\\.att");
static std::regex db_re = std::regex("(\\d+)_results\\.db");

template <typename T> T parse_string(const std::string &str) {
  std::istringstream iss(str);
  T result;
  iss >> result;
  return result;
}

} // namespace

AttOutputDir::AttOutputDir(const std::string &path) {
  for (const auto &dir_ent :
       std::filesystem::recursive_directory_iterator(path)) {
    if (!dir_ent.is_regular_file()) {
      continue;
    }
    auto &ent_path = dir_ent.path();
    auto filename = ent_path.filename().string();
    {
      std::smatch mat;
      if (std::regex_match(filename, mat, code_object_re)) {
        auto proc_id = parse_string<uint64_t>(mat[1]);
        std::string arch = mat[2];
        auto id = parse_string<uint64_t>(mat[3]);
        code_objects.push_back(
            CodeObjectPath{ent_path.string(), proc_id, arch, id});
        continue;
      }
    }
    {
      std::smatch mat;
      if (std::regex_match(filename, mat, att_re)) {
        auto proc_id = parse_string<uint64_t>(mat[1]);
        auto agent_id = parse_string<uint64_t>(mat[2]);
        auto se_id = parse_string<uint64_t>(mat[3]);
        auto dispatch_id = parse_string<uint64_t>(mat[4]);
        att_path =
            AttPath{ent_path.string(), proc_id, agent_id, se_id, dispatch_id};
        continue;
      }
    }
    {
      std::smatch mat;
      if (std::regex_match(filename, mat, db_re)) {
        auto proc_id = parse_string<uint64_t>(mat[1]);
        db_path = DBPath{ent_path.string(), proc_id};
        continue;
      }
    }
  }
}

std::unordered_map<int, uint64_t> AttOutputDir::read_load_bases() const {
  sqlite3 *db;
  if (sqlite3_open(db_path.path.c_str(), &db)) {
    std::cerr << "open db failed: " << sqlite3_errmsg(db) << std::endl;
  }
  std::unordered_map<int, uint64_t> object_load_bases;
  const char *sql = "SELECT * FROM 'code_objects';";
  sqlite3_stmt *stmt;
  int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
  if (rc != SQLITE_OK) {
    std::cerr << "error occurred: " << sqlite3_errmsg(db) << std::endl;
    sqlite3_close(db);
    return object_load_bases;
  }
  int NoOfCols = sqlite3_column_count(stmt);
  while (true) {
    switch (sqlite3_step(stmt)) {
    case SQLITE_ROW: {
      int obj_id;
      int64_t load_base;
      bool has_obj_id = false, has_load_base = false;
      for (int i = 0; i < NoOfCols; i++) {
        const char *col_name = sqlite3_column_name(stmt, i);
        if (!strcmp(col_name, "load_base")) {
          load_base = sqlite3_column_int64(stmt, i);
          assert(!has_load_base);
          has_load_base = true;
          continue;
        }
        if (!strcmp(col_name, "id")) {
          obj_id = sqlite3_column_int(stmt, i);
          assert(!has_obj_id);
          has_obj_id = true;
          continue;
        }
      }
      assert(has_obj_id && has_load_base);
      (void)has_obj_id;
      (void)has_load_base;
      object_load_bases[obj_id] = uint64_t(load_base);
      continue;
    }
    case SQLITE_DONE:
      sqlite3_finalize(stmt);
      break;
    }
    break;
  }
  sqlite3_close(db);
  return object_load_bases;
}

std::pair<std::unique_ptr<char[]>, size_t>
AttOutputDir::read_att_data() const {
  std::ifstream att_file(att_path.path, std::ios::binary);
  att_file.seekg(0, std::ios::end);
  size_t att_file_size = att_file.tellg();
  att_file.seekg(0, std::ios::beg);
  auto att_file_content = std::make_unique<char[]>(att_file_size);
  att_file.read(att_file_content.get(), att_file_size);
  return {std::move(att_file_content), att_file_size};
}

} // namespace my_rocperf_tool
