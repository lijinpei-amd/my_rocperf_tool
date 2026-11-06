#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include "gtest/gtest.h"

namespace my_rocperf_tool {

// The lines of a document that ends each line with '\n'.
inline std::vector<std::string_view> lines(std::string_view doc) {
  std::vector<std::string_view> result;
  while (!doc.empty()) {
    size_t end = doc.find('\n');
    EXPECT_NE(end, std::string_view::npos) << "unterminated line: " << doc;
    if (end == std::string_view::npos) end = doc.size() - 1;
    result.push_back(doc.substr(0, end));
    doc.remove_prefix(end + 1);
  }
  return result;
}

// Just enough of RFC 4180: a quoted field may hold commas, and "" in it is a
// quote.
inline std::vector<std::string> split_csv(std::string_view line) {
  std::vector<std::string> fields(1);
  bool quoted = false;
  for (size_t i = 0; i < line.size(); i++) {
    const char c = line[i];
    if (quoted) {
      if (c != '"') {
        fields.back() += c;
      } else if (i + 1 < line.size() && line[i + 1] == '"') {
        fields.back() += '"';
        i++;
      } else {
        quoted = false;
      }
    } else if (c == '"') {
      quoted = true;
    } else if (c == ',') {
      fields.emplace_back();
    } else {
      fields.back() += c;
    }
  }
  EXPECT_FALSE(quoted) << "unterminated quote: " << line;
  return fields;
}

inline size_t column(const std::vector<std::string>& header,
                     std::string_view name) {
  for (size_t i = 0; i < header.size(); i++)
    if (header[i] == name) return i;
  ADD_FAILURE() << "no column " << name;
  return 0;
}

}  // namespace my_rocperf_tool
