#pragma once

#include "my_rocperf_tool/report_config.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <utility>

namespace my_rocperf_tool {

// Small wrapper around fmt::format_to so call sites stay terse.
// fmt::format_string gives compile-time format-string checks; fmt::appender
// is fmt's canonical writer for memory_buffer (tighter codegen than
// std::back_inserter).
template <typename... T>
inline void writef(ReportChunk& out, fmt::format_string<T...> f, T&&... args) {
  fmt::format_to(fmt::appender(out), f, std::forward<T>(args)...);
}

// Escape a string for CSV output (RFC 4180: double embedded quotes)
inline std::string csv_escape(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '"')
      out += "\"\"";
    else
      out += c;
  }
  return out;
}

// Escape a string for JSON output (handles all JSON-required escapes)
inline std::string json_escape(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (unsigned char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

// ANSI color for bubble severity, relative to max bubble in the group.
// Green (0%) -> Yellow (1-25%) -> Red (25-75%) -> Bright Red (75-100%)
inline const char* bubble_color(uint64_t bubble, uint64_t max_bubble) {
  if (bubble == 0 || max_bubble == 0)
    return "\033[32m";  // green: fully covered
  double ratio = static_cast<double>(bubble) / max_bubble;
  if (ratio <= 0.25) return "\033[33m";  // yellow: low
  if (ratio <= 0.75) return "\033[31m";  // red: medium
  return "\033[1;31m";                   // bright red: high
}

// Dispatch labels are shared by every text report.
inline void write_dispatch_label(ReportChunk& out, const ReportConfig& cfg,
                                 std::string_view prefix,
                                 uint64_t dispatch_id) {
  if (cfg.show_prefix) writef(out, "Prefix={}, ", prefix);
  writef(out, "DispatchID={}", dispatch_id);
}

}  // namespace my_rocperf_tool
