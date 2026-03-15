#pragma once

#include "absl/base/log_severity.h"
#include "rocprofiler-sdk/fwd.h"

#include <string_view>

#define CHECK_ROCPERF_STAT(_API_FUNC, ...)                                     \
  ::my_rocperf_tool::check_rocprofiler_status_impl(                            \
      _API_FUNC(__VA_ARGS__), absl::LogSeverity::kWarning, #_API_FUNC,         \
      __FILE__, __LINE__)

namespace my_rocperf_tool {
void check_rocprofiler_status_impl(rocprofiler_status_t error,
                                   absl::LogSeverity log_severity,
                                   std::string_view api_name,
                                   std::string_view file, int line);
}
