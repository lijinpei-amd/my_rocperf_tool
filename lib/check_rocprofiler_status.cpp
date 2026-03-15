#include "my_rocperf_tool/check_rocprofiler_status.h"

#include "absl/log/log_streamer.h"
#include "rocprofiler-sdk/rocprofiler.h"

namespace my_rocperf_tool {
void check_rocprofiler_status_impl(rocprofiler_status_t status,
                                   absl::LogSeverity log_severity,
                                   std::string_view api_name,
                                   std::string_view file, int line) {
  if (status == ROCPROFILER_STATUS_SUCCESS) {
    return;
  }
  const char *error_name = rocprofiler_get_status_name(status);
  if (!error_name) {
    error_name = "<unknown status>";
  }
  const char *error_desc = rocprofiler_get_status_string(status);
  if (!error_desc) {
    error_desc = "<unknown description>";
  }
  absl::LogStreamer(log_severity, file, line).stream()
      << "rocprofiler-sdk API " << api_name << " failed with error_code "
      << (int)status << " : " << error_name << " : " << error_desc;
}
} // namespace my_rocperf_tool
