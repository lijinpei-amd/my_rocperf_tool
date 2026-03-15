#pragma once

#include "disassembler.h"
#include "rocprofiler-sdk/fwd.h"

namespace my_rocperf_tool {
class RocPerfTool {
  rocprofiler_context_id_t client_ctx{};
  std::unique_ptr<Disassembler> disassembler;

public:
  void init();
  void finish();
  void
  code_object_tracing_callback(rocprofiler_callback_tracing_record_t record,
                               rocprofiler_user_data_t *user_data);
};
} // namespace my_rocperf_tool
