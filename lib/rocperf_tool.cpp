#include "my_rocperf_tool/rocperf_tool.h"
#include "my_rocperf_tool/check_rocprofiler_status.h"
#include "my_rocperf_tool/init_llvm.h"

#include "rocprofiler-sdk/callback_tracing.h"
#include "rocprofiler-sdk/context.h"
#include "rocprofiler-sdk/registration.h"

#include "llvm/ADT/StringRef.h"

#include <iostream>

namespace my_rocperf_tool {

namespace {
void tool_code_object_tracing_callback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t *user_data, void *callback_data) {

  auto *perf_tool = static_cast<my_rocperf_tool::RocPerfTool *>(callback_data);
  return perf_tool->code_object_tracing_callback(record, user_data);
}

} // namespace

void RocPerfTool::init() {
  CHECK_ROCPERF_STAT(rocprofiler_create_context, &client_ctx);
  CHECK_ROCPERF_STAT(rocprofiler_configure_callback_tracing_service, client_ctx,
                     ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
                     tool_code_object_tracing_callback, this);
  CHECK_ROCPERF_STAT(rocprofiler_start_context, client_ctx);
  disassembler = std::move(std::make_unique<Disassembler>());
}

void RocPerfTool::finish() { disassembler.reset(); }

void RocPerfTool::code_object_tracing_callback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t *user_data) {
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT)
    return;
  if (record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD)
    return;
  if (record.operation == ROCPROFILER_CODE_OBJECT_LOAD) {
    auto *data =
        static_cast<rocprofiler_callback_tracing_code_object_load_data_t *>(
            record.payload);
    disassembler->addCodeObject(*data);
  } else if (record.operation ==
             ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) {
    auto *data = static_cast<
        rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t
            *>(record.payload);
    disassembler->registerSymbol(*data);
  }
}
} // namespace my_rocperf_tool

namespace {

int init_perftool(rocprofiler_client_finalize_t finalize_func,
                  void *tool_data) {
  const char *argv[] = {"my-rocperf-tool"};
  my_rocperf_tool::init_llvm(1, argv);
  auto *perf_tool = static_cast<my_rocperf_tool::RocPerfTool *>(tool_data);
  perf_tool->init();
  return ROCPROFILER_STATUS_SUCCESS;
}

void fini_perftool(void *tool_data) {
  auto *perf_tool = static_cast<my_rocperf_tool::RocPerfTool *>(tool_data);
  perf_tool->finish();
  my_rocperf_tool::fini_llvm();
}

} // namespace

// extern "C" __attribute__((visibility("default")))
// rocprofiler_tool_configure_result_t *
// rocprofiler_configure(uint32_t version, const char *runtime_version,
//                       uint32_t priority, rocprofiler_client_id_t *client_id)
//                       {
//   client_id->name = "my_rocperf_tool";
//
//   static my_rocperf_tool::RocPerfTool PerfTool;
//   static rocprofiler_tool_configure_result_t cfg{
//       sizeof(rocprofiler_tool_configure_result_t),
//       &init_perftool,
//       &fini_perftool,
//       &PerfTool,
//   };
//   return &cfg;
// }
