#pragma once

#include <string_view>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "my_rocperf_tool/report_config.h"

namespace my_rocperf_tool {

// Frames trace chunks as one document; None writes no bytes in any format.
void write_report_document(const ReportConfig& cfg,
                           llvm::ArrayRef<ReportChunk> chunks,
                           llvm::function_ref<void(std::string_view)> write);

}  // namespace my_rocperf_tool
