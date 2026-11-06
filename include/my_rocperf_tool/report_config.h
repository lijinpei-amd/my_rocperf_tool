#pragma once

#include "fmt/format.h"
#include "my_rocperf_tool/aggregate_scope.h"

namespace my_rocperf_tool {

enum class ReportKind {
  LoopSummary,
  LoopDetail,
  StaticInst,
  DynamicInst,
  None
};
enum class ReportFormat { Text, Csv, Json };

// Shared rendering options. CSV keeps stats and addresses; JSON always keeps
// row identities. show_prefix qualifies dispatches when several runs are read.
struct ReportConfig {
  ReportFormat format = ReportFormat::Text;
  ReportKind report = ReportKind::LoopSummary;
  AggregateScope aggregate = AggregateScope::Wave;
  bool show_stats = true;
  bool show_addr = true;
  bool show_avg = false;
  bool show_mfma_coexec = false;
  bool color_mfma_bubble = false;
  bool show_prefix = false;
};

// One trace's buffered part of the document, written in trace order.
using ReportChunk = fmt::basic_memory_buffer<char, 256>;

}  // namespace my_rocperf_tool
