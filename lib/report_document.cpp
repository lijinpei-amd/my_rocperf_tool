#include "my_rocperf_tool/report_document.h"
#include "my_rocperf_tool/loop_report.h"

namespace my_rocperf_tool {

void write_report_document(const ReportConfig& cfg,
                           llvm::ArrayRef<ReportChunk> chunks,
                           llvm::function_ref<void(std::string_view)> write) {
  if (cfg.report == ReportKind::None) return;
  auto write_chunk = [&](const ReportChunk& chunk) {
    write(std::string_view(chunk.data(), chunk.size()));
  };
  if (cfg.format == ReportFormat::Json) {
    write("[\n");
  } else if (cfg.format == ReportFormat::Csv) {
    ReportChunk header;
    write_loop_csv_header(header, cfg);
    write_chunk(header);
  }
  // JSON separates entries: a trace that printed none, whose chunk is empty,
  // gets no separator either.
  bool first_nonempty = true;
  for (const auto& chunk : chunks) {
    if (chunk.size() == 0) continue;
    if (cfg.format == ReportFormat::Json && !first_nonempty) write(",\n");
    write_chunk(chunk);
    first_nonempty = false;
  }
  if (cfg.format == ReportFormat::Json) write("\n]\n");
}

}  // namespace my_rocperf_tool
