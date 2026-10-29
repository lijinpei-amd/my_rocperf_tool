#include "my_rocperf_tool/loop_report.h"

#include "my_rocperf_tool/loop_profile.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include "fmt/format.h"

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace my_rocperf_tool {

namespace {

// Small wrapper around fmt::format_to so call sites stay terse.
// fmt::format_string gives compile-time format-string checks; fmt::appender
// is fmt's canonical writer for memory_buffer (tighter codegen than
// std::back_inserter).
template <typename... T>
inline void writef(LoopReportChunk& out, fmt::format_string<T...> f,
                   T&&... args) {
  fmt::format_to(fmt::appender(out), f, std::forward<T>(args)...);
}

// Escape a string for CSV output (RFC 4180: double embedded quotes)
std::string csv_escape(std::string_view s) {
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
std::string json_escape(std::string_view s) {
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

// ANSI color for bubble severity, relative to max bubble in the loop.
// Green (0%) -> Yellow (1-25%) -> Red (25-75%) -> Bright Red (75-100%)
const char* bubble_color(const LoopInstRow& ii, uint64_t max_bubble) {
  if (!ii.has_stats || ii.n == 0 || ii.bubble == 0 || max_bubble == 0)
    return "\033[32m";  // green: fully covered
  double ratio = static_cast<double>(ii.bubble) / max_bubble;
  if (ratio <= 0.25) return "\033[33m";  // yellow: low
  if (ratio <= 0.75) return "\033[31m";  // red: medium
  return "\033[1;31m";                   // bright red: high
}

void render_loop_json(LoopReportChunk& out, const ReportedLoop& loop,
                      const LoopOutputConfig& cfg,
                      std::optional<uint32_t> wave_count) {
  const auto& e = loop.record.back_edge;
  const auto& s = loop.record.stats;
  const auto& sum = loop.metrics.summary;
  writef(out,
         "    {{\"target_addr\":\"0x{:x}\",\"source_addr\":\"0x{:x}\","
         "\"code_object\":{}",
         e.target_addr, e.source_addr, e.code_object_id);
  if (wave_count.has_value()) writef(out, ",\"wave_count\":{}", *wave_count);
  writef(out,
         ",\"iterations\":{},\"total_duration\":{},"
         "\"total_stall\":{},\"total_idle\":{}",
         s.iteration_count, s.total_duration, s.total_stall, s.total_idle);
  writef(out, ",\"avg_dur\":{:.2f},\"avg_stall\":{:.2f},\"avg_idle\":{:.2f}",
         sum.avg_dur, sum.avg_stall, sum.avg_idle);
  if (cfg.show_mfma_coexec)
    writef(out, ",\"avg_bubble\":{:.2f}", sum.avg_bubble);
  writef(out, ",\"mfma_count\":{},\"mfma_time\":{},\"mfma_util\":{:.4f}",
         sum.mfma_count, sum.mfma_time, sum.mfma_util);
  if (cfg.show_contents) {
    writef(out, ",\"instructions\":[\n");
    bool first_inst = true;
    for (const auto& ii : loop.metrics.rows) {
      if (!first_inst) writef(out, ",\n");
      first_inst = false;
      writef(out, "      {{\"addr\":\"0x{:x}\",\"inst\":\"{}\"", ii.vaddr,
             json_escape(ii.text));
      if (cfg.show_stats && ii.has_stats) {
        writef(out, ",\"n\":{}", ii.n);
        if (cfg.show_avg) {
          writef(out,
                 ",\"avg_dur\":{:.2f},\"avg_stall\":{:.2f},"
                 "\"avg_idle\":{:.2f},\"avg_issue\":{:.2f}",
                 static_cast<double>(ii.dur) / ii.n,
                 static_cast<double>(ii.stall) / ii.n,
                 static_cast<double>(ii.idle) / ii.n,
                 static_cast<double>(ii.issue) / ii.n);
          if (cfg.show_mfma_coexec)
            writef(out, ",\"avg_bubble\":{:.2f}",
                   static_cast<double>(ii.bubble) / ii.n);
          writef(out, ",\"avg_cycle\":{:.2f}",
                 static_cast<double>(ii.rel_issue) / ii.n);
        } else {
          writef(out, ",\"dur\":{},\"stall\":{},\"idle\":{},\"issue\":{}",
                 ii.dur, ii.stall, ii.idle, ii.issue);
          if (cfg.show_mfma_coexec) writef(out, ",\"bubble\":{}", ii.bubble);
          writef(out, ",\"rel_issue\":{}", ii.rel_issue);
        }
      }
      writef(out, "}}");
    }
    writef(out, "\n    ]");
  }
  writef(out, "}}");
}

// Entries separated by ",\n", with no leading or trailing separator:
// write_loop_report_document joins the chunks and brackets the document.
void render_json(LoopReportChunk& out, const LoopOutputConfig& cfg,
                 const TraceLoopReport& report) {
  bool first_entry = true;
  for (const auto& g : report.groups) {
    if (!first_entry) writef(out, ",\n");
    first_entry = false;
    writef(out, "  {{");
    if (cfg.show_prefix)
      writef(out, "\"prefix\":\"{}\",", json_escape(report.prefix));
    if (g.is_aggregate)
      writef(out, "\"dispatch_id\":{},\"aggregate\":true,\"loops\":[\n",
             report.dispatch_id);
    else
      writef(out,
             "\"dispatch_id\":{},\"cu\":{},\"simd\":{},\"wave_id\":{},"
             "\"loops\":[\n",
             report.dispatch_id, g.cu, g.simd, g.wave_id);
    bool first_loop = true;
    for (const auto& loop : g.loops) {
      if (!first_loop) writef(out, ",\n");
      first_loop = false;
      auto wave_count = g.is_aggregate
                            ? std::optional<uint32_t>(loop.record.wave_count)
                            : std::nullopt;
      render_loop_json(out, loop, cfg, wave_count);
    }
    writef(out, "\n  ]}}");
  }
}

// What one CSV row is about: a loop record and, under show_contents, one of
// its body instructions.
struct CsvRow {
  const TraceLoopReport& report;
  const LoopGroup& group;
  const LoopRecord& record;
  const LoopSummary& summary;
  const LoopInstRow* inst = nullptr;  // null unless show_contents
};

// `total` per execution of the row's instruction. An instruction never
// executed has no average, and its field stays blank.
void write_per_execution(LoopReportChunk& out, const CsvRow& row,
                         uint64_t total) {
  if (row.inst->n > 0)
    writef(out, "{:.2f}", static_cast<double>(total) / row.inst->n);
}

// The CSV columns, in order: the one definition of both the header and every
// row, so that the two cannot disagree on a column. Calls column(name, value)
// for each column `cfg` shows, where value(out, row) appends that column's
// field of `row`, or nothing for a blank one.
template <typename Column>
void csv_columns(const LoopOutputConfig& cfg, Column&& column) {
  using Out = LoopReportChunk;
  if (cfg.aggregate_wave)
    column("record_type", [](Out& out, const CsvRow& r) {
      writef(out, "{}", r.group.is_aggregate ? "aggregate" : "wave");
    });
  if (cfg.show_prefix)
    column("prefix", [](Out& out, const CsvRow& r) {
      writef(out, "\"{}\"", csv_escape(r.report.prefix));
    });
  column("dispatch_id", [](Out& out, const CsvRow& r) {
    writef(out, "{}", r.report.dispatch_id);
  });
  // The aggregate is no one wave's, so its wave fields stay blank.
  column("cu", [](Out& out, const CsvRow& r) {
    if (!r.group.is_aggregate) writef(out, "{}", r.group.cu);
  });
  column("simd", [](Out& out, const CsvRow& r) {
    if (!r.group.is_aggregate) writef(out, "{}", r.group.simd);
  });
  column("wave_id", [](Out& out, const CsvRow& r) {
    if (!r.group.is_aggregate) writef(out, "{}", r.group.wave_id);
  });
  if (cfg.aggregate_wave)
    column("wave_count", [](Out& out, const CsvRow& r) {
      writef(out, "{}", r.record.wave_count);
    });

  column("target_addr", [](Out& out, const CsvRow& r) {
    writef(out, "0x{:x}", r.record.back_edge.target_addr);
  });
  column("source_addr", [](Out& out, const CsvRow& r) {
    writef(out, "0x{:x}", r.record.back_edge.source_addr);
  });
  column("code_object", [](Out& out, const CsvRow& r) {
    writef(out, "{}", r.record.back_edge.code_object_id);
  });
  column("iterations", [](Out& out, const CsvRow& r) {
    writef(out, "{}", r.record.stats.iteration_count);
  });
  column("total_duration", [](Out& out, const CsvRow& r) {
    writef(out, "{}", r.record.stats.total_duration);
  });
  column("total_stall", [](Out& out, const CsvRow& r) {
    writef(out, "{}", r.record.stats.total_stall);
  });
  column("total_idle", [](Out& out, const CsvRow& r) {
    writef(out, "{}", r.record.stats.total_idle);
  });
  column("loop_avg_dur", [](Out& out, const CsvRow& r) {
    writef(out, "{:.2f}", r.summary.avg_dur);
  });
  column("loop_avg_stall", [](Out& out, const CsvRow& r) {
    writef(out, "{:.2f}", r.summary.avg_stall);
  });
  column("loop_avg_idle", [](Out& out, const CsvRow& r) {
    writef(out, "{:.2f}", r.summary.avg_idle);
  });
  if (cfg.show_mfma_coexec)
    column("loop_avg_bubble", [](Out& out, const CsvRow& r) {
      writef(out, "{:.2f}", r.summary.avg_bubble);
    });
  column("mfma_count", [](Out& out, const CsvRow& r) {
    writef(out, "{}", r.summary.mfma_count);
  });
  column("mfma_time", [](Out& out, const CsvRow& r) {
    writef(out, "{}", r.summary.mfma_time);
  });
  column("mfma_util", [](Out& out, const CsvRow& r) {
    writef(out, "{:.4f}", r.summary.mfma_util);
  });
  if (!cfg.show_contents) return;

  column("inst_addr", [](Out& out, const CsvRow& r) {
    writef(out, "0x{:x}", r.inst->vaddr);
  });
  column("instruction", [](Out& out, const CsvRow& r) {
    writef(out, "\"{}\"", csv_escape(r.inst->text));
  });
  column("n", [](Out& out, const CsvRow& r) { writef(out, "{}", r.inst->n); });
  if (cfg.show_avg) {
    column("avg_dur", [](Out& out, const CsvRow& r) {
      write_per_execution(out, r, r.inst->dur);
    });
    column("avg_stall", [](Out& out, const CsvRow& r) {
      write_per_execution(out, r, r.inst->stall);
    });
    column("avg_idle", [](Out& out, const CsvRow& r) {
      write_per_execution(out, r, r.inst->idle);
    });
    column("avg_issue", [](Out& out, const CsvRow& r) {
      write_per_execution(out, r, r.inst->issue);
    });
    if (cfg.show_mfma_coexec)
      column("avg_bubble", [](Out& out, const CsvRow& r) {
        write_per_execution(out, r, r.inst->bubble);
      });
    column("avg_cycle", [](Out& out, const CsvRow& r) {
      write_per_execution(out, r, r.inst->rel_issue);
    });
  } else {
    column("dur",
           [](Out& out, const CsvRow& r) { writef(out, "{}", r.inst->dur); });
    column("stall",
           [](Out& out, const CsvRow& r) { writef(out, "{}", r.inst->stall); });
    column("idle",
           [](Out& out, const CsvRow& r) { writef(out, "{}", r.inst->idle); });
    column("issue",
           [](Out& out, const CsvRow& r) { writef(out, "{}", r.inst->issue); });
    if (cfg.show_mfma_coexec)
      column("bubble", [](Out& out, const CsvRow& r) {
        writef(out, "{}", r.inst->bubble);
      });
    column("rel_issue", [](Out& out, const CsvRow& r) {
      writef(out, "{}", r.inst->rel_issue);
    });
  }
}

void write_csv_header(LoopReportChunk& out, const LoopOutputConfig& cfg) {
  bool first = true;
  csv_columns(cfg, [&](std::string_view name, const auto&) {
    if (!first) out.push_back(',');
    first = false;
    writef(out, "{}", name);
  });
  out.push_back('\n');
}

void write_csv_row(LoopReportChunk& out, const LoopOutputConfig& cfg,
                   const CsvRow& row) {
  bool first = true;
  csv_columns(cfg, [&](std::string_view, const auto& value) {
    if (!first) out.push_back(',');
    first = false;
    value(out, row);
  });
  out.push_back('\n');
}

// A row per record or, under show_contents, per body instruction of each
// record.
void render_csv(LoopReportChunk& out, const LoopOutputConfig& cfg,
                const TraceLoopReport& report) {
  for (const auto& g : report.groups) {
    for (const auto& loop : g.loops) {
      CsvRow row{report, g, loop.record, loop.metrics.summary};
      if (!cfg.show_contents) {
        write_csv_row(out, cfg, row);
        continue;
      }
      for (const auto& inst : loop.metrics.rows) {
        row.inst = &inst;
        write_csv_row(out, cfg, row);
      }
    }
  }
}

// "DispatchID=N", qualified by the run when the output spans several.
void write_dispatch_label(LoopReportChunk& out, const LoopOutputConfig& cfg,
                          const TraceLoopReport& report) {
  if (cfg.show_prefix) writef(out, "Prefix={}, ", report.prefix);
  writef(out, "DispatchID={}", report.dispatch_id);
}

void render_loop_text_stats(LoopReportChunk& out,
                            const LoopRecordMetrics& metrics,
                            const LoopOutputConfig& cfg) {
  const auto& sum = metrics.summary;
  writef(out, "    avg/iter: dur={:.2f} stall={:.2f} idle={:.2f}", sum.avg_dur,
         sum.avg_stall, sum.avg_idle);
  if (cfg.show_mfma_coexec) writef(out, " bubble={:.2f}", sum.avg_bubble);
  writef(out, " mfma_time={} ({} mfma * {}) mfma_util={:.1f}%\n", sum.mfma_time,
         sum.mfma_count, kMfmaShadowCycles, sum.mfma_util * 100.0);
  if (!cfg.show_contents) return;

  writef(out, "    ");
  if (cfg.show_stats) {
    writef(out, "{:<6} ", "n");
    if (cfg.show_avg)
      writef(out, "{:<10} {:<10} {:<10} {:<10} ", "avg_dur", "avg_stall",
             "avg_idle", "avg_issue");
    else
      writef(out, "{:<10} {:<10} {:<10} {:<8} ", "dur", "stall", "idle",
             "issue");
    if (cfg.show_mfma_coexec)
      writef(out, "{:<10} ", cfg.show_avg ? "avg_bbl" : "bubble");
    writef(out, "{:<10} ", "avg_cycle");
  }
  if (cfg.show_addr) writef(out, "{:<14} ", "addr");
  writef(out, "instruction\n");

  uint64_t max_bubble = 0;
  if (cfg.color_mfma_bubble) {
    for (const auto& ii : metrics.rows)
      if (ii.bubble > max_bubble) max_bubble = ii.bubble;
  }
  for (const auto& ii : metrics.rows) {
    if (cfg.color_mfma_bubble && ii.has_stats)
      writef(out, "{}", bubble_color(ii, max_bubble));
    writef(out, "    ");
    if (cfg.show_stats && ii.has_stats) {
      writef(out, "{:<6} ", ii.n);
      if (cfg.show_avg) {
        writef(out, "{:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} ",
               static_cast<double>(ii.dur) / ii.n,
               static_cast<double>(ii.stall) / ii.n,
               static_cast<double>(ii.idle) / ii.n,
               static_cast<double>(ii.issue) / ii.n);
      } else {
        writef(out, "{:<10} {:<10} {:<10} {:<8} ", ii.dur, ii.stall, ii.idle,
               ii.issue);
      }
      if (cfg.show_mfma_coexec) {
        if (cfg.show_avg)
          writef(out, "{:<10.2f} ", static_cast<double>(ii.bubble) / ii.n);
        else
          writef(out, "{:<10} ", ii.bubble);
      }
      writef(out, "{:<10.2f} ", static_cast<double>(ii.rel_issue) / ii.n);
    } else if (cfg.show_stats) {
      if (cfg.show_avg)
        writef(out, "{:<6} {:<10} {:<10} {:<10} {:<10} ", "", "", "", "", "");
      else
        writef(out, "{:<6} {:<10} {:<10} {:<10} {:<8} ", "", "", "", "", "");
      if (cfg.show_mfma_coexec) writef(out, "{:<10} ", "");
      writef(out, "{:<10} ", "");
    }
    if (cfg.show_addr) writef(out, "0x{:x}:  ", ii.vaddr);
    writef(out, "{}", ii.text);
    if (cfg.color_mfma_bubble && ii.has_stats) writef(out, "\033[0m");
    writef(out, "\n");
  }
}

void render_text(LoopReportChunk& out, const LoopOutputConfig& cfg,
                 const TraceLoopReport& report) {
  for (const auto& g : report.groups) {
    writef(out, g.is_aggregate ? "=== Aggregate Across Waves (" : "=== Wave (");
    write_dispatch_label(out, cfg, report);
    if (g.is_aggregate)
      writef(out, ") ===\n");
    else
      writef(out, ", CU={}, SIMD={}, WaveID={}) ===\n", g.cu, g.simd,
             g.wave_id);
    for (const auto& loop : g.loops) {
      const auto& e = loop.record.back_edge;
      const auto& s = loop.record.stats;
      const auto& sum = loop.metrics.summary;
      // This line has always shown its averages as whole cycles, truncated;
      // the avg/iter line under it shows them to two decimals.
      const auto avg_dur = static_cast<uint64_t>(sum.avg_dur);
      const auto avg_stall = static_cast<uint64_t>(sum.avg_stall);
      if (g.is_aggregate)
        writef(out,
               "  Loop [0x{:x} -> 0x{:x}] code_object={}: "
               "{} waves, {} iterations, {} total cycles (avg {}/iter), "
               "stall {} (avg {}/iter), idle {}\n",
               e.target_addr, e.source_addr, e.code_object_id,
               loop.record.wave_count, s.iteration_count, s.total_duration,
               avg_dur, s.total_stall, avg_stall, s.total_idle);
      else
        writef(out,
               "  Loop [0x{:x} -> 0x{:x}] code_object={}: "
               "{} iterations, {} total cycles (avg {}/iter), "
               "stall {} (avg {}/iter), idle {}\n",
               e.target_addr, e.source_addr, e.code_object_id,
               s.iteration_count, s.total_duration, avg_dur, s.total_stall,
               avg_stall, s.total_idle);
      render_loop_text_stats(out, loop.metrics, cfg);
    }
  }

  // Text prints a heading per group, so a trace without groups printed
  // nothing, which would read like a failed run. A group without loops still
  // counts: --aggregate_wave prints the aggregate's heading even when no wave
  // has a loop, and a banner would contradict it.
  if (!report.groups.empty()) return;
  writef(out, "=== ");
  write_dispatch_label(out, cfg, report);
  writef(out, ", SE={}: no loops detected ({} waves, {} instructions",
         report.se_id, report.counts.waves, report.counts.insts);
  if (report.simd_filter >= 0)
    writef(out, "; {} waves on other SIMDs skipped by --simd={}",
           report.counts.waves_filtered, report.simd_filter);
  writef(out, ") ===\n");
}

}  // namespace

void render_loop_report(LoopReportChunk& out, const LoopOutputConfig& cfg,
                        const TraceLoopReport& report) {
  switch (cfg.format) {
    case LoopOutputFormat::Json:
      render_json(out, cfg, report);
      break;
    case LoopOutputFormat::Csv:
      render_csv(out, cfg, report);
      break;
    case LoopOutputFormat::Text:
      render_text(out, cfg, report);
      break;
  }
}

void write_loop_report_document(
    const LoopOutputConfig& cfg, llvm::ArrayRef<LoopReportChunk> chunks,
    llvm::function_ref<void(std::string_view)> write) {
  auto write_chunk = [&](const LoopReportChunk& chunk) {
    write(std::string_view(chunk.data(), chunk.size()));
  };
  if (cfg.format == LoopOutputFormat::Json) {
    write("[\n");
  } else if (cfg.format == LoopOutputFormat::Csv) {
    LoopReportChunk header;
    write_csv_header(header, cfg);
    write_chunk(header);
  }
  // JSON separates entries: a trace that printed none, whose chunk is empty,
  // gets no separator either.
  bool first_nonempty = true;
  for (const auto& chunk : chunks) {
    if (chunk.size() == 0) continue;
    if (cfg.format == LoopOutputFormat::Json && !first_nonempty) write(",\n");
    write_chunk(chunk);
    first_nonempty = false;
  }
  if (cfg.format == LoopOutputFormat::Json) write("\n]\n");
}

}  // namespace my_rocperf_tool
