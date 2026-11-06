#include "my_rocperf_tool/loop_report.h"

#include "my_rocperf_tool/loop_profile.h"
#include "report_writer.h"

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

void render_loop_json(ReportChunk& out, const ReportedLoop& loop,
                      const ReportConfig& cfg,
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
  if ((cfg.report == ReportKind::LoopDetail)) {
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
// write_report_document joins the chunks and brackets the document.
void render_json(ReportChunk& out, const ReportConfig& cfg,
                 const TraceLoopReport& report) {
  bool first_entry = true;
  for (const auto& g : report.groups) {
    if (!first_entry) writef(out, ",\n");
    first_entry = false;
    writef(out, "  {{");
    if (cfg.show_prefix)
      writef(out, "\"prefix\":\"{}\",", json_escape(report.prefix));
    if ((g.scope == AggregateScope::All))
      writef(out, "\"dispatch_id\":{},\"aggregate\":true,\"loops\":[\n",
             report.dispatch_id);
    else {
      if (g.scope == AggregateScope::SimdSlot)
        writef(out, "\"simd_slot\":true,");
      writef(out,
             "\"dispatch_id\":{},\"cu\":{},\"simd\":{},\"wave_id\":{},"
             "\"loops\":[\n",
             report.dispatch_id, g.slot.cu, g.slot.simd, g.slot.wave_id);
    }
    bool first_loop = true;
    for (const auto& loop : g.loops) {
      if (!first_loop) writef(out, ",\n");
      first_loop = false;
      auto wave_count = (g.scope != AggregateScope::Wave)
                            ? std::optional<uint32_t>(loop.record.wave_count)
                            : std::nullopt;
      render_loop_json(out, loop, cfg, wave_count);
    }
    writef(out, "\n  ]}}");
  }
}

// What one CSV row is about: a loop record and, under LoopDetail, one of
// its body instructions.
struct CsvRow {
  const TraceLoopReport& report;
  const LoopGroup& group;
  const LoopRecord& record;
  const LoopSummary& summary;
  const LoopInstRow* inst = nullptr;  // null unless LoopDetail
};

// `total` per execution of the row's instruction. An instruction never
// executed has no average, and its field stays blank.
void write_per_execution(ReportChunk& out, const CsvRow& row, uint64_t total) {
  if (row.inst->n > 0)
    writef(out, "{:.2f}", static_cast<double>(total) / row.inst->n);
}

// The CSV columns, in order: the one definition of both the header and every
// row, so that the two cannot disagree on a column. Calls column(name, value)
// for each column `cfg` shows, where value(out, row) appends that column's
// field of `row`, or nothing for a blank one.
template <typename Column>
void csv_columns(const ReportConfig& cfg, Column&& column) {
  using Out = ReportChunk;
  if ((cfg.aggregate != AggregateScope::Wave))
    column("record_type", [](Out& out, const CsvRow& r) {
      writef(
          out, "{}",
          (r.group.scope == AggregateScope::All) ? "aggregate" : "simd_slot");
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
    if (!(r.group.scope == AggregateScope::All))
      writef(out, "{}", r.group.slot.cu);
  });
  column("simd", [](Out& out, const CsvRow& r) {
    if (!(r.group.scope == AggregateScope::All))
      writef(out, "{}", r.group.slot.simd);
  });
  column("wave_id", [](Out& out, const CsvRow& r) {
    if (!(r.group.scope == AggregateScope::All))
      writef(out, "{}", r.group.slot.wave_id);
  });
  if ((cfg.aggregate != AggregateScope::Wave))
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
  if (!(cfg.report == ReportKind::LoopDetail)) return;

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

void write_csv_header(ReportChunk& out, const ReportConfig& cfg) {
  bool first = true;
  csv_columns(cfg, [&](std::string_view name, const auto&) {
    if (!first) out.push_back(',');
    first = false;
    writef(out, "{}", name);
  });
  out.push_back('\n');
}

void write_csv_row(ReportChunk& out, const ReportConfig& cfg,
                   const CsvRow& row) {
  bool first = true;
  csv_columns(cfg, [&](std::string_view, const auto& value) {
    if (!first) out.push_back(',');
    first = false;
    value(out, row);
  });
  out.push_back('\n');
}

// A row per record or, under LoopDetail, per body instruction of each
// record.
void render_csv(ReportChunk& out, const ReportConfig& cfg,
                const TraceLoopReport& report) {
  for (const auto& g : report.groups) {
    for (const auto& loop : g.loops) {
      CsvRow row{report, g, loop.record, loop.metrics.summary};
      if (!(cfg.report == ReportKind::LoopDetail)) {
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

void render_loop_text_stats(ReportChunk& out, const LoopRecordMetrics& metrics,
                            const ReportConfig& cfg) {
  const auto& sum = metrics.summary;
  writef(out, "    avg/iter: dur={:.2f} stall={:.2f} idle={:.2f}", sum.avg_dur,
         sum.avg_stall, sum.avg_idle);
  if (cfg.show_mfma_coexec) writef(out, " bubble={:.2f}", sum.avg_bubble);
  writef(out, " mfma_time={} ({} mfma * {}) mfma_util={:.1f}%\n", sum.mfma_time,
         sum.mfma_count, kMfmaShadowCycles, sum.mfma_util * 100.0);
  if (!(cfg.report == ReportKind::LoopDetail)) return;

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
      writef(out, "{}", bubble_color(ii.bubble, max_bubble));
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

void render_text(ReportChunk& out, const ReportConfig& cfg,
                 const TraceLoopReport& report) {
  for (const auto& g : report.groups) {
    if (g.scope == AggregateScope::All)
      writef(out, "=== Aggregate Across Waves (");
    else if (g.scope == AggregateScope::SimdSlot)
      writef(out, "=== SIMD Slot (");
    else
      writef(out, "=== Wave (");
    write_dispatch_label(out, cfg, report.prefix, report.dispatch_id);
    if ((g.scope == AggregateScope::All))
      writef(out, ") ===\n");
    else
      writef(out, ", CU={}, SIMD={}, WaveID={}) ===\n", g.slot.cu, g.slot.simd,
             g.slot.wave_id);
    for (const auto& loop : g.loops) {
      const auto& e = loop.record.back_edge;
      const auto& s = loop.record.stats;
      const auto& sum = loop.metrics.summary;
      // This line has always shown its averages as whole cycles, truncated;
      // the avg/iter line under it shows them to two decimals.
      const auto avg_dur = static_cast<uint64_t>(sum.avg_dur);
      const auto avg_stall = static_cast<uint64_t>(sum.avg_stall);
      if (g.scope != AggregateScope::Wave)
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
  // counts: --aggregate=all prints the aggregate's heading even when no wave
  // has a loop, and a banner would contradict it.
  if (!report.groups.empty()) return;
  writef(out, "=== ");
  write_dispatch_label(out, cfg, report.prefix, report.dispatch_id);
  writef(out, ", SE={}: no loops detected ({} waves, {} instructions",
         report.se_id, report.counts.waves, report.counts.insts);
  if (report.simd_filter >= 0)
    writef(out, "; {} waves on other SIMDs skipped by --simd={}",
           report.counts.waves_filtered, report.simd_filter);
  writef(out, ") ===\n");
}

}  // namespace

void write_loop_csv_header(ReportChunk& out, const ReportConfig& cfg) {
  write_csv_header(out, cfg);
}

void render_loop_report(ReportChunk& out, const ReportConfig& cfg,
                        const TraceLoopReport& report) {
  switch (cfg.format) {
    case ReportFormat::Json:
      render_json(out, cfg, report);
      break;
    case ReportFormat::Csv:
      render_csv(out, cfg, report);
      break;
    case ReportFormat::Text:
      render_text(out, cfg, report);
      break;
  }
}

}  // namespace my_rocperf_tool
