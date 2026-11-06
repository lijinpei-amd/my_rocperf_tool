#include "my_rocperf_tool/inst_report.h"

#include "report_writer.h"

#include <algorithm>
#include <cassert>

namespace my_rocperf_tool {
namespace {

double per_execution(uint64_t total, uint64_t n) {
  return n ? static_cast<double>(total) / n : 0;
}

// The timing columns shared by CSV and JSON, in report order.
template <typename Column>
void timing_columns(const ReportConfig& cfg, Column&& column) {
  auto stat = [&](const char* sum, const char* avg,
                  uint64_t InstSums::* member) {
    column(cfg.show_avg ? avg : sum, member);
  };
  stat("dur", "avg_dur", &InstSums::dur);
  stat("stall", "avg_stall", &InstSums::stall);
  stat("idle", "avg_idle", &InstSums::idle);
  stat("issue", "avg_issue", &InstSums::issue);
  if (cfg.show_mfma_coexec) stat("bubble", "avg_bubble", &InstSums::bubble);
  stat("rel_issue", "avg_cycle", &InstSums::rel_issue);
}

struct CsvRow {
  const TraceInstReport& report;
  const InstGroup& group;
  const InstRow& inst;
};

// One column list produces both the header and every CSV row.
template <typename Column>
void csv_columns(const ReportConfig& cfg, Column&& column) {
  using Out = ReportChunk;
  if (cfg.aggregate != AggregateScope::Wave)
    column("record_type", [](Out& out, const CsvRow& r) {
      writef(out, "{}",
             r.group.scope == AggregateScope::All ? "aggregate" : "simd_slot");
    });
  if (cfg.show_prefix)
    column("prefix", [](Out& out, const CsvRow& r) {
      writef(out, "\"{}\"", csv_escape(r.report.prefix));
    });
  column("dispatch_id", [](Out& out, const CsvRow& r) {
    writef(out, "{}", r.report.dispatch_id);
  });
  column("se_id",
         [](Out& out, const CsvRow& r) { writef(out, "{}", r.report.se_id); });
  column("cu", [](Out& out, const CsvRow& r) {
    if (r.group.scope != AggregateScope::All)
      writef(out, "{}", r.group.slot.cu);
  });
  column("simd", [](Out& out, const CsvRow& r) {
    if (r.group.scope != AggregateScope::All)
      writef(out, "{}", r.group.slot.simd);
  });
  column("wave_id", [](Out& out, const CsvRow& r) {
    if (r.group.scope != AggregateScope::All)
      writef(out, "{}", r.group.slot.wave_id);
  });
  if (cfg.aggregate != AggregateScope::Wave)
    column("wave_count", [](Out& out, const CsvRow& r) {
      writef(out, "{}", r.group.summary.waves);
    });
  column("code_object", [](Out& out, const CsvRow& r) {
    writef(out, "{}", r.inst.pc.code_object_id);
  });
  column("inst_addr", [](Out& out, const CsvRow& r) {
    writef(out, "0x{:x}", r.inst.pc.address);
  });
  if (cfg.report == ReportKind::DynamicInst)
    column("occurrence", [](Out& out, const CsvRow& r) {
      writef(out, "{}", r.inst.occurrence);
    });
  column("instruction", [](Out& out, const CsvRow& r) {
    writef(out, "\"{}\"", csv_escape(r.inst.text));
  });
  column("n",
         [](Out& out, const CsvRow& r) { writef(out, "{}", r.inst.sums.n); });
  column("exec_waves", [](Out& out, const CsvRow& r) {
    writef(out, "{}", r.inst.sums.exec_waves);
  });
  timing_columns(cfg, [&](const char* name, uint64_t InstSums::* member) {
    column(name, [&](Out& out, const CsvRow& r) {
      if (cfg.show_avg)
        writef(out, "{:.2f}",
               per_execution(r.inst.sums.*member, r.inst.sums.n));
      else
        writef(out, "{}", r.inst.sums.*member);
    });
  });
}

void render_csv(ReportChunk& out, const ReportConfig& cfg,
                const TraceInstReport& report) {
  for (const auto& g : report.groups)
    for (const auto& row : g.rows) {
      bool first = true;
      csv_columns(cfg, [&](std::string_view, const auto& value) {
        if (!first) out.push_back(',');
        first = false;
        value(out, CsvRow{report, g, row});
      });
      out.push_back('\n');
    }
}

void render_json(ReportChunk& out, const ReportConfig& cfg,
                 const TraceInstReport& report) {
  bool first = true;
  for (const auto& g : report.groups) {
    if (!first) writef(out, ",\n");
    first = false;
    writef(out, "  {{");
    if (cfg.show_prefix)
      writef(out, "\"prefix\":\"{}\",", json_escape(report.prefix));
    writef(out, "\"dispatch_id\":{},\"se_id\":{}", report.dispatch_id,
           report.se_id);
    if (g.scope == AggregateScope::All)
      writef(out, ",\"aggregate\":true");
    else {
      if (g.scope == AggregateScope::SimdSlot)
        writef(out, ",\"simd_slot\":true");
      writef(out, ",\"cu\":{},\"simd\":{},\"wave_id\":{}", g.slot.cu,
             g.slot.simd, g.slot.wave_id);
    }
    if (g.scope != AggregateScope::Wave)
      writef(out, ",\"wave_count\":{}", g.summary.waves);
    const auto& s = g.summary;
    writef(out,
           ",\"summary\":{{\"waves\":{},\"avg_records\":{:.2f},\"avg_dur\":{:."
           "2f},\"avg_stall\":{:.2f},\"avg_idle\":{:.2f}",
           s.waves, s.avg_records, s.avg_dur, s.avg_stall, s.avg_idle);
    if (cfg.show_mfma_coexec)
      writef(out, ",\"avg_bubble\":{:.2f}", s.avg_bubble);
    writef(out,
           ",\"avg_mfma\":{:.2f},\"mfma_util\":{:.4f}}},\"instructions\":[\n",
           s.avg_mfma, s.mfma_util);
    bool first_row = true;
    for (const auto& row : g.rows) {
      if (!first_row) writef(out, ",\n");
      first_row = false;
      writef(out, "    {{\"code_object\":{},\"addr\":\"0x{:x}\"",
             row.pc.code_object_id, row.pc.address);
      if (cfg.report == ReportKind::DynamicInst)
        writef(out, ",\"occurrence\":{}", row.occurrence);
      writef(out, ",\"inst\":\"{}\"", json_escape(row.text));
      if (cfg.show_stats) {
        writef(out, ",\"n\":{},\"exec_waves\":{}", row.sums.n,
               row.sums.exec_waves);
        timing_columns(cfg, [&](const char* name, uint64_t InstSums::* member) {
          if (cfg.show_avg)
            writef(out, ",\"{}\":{:.2f}", name,
                   per_execution(row.sums.*member, row.sums.n));
          else
            writef(out, ",\"{}\":{}", name, row.sums.*member);
        });
      }
      writef(out, "}}");
    }
    writef(out, "\n  ]}}");
  }
}

void render_text_rows(ReportChunk& out, const ReportConfig& cfg,
                      const InstGroup& g) {
  writef(out, "    ");
  if (cfg.show_stats) {
    writef(out, "{:<6} {:<6} ", "n", "waves");
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
  if (cfg.report == ReportKind::DynamicInst) writef(out, "{:<5} ", "occ");
  if (cfg.show_addr) writef(out, "{:<14} ", "addr");
  writef(out, "instruction\n");
  uint64_t max_bubble = 0;
  if (cfg.color_mfma_bubble)
    for (const auto& row : g.rows)
      max_bubble = std::max(max_bubble, row.sums.bubble);
  for (const auto& row : g.rows) {
    const auto& s = row.sums;
    if (cfg.color_mfma_bubble)
      writef(out, "{}", bubble_color(s.bubble, max_bubble));
    writef(out, "    ");
    if (cfg.show_stats) {
      writef(out, "{:<6} {:<6} ", s.n, s.exec_waves);
      if (cfg.show_avg)
        writef(out, "{:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} ",
               per_execution(s.dur, s.n), per_execution(s.stall, s.n),
               per_execution(s.idle, s.n), per_execution(s.issue, s.n));
      else
        writef(out, "{:<10} {:<10} {:<10} {:<8} ", s.dur, s.stall, s.idle,
               s.issue);
      if (cfg.show_mfma_coexec) {
        if (cfg.show_avg)
          writef(out, "{:<10.2f} ", per_execution(s.bubble, s.n));
        else
          writef(out, "{:<10} ", s.bubble);
      }
      writef(out, "{:<10.2f} ", per_execution(s.rel_issue, s.n));
    }
    if (cfg.report == ReportKind::DynamicInst)
      writef(out, "{:<5} ", row.occurrence);
    if (cfg.show_addr) writef(out, "0x{:x}:  ", row.pc.address);
    writef(out, "{}", row.text);
    if (cfg.color_mfma_bubble) writef(out, "\033[0m");
    writef(out, "\n");
  }
}

void render_text(ReportChunk& out, const ReportConfig& cfg,
                 const TraceInstReport& report) {
  for (const auto& g : report.groups) {
    if (g.scope == AggregateScope::All)
      writef(out, "=== Aggregate Across Waves (");
    else if (g.scope == AggregateScope::SimdSlot)
      writef(out, "=== SIMD Slot (");
    else
      writef(out, "=== Wave (");
    write_dispatch_label(out, cfg, report.prefix, report.dispatch_id);
    writef(out, ", SE={}", report.se_id);
    if (g.scope != AggregateScope::All)
      writef(out, ", CU={}, SIMD={}, WaveID={}", g.slot.cu, g.slot.simd,
             g.slot.wave_id);
    if (g.scope != AggregateScope::Wave)
      writef(out, ", {} waves", g.summary.waves);
    writef(out, ") ===\n");
    const auto& s = g.summary;
    writef(out,
           "  avg/wave: records={:.2f} dur={:.2f} stall={:.2f} idle={:.2f}",
           s.avg_records, s.avg_dur, s.avg_stall, s.avg_idle);
    if (cfg.show_mfma_coexec) writef(out, " bubble={:.2f}", s.avg_bubble);
    writef(out, " mfma={:.2f} mfma_util={:.1f}%\n", s.avg_mfma,
           s.mfma_util * 100.0);
    render_text_rows(out, cfg, g);
  }
  if (!report.groups.empty()) return;
  writef(out, "=== ");
  write_dispatch_label(out, cfg, report.prefix, report.dispatch_id);
  writef(out, ", SE={}: no waves analyzed", report.se_id);
  if (report.simd_filter >= 0)
    writef(out, " ({} waves on other SIMDs skipped by --simd={})",
           report.counts.waves_filtered, report.simd_filter);
  writef(out, " ===\n");
}

}  // namespace

void write_inst_csv_header(ReportChunk& out, const ReportConfig& cfg) {
  bool first = true;
  csv_columns(cfg, [&](std::string_view name, const auto&) {
    if (!first) out.push_back(',');
    first = false;
    writef(out, "{}", name);
  });
  out.push_back('\n');
}

void render_inst_report(ReportChunk& out, const ReportConfig& cfg,
                        const TraceInstReport& report) {
  assert((cfg.report == ReportKind::StaticInst ||
          cfg.report == ReportKind::DynamicInst) &&
         "not an instruction report");
  switch (cfg.format) {
    case ReportFormat::Text:
      render_text(out, cfg, report);
      break;
    case ReportFormat::Csv:
      render_csv(out, cfg, report);
      break;
    case ReportFormat::Json:
      render_json(out, cfg, report);
      break;
  }
}

}  // namespace my_rocperf_tool
