#include "my_rocperf_tool/att_output_dir.h"
#include "my_rocperf_tool/disassembler.h"
#include "my_rocperf_tool/init_llvm.h"
#include "my_rocperf_tool/loop_detection.h"
#include "my_rocperf_tool/loop_profile.h"
#include "my_rocperf_tool/wave_analysis.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "trace_decoder_api.h"
#include "trace_decoder_types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include "fmt/base.h"
#include "fmt/format.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

// Phase timing rides on LLVM's -ftime-trace machinery, already linked via
// libLLVM. Printing one line per scope was fine for the serial phases but not
// for the five inside run_trace: those run once per trace on up to 64 workers,
// so the interesting numbers are per-phase totals, not thousands of interleaved
// lines. TimeProfiler aggregates same-named scopes into "Total <name>" entries
// and keeps the nesting, viewable in perfetto.dev / speedscope.
#ifdef ENABLE_TIMERS
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"

ABSL_FLAG(std::string, time_trace_file, "",
          "path for the Chrome Trace Event JSON emitted by ENABLE_TIMERS "
          "builds; empty writes ./load_issue_stall.time-trace");

// 0 keeps every scope in the event list. Totals are accumulated before this
// threshold is applied, so raising it only thins the flame chart.
static constexpr unsigned kTimeTraceGranularityUs = 0;

#define PHASE_TIMER_CONCAT_(a, b) a##b
#define PHASE_TIMER_CONCAT(a, b) PHASE_TIMER_CONCAT_(a, b)
#define PHASE_TIMER(name) \
  llvm::TimeTraceScope PHASE_TIMER_CONCAT(_phase_timer_, __LINE__)(name)

// The profiler instance is per-thread: a worker's scopes are silently dropped
// unless it initializes one, and they reach the output only if it hands the
// instance back before the main thread writes. Declare this first in a thread
// body so it outlives everything it is meant to time.
struct WorkerPhaseTimers {
  WorkerPhaseTimers() {
    llvm::timeTraceProfilerInitialize(kTimeTraceGranularityUs, "worker");
  }
  ~WorkerPhaseTimers() { llvm::timeTraceProfilerFinishThread(); }
};
#define WORKER_PHASE_TIMERS() WorkerPhaseTimers _worker_phase_timers

// Owns the main thread's instance. The write asserts that every scope has
// closed, so this has to outlive run_main's TOTAL scope -- hence main(), not
// run_main().
struct PhaseTimerSession {
  PhaseTimerSession() {
    llvm::timeTraceProfilerInitialize(kTimeTraceGranularityUs,
                                      "load_issue_stall");
  }
  ~PhaseTimerSession() {
    auto path = absl::GetFlag(FLAGS_time_trace_file);
    if (auto err = llvm::timeTraceProfilerWrite(path, "load_issue_stall"))
      llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), "[TIMER] ");
    else
      fmt::print(stderr, "[TIMER] wrote {}\n",
                 path.empty() ? "load_issue_stall.time-trace" : path);
    llvm::timeTraceProfilerCleanup();
  }
};
#define PHASE_TIMER_SESSION() PhaseTimerSession _phase_timer_session
#else
#define PHASE_TIMER(name) ((void)0)
#define WORKER_PHASE_TIMERS() ((void)0)
#define PHASE_TIMER_SESSION() ((void)0)
#endif

ABSL_FLAG(std::optional<std::string>, att_output_dir, std::nullopt,
          "output file dir");
ABSL_FLAG(bool, detect_loops, true, "detect and report loops in ATT traces");
ABSL_FLAG(bool, dump_loop_contents, false,
          "disassemble instructions inside each detected loop");
ABSL_FLAG(bool, loop_inst_stats, true,
          "show per-instruction stats column in loop contents");
ABSL_FLAG(bool, loop_inst_addr, true,
          "show instruction address column in loop contents");
ABSL_FLAG(bool, loop_inst_avg, false,
          "show average per-execution stats in loop contents");
ABSL_FLAG(std::string, loop_output_format, "text",
          "output format for loop data: text, csv, json");
ABSL_FLAG(bool, aggregate_wave, false,
          "show loop statistics aggregated across waves");
ABSL_FLAG(bool, aggregate_wave_only, false,
          "show only loop statistics aggregated across waves");
ABSL_FLAG(std::string, verify_trace_duration, "error",
          "check that each trace record's duration >= stall: off; warn (print "
          "a WARNING per offending record and continue); error (print one "
          "error per offending trace with its count and first offending "
          "record, emit nothing for that trace, and exit 1 once the other "
          "traces are done). true and false are accepted as warn and off");
ABSL_FLAG(bool, mfma_coexec, false,
          "analyze MFMA shadow coverage for loop instructions");
ABSL_FLAG(bool, color_mfma_bubble, false,
          "colorize instructions by bubble severity (text output only)");
ABSL_FLAG(bool, decode_all_insts, false,
          "decode every instruction in all code objects' .text sections "
          "(slow for large binaries)");
ABSL_FLAG(uint32_t, jobs, 0,
          "number of parallel decode workers; 0 = auto (min of trace count, "
          "hardware concurrency, and 64)");
ABSL_FLAG(int32_t, simd, -1,
          "analyze only waves on this SIMD ([0,3]); -1 analyzes all SIMDs");

enum class LoopOutputFormat { Text, Csv, Json };

// true and false were the values of the boolean flag this replaced; they stay
// accepted so existing invocations keep working.
std::optional<my_rocperf_tool::DurationCheck> parse_duration_check(
    const std::string& text) {
  using my_rocperf_tool::DurationCheck;
  if (text == "off" || text == "false") return DurationCheck::Off;
  if (text == "warn" || text == "true") return DurationCheck::Warn;
  if (text == "error") return DurationCheck::Error;
  return std::nullopt;
}

struct LoopOutputConfig {
  LoopOutputFormat format = LoopOutputFormat::Text;
  bool show_contents = false;
  bool show_stats = true;
  bool show_addr = true;
  bool show_avg = false;
  bool show_mfma_coexec = false;
  bool color_mfma_bubble = false;
  bool aggregate_wave = false;
  bool aggregate_wave_only = false;
  // Set when the directory holds more than one run (output-file prefix):
  // dispatch ids repeat across runs, so every record also names its prefix.
  // Off for a single run, which keeps that output unchanged.
  bool show_prefix = false;

  static LoopOutputConfig from_flags() {
    LoopOutputConfig cfg;
    cfg.show_contents = absl::GetFlag(FLAGS_dump_loop_contents);
    cfg.show_stats = absl::GetFlag(FLAGS_loop_inst_stats);
    cfg.show_addr = absl::GetFlag(FLAGS_loop_inst_addr);
    cfg.show_avg = absl::GetFlag(FLAGS_loop_inst_avg);
    cfg.show_mfma_coexec = absl::GetFlag(FLAGS_mfma_coexec);
    cfg.color_mfma_bubble = absl::GetFlag(FLAGS_color_mfma_bubble);
    cfg.aggregate_wave = absl::GetFlag(FLAGS_aggregate_wave);
    cfg.aggregate_wave_only = absl::GetFlag(FLAGS_aggregate_wave_only);
    if (cfg.aggregate_wave_only) cfg.aggregate_wave = true;
    auto fmt = absl::GetFlag(FLAGS_loop_output_format);
    if (fmt == "json")
      cfg.format = LoopOutputFormat::Json;
    else if (fmt == "csv")
      cfg.format = LoopOutputFormat::Csv;
    else
      cfg.format = LoopOutputFormat::Text;
    return cfg;
  }
};

// What the decoder's callbacks share through their void* userdata while one
// trace decodes: the trace, handed over on the first se_data call; the run's
// code objects, which answer ISA queries; and the analysis each decoded wave
// is forwarded to.
struct trace_decoder_context {
  bool first_run = true;
  std::unique_ptr<llvm::MemoryBuffer> att_data;
  const my_rocperf_tool::Disassembler& disas;
  my_rocperf_tool::WaveAnalysis& analysis;
  // The first code object id the decoder asked about that the run lacks.
  std::optional<uint64_t> unknown_code_object_id;

  trace_decoder_context(const my_rocperf_tool::Disassembler& disas,
                        my_rocperf_tool::WaveAnalysis& analysis)
      : disas(disas), analysis(analysis) {}
};

std::optional<uint64_t> parse_dispatch_id_from_path(const std::string& path) {
  auto name = std::filesystem::path(path).filename().string();
  const std::string marker = "_dispatch_";
  auto marker_pos = name.rfind(marker);
  if (marker_pos == std::string::npos) return std::nullopt;
  auto digit_pos = marker_pos + marker.size();
  if (digit_pos >= name.size() ||
      !std::isdigit(static_cast<unsigned char>(name[digit_pos])))
    return std::nullopt;
  return std::stoull(name.substr(digit_pos));
}

uint64_t att_decoder_se_data_callback(uint8_t** buffer, uint64_t* buffer_size,
                                      void* userdata) {
  auto& ctx = *reinterpret_cast<trace_decoder_context*>(userdata);
  if (!ctx.first_run) {
    return 0;
  }
  ctx.first_run = false;
  uint64_t data_size = ctx.att_data->getBufferSize();
  *buffer_size = data_size;
  // The decoder only reads the trace through this pointer (it parses a
  // const uint8_t*); its callback type is just not const-qualified.
  *buffer = reinterpret_cast<uint8_t*>(
      const_cast<char*>(ctx.att_data->getBufferStart()));
  return data_size;
}

rocprofiler_thread_trace_decoder_status_t att_decoder_isa_callback(
    char* isa_instruction, uint64_t* isa_memory_size, uint64_t* isa_size,
    rocprofiler_thread_trace_decoder_pc_t pc, void* userdata) {
  assert(pc.code_object_id != 0);
  auto& ctx = *reinterpret_cast<trace_decoder_context*>(userdata);
  const auto* obj = ctx.disas.find_code_object(pc.code_object_id);
  if (!obj) {
    // The decoder takes an error as "cannot stitch this PC" and decodes on,
    // so run_trace fails the trace once the decode returns.
    if (!ctx.unknown_code_object_id)
      ctx.unknown_code_object_id = pc.code_object_id;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;
  }
  size_t text_size = obj->copy_text_at(
      pc.address, llvm::MutableArrayRef<char>(isa_instruction, *isa_size));
  if (*isa_size < text_size) {
    *isa_size = text_size;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  *isa_size = text_size;
  *isa_memory_size = obj->size_at(pc.address);
  return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

rocprofiler_thread_trace_decoder_status_t att_decoder_trace_callback(
    rocprofiler_thread_trace_decoder_record_type_t record_type_id,
    void* trace_events, uint64_t trace_size, void* userdata) {
  auto& ctx = *reinterpret_cast<trace_decoder_context*>(userdata);
  switch (record_type_id) {
    default:
      break;
    case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE:
      const auto* waves =
          static_cast<const rocprofiler_thread_trace_decoder_wave_t*>(
              trace_events);
      for (size_t wave_n = 0; wave_n < trace_size; wave_n++)
        ctx.analysis.add_wave(waves[wave_n]);
  }
  return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

// A trace's loop bodies, each listed once: listing prints every instruction,
// and every record of a loop -- one per wave that has it, and the aggregate --
// needs the loop's body.
class LoopBodies {
 public:
  explicit LoopBodies(const my_rocperf_tool::Disassembler& disas)
      : disas(disas) {}

  const my_rocperf_tool::LoopBody& of(const my_rocperf_tool::BackEdge& edge) {
    auto [it, inserted] = listed.try_emplace(edge);
    auto& body = it->second;
    if (!inserted) return body;
    const auto* obj = disas.find_code_object(edge.code_object_id);
    if (!obj) return body;
    for (uint64_t vaddr : obj->inst_vaddrs(edge.target_addr, edge.source_addr))
      body.push_back({vaddr, obj->text_at(vaddr), obj->kind_at(vaddr)});
    return body;
  }

 private:
  const my_rocperf_tool::Disassembler& disas;
  llvm::DenseMap<my_rocperf_tool::BackEdge, my_rocperf_tool::LoopBody> listed;
};

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
const char* bubble_color(const my_rocperf_tool::LoopInstRow& ii,
                         uint64_t max_bubble) {
  if (!ii.has_stats || ii.n == 0 || ii.bubble == 0 || max_bubble == 0)
    return "\033[32m";  // green: fully covered
  double ratio = static_cast<double>(ii.bubble) / max_bubble;
  if (ratio <= 0.25) return "\033[33m";  // yellow: low
  if (ratio <= 0.75) return "\033[31m";  // red: medium
  return "\033[1;31m";                   // bright red: high
}

std::vector<my_rocperf_tool::LoopRecord> aggregate_loops_by_back_edge(
    const std::vector<my_rocperf_tool::WaveLoopInfo>& wave_loops) {
  std::vector<my_rocperf_tool::LoopRecord> result;
  llvm::DenseMap<my_rocperf_tool::BackEdge, size_t> index_by_edge;
  for (const auto& wl : wave_loops) {
    for (const auto& dl : wl.loops) {
      const auto& edge = dl.back_edge;
      auto it = index_by_edge.find(edge);
      if (it == index_by_edge.end()) {
        it = index_by_edge.try_emplace(edge, result.size()).first;
        result.push_back({edge, {}, 0, std::nullopt});
      }

      auto& agg = result[it->second];
      agg.wave_count++;
      agg.stats += dl.stats;
    }
  }
  std::sort(result.begin(), result.end(),
            [](const my_rocperf_tool::LoopRecord& a,
               const my_rocperf_tool::LoopRecord& b) {
              return std::tie(a.back_edge.code_object_id,
                              a.back_edge.target_addr,
                              a.back_edge.source_addr) <
                     std::tie(b.back_edge.code_object_id,
                              b.back_edge.target_addr, b.back_edge.source_addr);
            });
  return result;
}

struct LoopGroup {
  bool is_aggregate;
  uint8_t cu = 0;        // valid only when !is_aggregate
  uint8_t simd = 0;      // valid only when !is_aggregate
  uint32_t wave_id = 0;  // valid only when !is_aggregate
  std::vector<my_rocperf_tool::LoopRecord> loops;
};

std::vector<LoopGroup> build_loop_groups(
    const std::vector<my_rocperf_tool::WaveLoopInfo>& wave_loops,
    const LoopOutputConfig& cfg) {
  std::vector<LoopGroup> groups;
  if (!cfg.aggregate_wave_only) {
    for (const auto& wl : wave_loops) {
      LoopGroup g{false, wl.cu, wl.simd, wl.wave_id, {}};
      g.loops.reserve(wl.loops.size());
      for (const auto& dl : wl.loops)
        g.loops.push_back({dl.back_edge, dl.stats, 1, wl.wave_index});
      groups.push_back(std::move(g));
    }
  }
  if (cfg.aggregate_wave) {
    LoopGroup g{true, 0, 0, 0, aggregate_loops_by_back_edge(wave_loops)};
    groups.push_back(std::move(g));
  }
  return groups;
}

// Per-trace output buffer type. Lower SBO than fmt::memory_buffer's default
// (500): aggregate_wave_only output fits in <256 B; reduces the upfront
// allocation of std::vector<TraceBuf>(n_traces) by ~half.
using TraceBuf = fmt::basic_memory_buffer<char, 256>;

// Small wrapper around fmt::format_to so call sites stay terse.
// fmt::format_string gives compile-time format-string checks; fmt::appender
// is fmt's canonical writer for memory_buffer (tighter codegen than
// std::back_inserter).
template <typename... T>
inline void writef(TraceBuf& out, fmt::format_string<T...> f, T&&... args) {
  fmt::format_to(fmt::appender(out), f, std::forward<T>(args)...);
}

// For stderr text, which is rare enough that how it is appended does not
// matter.
template <typename... T>
inline void writef(std::string& out, fmt::format_string<T...> f, T&&... args) {
  fmt::format_to(std::back_inserter(out), f, std::forward<T>(args)...);
}

void dump_loop_json(TraceBuf& out, const my_rocperf_tool::BackEdge& e,
                    const my_rocperf_tool::LoopStats& s,
                    const my_rocperf_tool::LoopRecordMetrics& metrics,
                    const LoopOutputConfig& cfg,
                    std::optional<uint32_t> wave_count = std::nullopt) {
  const auto& sum = metrics.summary;
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
    for (const auto& ii : metrics.rows) {
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

void dump_csv_header(TraceBuf& out, const LoopOutputConfig& cfg) {
  if (cfg.aggregate_wave) writef(out, "record_type,");
  if (cfg.show_prefix) writef(out, "prefix,");
  writef(out, "dispatch_id,cu,simd,wave_id,");
  if (cfg.aggregate_wave) writef(out, "wave_count,");
  writef(out,
         "target_addr,source_addr,code_object,"
         "iterations,total_duration,total_stall,total_idle,"
         "loop_avg_dur,loop_avg_stall,loop_avg_idle,");
  if (cfg.show_mfma_coexec) writef(out, "loop_avg_bubble,");
  writef(out, "mfma_count,mfma_time,mfma_util");
  if (cfg.show_contents) {
    writef(out, ",inst_addr,instruction,n,");
    if (cfg.show_avg) {
      writef(out, "avg_dur,avg_stall,avg_idle,avg_issue");
      if (cfg.show_mfma_coexec) writef(out, ",avg_bubble");
      writef(out, ",avg_cycle");
    } else {
      writef(out, "dur,stall,idle,issue");
      if (cfg.show_mfma_coexec) writef(out, ",bubble");
      writef(out, ",rel_issue");
    }
  }
  writef(out, "\n");
}

void dump_csv_row_prefix(TraceBuf& out, const LoopOutputConfig& cfg,
                         const std::string& prefix, uint64_t dispatch_id,
                         const LoopGroup& g, uint32_t wave_count) {
  if (cfg.aggregate_wave)
    writef(out, "{},", g.is_aggregate ? "aggregate" : "wave");
  if (cfg.show_prefix) writef(out, "\"{}\",", csv_escape(prefix));
  writef(out, "{},", dispatch_id);
  if (cfg.aggregate_wave && g.is_aggregate)
    writef(out, ",,,");
  else
    writef(out, "{},{},{},", g.cu, g.simd, g.wave_id);
  if (cfg.aggregate_wave) writef(out, "{},", wave_count);
}

void dump_csv_loop_rows(TraceBuf& out, const LoopOutputConfig& cfg,
                        const std::string& prefix, uint64_t dispatch_id,
                        const LoopGroup& g,
                        const my_rocperf_tool::LoopRecord& lr,
                        const my_rocperf_tool::LoopRecordMetrics& metrics) {
  const auto& e = lr.back_edge;
  const auto& s = lr.stats;
  const auto& sum = metrics.summary;
  auto print_loop_fields = [&]() {
    writef(out, "0x{:x},0x{:x},{},{},{},{},{},{:.2f},{:.2f},{:.2f},",
           e.target_addr, e.source_addr, e.code_object_id, s.iteration_count,
           s.total_duration, s.total_stall, s.total_idle, sum.avg_dur,
           sum.avg_stall, sum.avg_idle);
    if (cfg.show_mfma_coexec) writef(out, "{:.2f},", sum.avg_bubble);
    writef(out, "{},{},{:.4f}", sum.mfma_count, sum.mfma_time, sum.mfma_util);
  };

  if (!cfg.show_contents) {
    dump_csv_row_prefix(out, cfg, prefix, dispatch_id, g, lr.wave_count);
    print_loop_fields();
    writef(out, "\n");
    return;
  }

  for (const auto& ii : metrics.rows) {
    dump_csv_row_prefix(out, cfg, prefix, dispatch_id, g, lr.wave_count);
    print_loop_fields();
    writef(out, ",0x{:x},\"{}\",{},", ii.vaddr, csv_escape(ii.text), ii.n);
    if (cfg.show_avg) {
      if (ii.n > 0) {
        writef(out, "{:.2f},{:.2f},{:.2f},{:.2f}",
               static_cast<double>(ii.dur) / ii.n,
               static_cast<double>(ii.stall) / ii.n,
               static_cast<double>(ii.idle) / ii.n,
               static_cast<double>(ii.issue) / ii.n);
        if (cfg.show_mfma_coexec)
          writef(out, ",{:.2f}", static_cast<double>(ii.bubble) / ii.n);
        writef(out, ",{:.2f}", static_cast<double>(ii.rel_issue) / ii.n);
      } else {
        // Same shape as the branch above with every field empty, so the row
        // keeps the header's column count.
        writef(out, ",,,");
        if (cfg.show_mfma_coexec) writef(out, ",");
        writef(out, ",");
      }
      writef(out, "\n");
    } else {
      writef(out, "{},{},{},{}", ii.dur, ii.stall, ii.idle, ii.issue);
      if (cfg.show_mfma_coexec) writef(out, ",{}", ii.bubble);
      writef(out, ",{}\n", ii.rel_issue);
    }
  }
}

// Emits per-trace dispatch entries separated by ",\n" with no leading or
// trailing comma. Caller is responsible for the surrounding "[\n" / "\n]\n"
// and for inserting ",\n" between non-empty per-trace bodies.
// Per-trace dump state, threaded through dump_loops*.  Holds references —
// owned by run_trace and its analysis result.
struct DumpCtx {
  TraceBuf& out;
  const my_rocperf_tool::LoopProfile& profile;
  LoopBodies& bodies;
  const LoopOutputConfig& cfg;
  const std::string& prefix;
  uint64_t dispatch_id;

  my_rocperf_tool::LoopRecordMetrics metrics(
      const my_rocperf_tool::LoopRecord& lr) const {
    return profile.metrics(lr, bodies.of(lr.back_edge));
  }
};

// "DispatchID=N", qualified by the run when the output spans several.
void write_dispatch_label(TraceBuf& out, const LoopOutputConfig& cfg,
                          const std::string& prefix, uint64_t dispatch_id) {
  if (cfg.show_prefix) writef(out, "Prefix={}, ", prefix);
  writef(out, "DispatchID={}", dispatch_id);
}

// Emits per-trace dispatch entries separated by ",\n" with no leading or
// trailing comma. Caller is responsible for the surrounding "[\n" / "\n]\n"
// and for inserting ",\n" between non-empty per-trace bodies.
void dump_loops_json(const DumpCtx& ctx, const std::vector<LoopGroup>& groups) {
  bool first_entry = true;
  for (const auto& g : groups) {
    if (!first_entry) writef(ctx.out, ",\n");
    first_entry = false;
    writef(ctx.out, "  {{");
    if (ctx.cfg.show_prefix)
      writef(ctx.out, "\"prefix\":\"{}\",", json_escape(ctx.prefix));
    if (g.is_aggregate)
      writef(ctx.out, "\"dispatch_id\":{},\"aggregate\":true,\"loops\":[\n",
             ctx.dispatch_id);
    else
      writef(ctx.out,
             "\"dispatch_id\":{},\"cu\":{},\"simd\":{},\"wave_id\":{},"
             "\"loops\":[\n",
             ctx.dispatch_id, g.cu, g.simd, g.wave_id);
    bool first_loop = true;
    for (const auto& lr : g.loops) {
      if (!first_loop) writef(ctx.out, ",\n");
      first_loop = false;
      auto wave_count = g.is_aggregate ? std::optional<uint32_t>(lr.wave_count)
                                       : std::nullopt;
      dump_loop_json(ctx.out, lr.back_edge, lr.stats, ctx.metrics(lr), ctx.cfg,
                     wave_count);
    }
    writef(ctx.out, "\n  ]}}");
  }
}

void dump_loops_csv(const DumpCtx& ctx, const std::vector<LoopGroup>& groups) {
  for (const auto& g : groups) {
    for (const auto& lr : g.loops) {
      dump_csv_loop_rows(ctx.out, ctx.cfg, ctx.prefix, ctx.dispatch_id, g, lr,
                         ctx.metrics(lr));
    }
  }
}

void dump_loop_text_stats(TraceBuf& out,
                          const my_rocperf_tool::LoopRecordMetrics& metrics,
                          const LoopOutputConfig& cfg) {
  const auto& sum = metrics.summary;
  writef(out, "    avg/iter: dur={:.2f} stall={:.2f} idle={:.2f}", sum.avg_dur,
         sum.avg_stall, sum.avg_idle);
  if (cfg.show_mfma_coexec) writef(out, " bubble={:.2f}", sum.avg_bubble);
  writef(out, " mfma_time={} ({} mfma * {}) mfma_util={:.1f}%\n", sum.mfma_time,
         sum.mfma_count, my_rocperf_tool::kMfmaShadowCycles,
         sum.mfma_util * 100.0);
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

void dump_loops_text(const DumpCtx& ctx, const std::vector<LoopGroup>& groups) {
  for (const auto& g : groups) {
    writef(ctx.out,
           g.is_aggregate ? "=== Aggregate Across Waves (" : "=== Wave (");
    write_dispatch_label(ctx.out, ctx.cfg, ctx.prefix, ctx.dispatch_id);
    if (g.is_aggregate)
      writef(ctx.out, ") ===\n");
    else
      writef(ctx.out, ", CU={}, SIMD={}, WaveID={}) ===\n", g.cu, g.simd,
             g.wave_id);
    for (const auto& lr : g.loops) {
      const auto& e = lr.back_edge;
      const auto& s = lr.stats;
      uint32_t avg_dur =
          s.iteration_count ? s.total_duration / s.iteration_count : 0;
      uint32_t avg_stall =
          s.iteration_count ? s.total_stall / s.iteration_count : 0;
      if (g.is_aggregate)
        writef(ctx.out,
               "  Loop [0x{:x} -> 0x{:x}] code_object={}: "
               "{} waves, {} iterations, {} total cycles (avg {}/iter), "
               "stall {} (avg {}/iter), idle {}\n",
               e.target_addr, e.source_addr, e.code_object_id, lr.wave_count,
               s.iteration_count, s.total_duration, avg_dur, s.total_stall,
               avg_stall, s.total_idle);
      else
        writef(ctx.out,
               "  Loop [0x{:x} -> 0x{:x}] code_object={}: "
               "{} iterations, {} total cycles (avg {}/iter), "
               "stall {} (avg {}/iter), idle {}\n",
               e.target_addr, e.source_addr, e.code_object_id,
               s.iteration_count, s.total_duration, avg_dur, s.total_stall,
               avg_stall, s.total_idle);
      dump_loop_text_stats(ctx.out, ctx.metrics(lr), ctx.cfg);
    }
  }
}

void dump_loops(const DumpCtx& ctx,
                const std::vector<my_rocperf_tool::WaveLoopInfo>& wave_loops) {
  auto groups = build_loop_groups(wave_loops, ctx.cfg);
  switch (ctx.cfg.format) {
    case LoopOutputFormat::Json:
      dump_loops_json(ctx, groups);
      break;
    case LoopOutputFormat::Csv:
      dump_loops_csv(ctx, groups);
      break;
    case LoopOutputFormat::Text:
      dump_loops_text(ctx, groups);
      break;
  }
}

// What one trace decoded, for the end-of-run summary on stderr. Written by
// run_trace, aggregated by run_main.
struct TraceSummary {
  // Why the trace emitted nothing, if it failed. A failure is reported on
  // stderr (a code object load failure once per run), and is kept apart from
  // counts.waves == 0, which is a capture that was read and found empty.
  enum class Failure {
    None,
    Unreadable,          // the .att could not be read
    CodeObjectLoad,      // a code object of its run could not be loaded
    UnknownCodeObject,   // it names a code object its run lacks
    DurationBelowStall,  // rejected by --verify_trace_duration=error
  };
  Failure failure = Failure::None;
  std::string prefix;
  uint64_t dispatch_id = 0;
  uint64_t se_id = 0;  // one .att per shader engine per dispatch
  my_rocperf_tool::WaveCounts counts;
  // The trace's stderr text, which run_main prints in trace order. Most
  // traces have none, and an empty string, unlike a TraceBuf, reserves
  // nothing.
  std::string diagnostics;
};

// Renders the analysis' diagnostics as stderr lines.
struct DiagnosticWriter {
  std::string& out;

  void operator()(const my_rocperf_tool::DurationBelowStall& d) const {
    writef(out,
           "WARNING: duration < stall at PC 0x{:x} code_object={}: "
           "duration={} stall={}\n",
           d.pc.address, d.pc.code_object_id, d.duration, d.stall);
  }
  void operator()(const my_rocperf_tool::MfmaShadowOverlap& o) const {
    writef(out,
           "WARNING: MFMA shadow overlap at PC 0x{:x} code_object={}: "
           "issues at {}, previous shadow ends at {} (overlap={} cycles)\n",
           o.pc.address, o.pc.code_object_id, o.issue_point, o.shadow_end,
           o.shadow_end - o.issue_point);
  }
};

// `disas` is null when a code object of the trace's run failed to load.
void run_trace(TraceBuf& out, TraceSummary& summary,
               const my_rocperf_tool::AttOutputDir& out_dir,
               const my_rocperf_tool::AttPath& att_path,
               my_rocperf_tool::Disassembler* disas,
               const LoopOutputConfig& cfg,
               const my_rocperf_tool::WaveAnalysisOptions& options,
               std::optional<uint64_t> dispatch_id_override) {
  // Keyed before the read, so an unreadable trace still names the dispatch
  // whose loop-free verdict it leaves open.
  summary.prefix = att_path.prefix;
  summary.dispatch_id = dispatch_id_override.value_or(att_path.dispatch_id);
  summary.se_id = att_path.se_id;
  if (!disas) {
    // Its run's load errors are reported with the run's first trace; the
    // trace itself is not.
    summary.failure = TraceSummary::Failure::CodeObjectLoad;
    return;
  }
  // A PC in a code object the run lacks is none of the kinds: the decoder,
  // unable to read the instruction, cannot have taken it for anything. The
  // ISA callback has failed the trace for it anyway.
  my_rocperf_tool::WaveAnalysis analysis(
      options, [disas](const rocprofiler_thread_trace_decoder_pc_t& pc) {
        const auto* obj = disas->find_code_object(pc.code_object_id);
        return obj ? obj->kind_at(pc.address)
                   : my_rocperf_tool::InstKind::Other;
      });
  trace_decoder_context decoder_ctx(*disas, analysis);
  {
    PHASE_TIMER("trace setup + I/O");
    auto att_data = out_dir.read_att_data(att_path);
    if (!att_data) {
      // One bad file should not cost the rest of the capture: report it, emit
      // nothing for it, and leave failing the run to run_main.
      writef(summary.diagnostics, "error: {}\n",
             llvm::toString(att_data.takeError()));
      summary.failure = TraceSummary::Failure::Unreadable;
      return;
    }
    decoder_ctx.att_data = std::move(*att_data);
  }

  rocprofiler_thread_trace_decoder_status_t parse_res;
  {
    PHASE_TIMER("trace decode");
    parse_res = rocprof_trace_decoder_parse_data(
        att_decoder_se_data_callback, att_decoder_trace_callback,
        att_decoder_isa_callback, &decoder_ctx);
  }
  auto result = std::move(analysis).finish();
  // Found during the decode, so they precede any verdict on the trace.
  for (const auto& diagnostic : result.diagnostics)
    std::visit(DiagnosticWriter{summary.diagnostics}, diagnostic);

  // Checked first: the decoder survives an ISA callback error and still
  // returns success, but the waves it delivered are partly unattributed.
  // Fails like an unreadable trace.
  if (decoder_ctx.unknown_code_object_id) {
    writef(summary.diagnostics,
           "error: {}: names code object id {}, which its run does not "
           "have\n",
           att_path.path, *decoder_ctx.unknown_code_object_id);
    summary.failure = TraceSummary::Failure::UnknownCodeObject;
    return;
  }
  assert(parse_res == ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);

  if (result.duration_check_failure) {
    // Fails like an unreadable trace: emit nothing for it and leave failing
    // the run to run_main.
    const auto& failure = *result.duration_check_failure;
    writef(summary.diagnostics,
           "error: {}: {} record(s) with duration < stall, the first at "
           "PC 0x{:x} code_object={}: duration={} stall={}\n",
           att_path.path, failure.records, failure.first.pc.address,
           failure.first.pc.code_object_id, failure.first.duration,
           failure.first.stall);
    summary.failure = TraceSummary::Failure::DurationBelowStall;
    return;
  }

  if (absl::GetFlag(FLAGS_decode_all_insts)) {
    PHASE_TIMER("decode_all_insts");
    disas->decode_all();
  }

  {
    PHASE_TIMER("sort wave_loops");
    std::sort(result.wave_loops.begin(), result.wave_loops.end(),
              [](const my_rocperf_tool::WaveLoopInfo& a,
                 const my_rocperf_tool::WaveLoopInfo& b) {
                return std::tie(a.cu, a.simd, a.wave_id) <
                       std::tie(b.cu, b.simd, b.wave_id);
              });
  }

  summary.counts = result.counts;

  {
    PHASE_TIMER("dump_loops");
    if (options.detect_loops) {
      LoopBodies bodies(*disas);
      DumpCtx dump_ctx{out, result.profile,  bodies,
                       cfg, att_path.prefix, summary.dispatch_id};
      // Test what dump_loops actually emitted rather than whether loops were
      // found: --aggregate_wave writes an (empty) aggregate header even with
      // no loops, and a second "no loops" banner would contradict it.
      size_t before = out.size();
      dump_loops(dump_ctx, result.wave_loops);
      // A trace that emitted nothing would otherwise be indistinguishable
      // from a failed run. Text output only -- csv/json stay parseable.
      if (out.size() == before && cfg.format == LoopOutputFormat::Text) {
        writef(out, "=== ");
        write_dispatch_label(out, cfg, att_path.prefix, summary.dispatch_id);
        writef(out, ", SE={}: no loops detected ({} waves, {} instructions",
               att_path.se_id, result.counts.waves, result.counts.insts);
        if (options.simd >= 0)
          writef(out, "; {} waves on other SIMDs skipped by --simd={}",
                 result.counts.waves_filtered, options.simd);
        writef(out, ") ===\n");
      }
    }
  }
}

int run_main(const std::string& att_output_dir_path,
             const my_rocperf_tool::WaveAnalysisOptions& options) {
  PHASE_TIMER("TOTAL");

  auto scanned = my_rocperf_tool::AttOutputDir::scan(att_output_dir_path);
  if (!scanned) {
    std::cerr << "error: " << llvm::toString(scanned.takeError()) << "\n";
    return 2;
  }
  const auto& out_dir = *scanned;
  if (out_dir.att_paths.empty()) {
    std::cerr << "no ATT trace files found in " << att_output_dir_path << "\n";
    return 2;
  }

  // Code object ids restart in every run, so a trace can only be decoded
  // against code objects carrying its own prefix.
  std::set<std::string> code_object_prefixes;
  for (const auto& obj_file : out_dir.code_objects)
    code_object_prefixes.insert(obj_file.prefix);
  std::set<std::string> trace_prefixes;
  for (const auto& att_path : out_dir.att_paths)
    trace_prefixes.insert(att_path.prefix);
  for (const auto& prefix : trace_prefixes) {
    if (!code_object_prefixes.count(prefix)) {
      std::cerr << "no code objects found for ATT traces with prefix '"
                << prefix << "' in " << att_output_dir_path << "\n";
      return 2;
    }
  }

  auto cfg = LoopOutputConfig::from_flags();
  cfg.show_prefix = trace_prefixes.size() > 1;
  auto dispatch_id_override =
      out_dir.att_paths.size() == 1
          ? parse_dispatch_id_from_path(att_output_dir_path)
          : std::optional<uint64_t>{};
  const size_t n_traces = out_dir.att_paths.size();

  // Past ~64 workers, contention/scheduling overhead dominates the
  // per-trace decode (~55ms) on this workload.
  constexpr size_t kAutoJobsCap = 64;
  size_t jobs;
  if (uint32_t flag = absl::GetFlag(FLAGS_jobs); flag != 0) {
    jobs = std::min<size_t>(n_traces, flag);
  } else {
    auto hw = std::thread::hardware_concurrency();
    jobs = std::min<size_t>({n_traces, hw ? hw : 1u, kAutoJobsCap});
  }

  // One Disassembler per run, built on a worker's first trace from that run
  // and holding only that run's code objects. A Disassembler cannot move (each
  // of its MCContexts points at its MC options), so it is built in place by
  // try_emplace in a node-based map, which never relocates its values.
  struct RunDisassembler {
    my_rocperf_tool::Disassembler disas;
    // One "error: <path>: <reason>" line per code object of the run that
    // failed to load. Any fails all the run's traces: their PCs may land in
    // it.
    std::string load_errors;
  };
  using DisasByPrefix = std::map<std::string, RunDisassembler>;
  auto run_for = [&](DisasByPrefix& by_prefix,
                     const std::string& prefix) -> RunDisassembler& {
    auto [it, inserted] = by_prefix.try_emplace(prefix);
    auto& run = it->second;
    if (inserted) {
      for (const auto& obj_file : out_dir.code_objects) {
        if (obj_file.prefix != prefix) continue;
        if (auto err = run.disas.add_code_object(obj_file.id, obj_file.path))
          writef(run.load_errors, "error: {}\n",
                 llvm::toString(std::move(err)));
      }
    }
    return run;
  };

  // Per-trace output buffers. Indexed in (prefix, dispatch_id) order; workers
  // may complete out of order. fmt::memory_buffer owns its storage (RAII), so
  // exceptions between fill and emit don't leak.
  // Memory note: chunks hold all per-trace output until emit. For
  // aggregate_wave_only this is small; --dump_loop_contents over thousands
  // of traces could reach hundreds of MB.
  std::vector<TraceBuf> chunks(n_traces);
  // Same indexing as chunks; each worker writes only its own slot.
  std::vector<TraceSummary> summaries(n_traces);

  // Returns whether the trace reached the decoder.
  auto process_one = [&](DisasByPrefix& by_prefix, size_t i) {
    const auto& att_path = out_dir.att_paths[i];
    auto& run = run_for(by_prefix, att_path.prefix);
    // Every worker that loads a run meets the same errors. Traces are sorted
    // by prefix, so the run's first trace is the one after a prefix change:
    // it alone reports them, once and in trace order.
    if (i == 0 || out_dir.att_paths[i - 1].prefix != att_path.prefix)
      summaries[i].diagnostics = run.load_errors;
    run_trace(chunks[i], summaries[i], out_dir, att_path,
              run.load_errors.empty() ? &run.disas : nullptr, cfg, options,
              dispatch_id_override);
    const auto failure = summaries[i].failure;
    return failure != TraceSummary::Failure::Unreadable &&
           failure != TraceSummary::Failure::CodeObjectLoad;
  };

  // Always process traces serially until one has been decoded. Two reasons:
  // 1. The decoder library lazily fills a process-wide singleton trie
  //    (Trie::root_trie in rocprof-trace-decoder/source/trie.cpp), guarded
  //    only by a non-atomic bool. It does so not on its first call but the
  //    first time it classifies an instruction whose text neither starts
  //    with "v_" nor contains "branch": an s_ instruction, which nearly
  //    every wave runs (s_endpgm at least). Concurrent fills race on its
  //    std::unordered_map writes — UB. Once a trace has filled it,
  //    std::thread construction publishes those writes to workers, which
  //    then only read. A trace that is unreadable, or whose run's code
  //    objects did not all load, never reaches the decoder, hence "until
  //    one has been decoded" rather than just trace[0]. One decoded without
  //    a single wave fills nothing either, and still ends the warm-up.
  // 2. The serial Disassembler then becomes the main thread's worker, so
  //    no per-worker setup work is wasted.
  DisasByPrefix main_disas;
  size_t next_serial = 0;
  {
    PHASE_TIMER("load code objects + warmup");
    while (next_serial < n_traces) {
      if (process_one(main_disas, next_serial++)) break;
    }
  }

  if (next_serial < n_traces) {
    PHASE_TIMER("parallel decode");
    std::atomic<size_t> next{next_serial};
    auto worker_loop = [&](DisasByPrefix& by_prefix) {
      while (true) {
        size_t i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= n_traces) break;
        process_one(by_prefix, i);
      }
    };
    std::vector<std::thread> workers;
    workers.reserve(jobs - 1);
    for (size_t w = 1; w < jobs; w++) {
      workers.emplace_back([&]() {
        WORKER_PHASE_TIMERS();
        DisasByPrefix local_disas;
        worker_loop(local_disas);
      });
    }
    worker_loop(main_disas);
    for (auto& t : workers) t.join();
  }

  // In trace order whatever order the workers ran in, so stderr reads the
  // same for any --jobs. Ahead of stdout's payload, like the decode that
  // found them.
  for (const auto& s : summaries)
    std::fwrite(s.diagnostics.data(), 1, s.diagnostics.size(), stderr);

  // Emit prologue + chunks (in order) + epilogue.
  if (cfg.format == LoopOutputFormat::Json)
    std::fputs("[\n", stdout);
  else if (cfg.format == LoopOutputFormat::Csv) {
    TraceBuf header;
    dump_csv_header(header, cfg);
    std::fwrite(header.data(), 1, header.size(), stdout);
  }
  bool first_nonempty = true;
  for (const auto& c : chunks) {
    if (c.size() == 0) continue;
    if (cfg.format == LoopOutputFormat::Json && !first_nonempty)
      std::fputs(",\n", stdout);
    std::fwrite(c.data(), 1, c.size(), stdout);
    first_nonempty = false;
  }
  if (cfg.format == LoopOutputFormat::Json) std::fputs("\n]\n", stdout);

  // Run summary on stderr. stdout is fully buffered when redirected, so flush
  // it first: otherwise, under `2>&1`, the unbuffered stderr text lands ahead
  // of (or inside) the csv/json payload it is meant to sit after.
  std::fflush(stdout);

  // Reports what was decoded, not just what was found, so an empty result set
  // is visibly a loop-free capture rather than a silent decode failure. Note
  // a trace here is one .att file = one shader engine of one dispatch.
  {
    uint64_t tot_waves = 0, tot_insts = 0, tot_loop_waves = 0, tot_loops = 0;
    uint64_t tot_filtered = 0;
    uint64_t no_wave_traces = 0, filtered_only_traces = 0;
    uint64_t unreadable_traces = 0, rejected_traces = 0;
    uint64_t code_object_load_traces = 0, unknown_code_object_traces = 0;
    // A dispatch is loop-free only if some SE decoded waves, none found a
    // loop, and no SE failed; per-SE slots would otherwise report the same
    // dispatch repeatedly, including dispatches whose loops all landed in
    // another SE's trace.
    struct DispatchAgg {
      uint64_t waves = 0;
      uint64_t loops = 0;
      // An SE trace failed, so its loops (if any) are unknown.
      bool failed = false;
    };
    // Keyed by run too: the same dispatch id in two runs is two dispatches.
    std::map<std::pair<std::string, uint64_t>, DispatchAgg> by_dispatch;
    for (const auto& s : summaries) {
      auto& d = by_dispatch[{s.prefix, s.dispatch_id}];
      // Never reported, so not an empty capture: it adds nothing to the
      // totals, but leaves its dispatch's verdict open.
      if (s.failure != TraceSummary::Failure::None) {
        if (s.failure == TraceSummary::Failure::Unreadable)
          unreadable_traces++;
        else if (s.failure == TraceSummary::Failure::CodeObjectLoad)
          code_object_load_traces++;
        else if (s.failure == TraceSummary::Failure::UnknownCodeObject)
          unknown_code_object_traces++;
        else
          rejected_traces++;
        d.failed = true;
        continue;
      }
      tot_waves += s.counts.waves;
      tot_insts += s.counts.insts;
      tot_loop_waves += s.counts.loop_waves;
      tot_loops += s.counts.loops;
      tot_filtered += s.counts.waves_filtered;
      if (s.counts.waves == 0) {
        if (s.counts.waves_filtered > 0)
          filtered_only_traces++;
        else
          no_wave_traces++;
      }
      d.waves += s.counts.waves;
      d.loops += s.counts.loops;
    }
    const uint64_t failed_traces = unreadable_traces + code_object_load_traces +
                                   unknown_code_object_traces + rejected_traces;
    const uint64_t n_reported = n_traces - failed_traces;

    if (options.detect_loops)
      fmt::print(stderr,
                 "[summary] {} trace(s), {} waves, {} instructions decoded; "
                 "{} loop instance(s) in {} wave(s)\n",
                 n_reported, tot_waves, tot_insts, tot_loops, tot_loop_waves);
    else
      fmt::print(stderr,
                 "[summary] {} trace(s), {} waves, {} instructions decoded; "
                 "loop detection disabled\n",
                 n_reported, tot_waves, tot_insts);
    if (unreadable_traces)
      fmt::print(stderr, "[summary] {} of {} trace(s) could not be read\n",
                 unreadable_traces, n_traces);
    if (code_object_load_traces)
      fmt::print(stderr,
                 "[summary] {} of {} trace(s) not decoded: a code object of "
                 "their run could not be loaded\n",
                 code_object_load_traces, n_traces);
    if (unknown_code_object_traces)
      fmt::print(stderr,
                 "[summary] {} of {} trace(s) name a code object their run "
                 "does not have\n",
                 unknown_code_object_traces, n_traces);
    if (rejected_traces)
      fmt::print(stderr,
                 "[summary] {} of {} trace(s) rejected by "
                 "--verify_trace_duration=error\n",
                 rejected_traces, n_traces);

    // Decoding nothing is a different fact from finding no loop -- keep the
    // two apart, or the loop-free report makes a claim about a capture whose
    // instruction stream was never read.
    if (no_wave_traces)
      fmt::print(stderr,
                 "[summary] {} of {} trace(s) decoded no waves "
                 "(empty, truncated, or wrong-agent capture)\n",
                 no_wave_traces, n_reported);
    if (options.simd >= 0) {
      fmt::print(stderr,
                 "[summary] --simd={}: {} wave(s) on other SIMDs skipped\n",
                 options.simd, tot_filtered);
      if (filtered_only_traces)
        fmt::print(stderr,
                   "[summary] {} of {} trace(s) had waves only on other "
                   "SIMDs\n",
                   filtered_only_traces, n_reported);
    }

    if (options.detect_loops) {
      // detect_loops keeps a loop at iteration_count >= 2, and one taken
      // back-edge already yields 2 (the flush at the edge plus the trailing
      // partial iteration) -- so the criterion is "taken", not "taken twice".
      // Dispatches with a failed trace stay in the "of M" count, like those
      // that decoded no waves: M is every dispatch in the capture, the
      // verdict only the ones that could be judged.
      std::vector<std::pair<std::string, uint64_t>> loop_free;
      for (const auto& [key, d] : by_dispatch)
        if (d.waves > 0 && d.loops == 0 && !d.failed) loop_free.push_back(key);
      if (!loop_free.empty()) {
        fmt::print(stderr,
                   "[summary] {} of {} dispatch(es) contained no loop "
                   "(no back-edge was taken)\n",
                   loop_free.size(), by_dispatch.size());
        // Buffer then one write: the list is unbounded and stderr is
        // unbuffered, so per-id writes would be a syscall apiece.
        if (loop_free.size() < by_dispatch.size()) {
          TraceBuf ids;
          writef(ids, "[summary] loop-free dispatch ids:");
          for (const auto& [prefix, id] : loop_free) {
            if (cfg.show_prefix)
              writef(ids, " {}:{}", prefix, id);
            else
              writef(ids, " {}", id);
          }
          writef(ids, "\n");
          std::fwrite(ids.data(), 1, ids.size(), stderr);
        }
      }
    }
    // Each failed trace was reported above; the exit status keeps a script
    // from taking the partial result for the whole capture.
    if (failed_traces) return 1;
  }
  return 0;
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  std::optional<std::string> att_output_dir =
      absl::GetFlag(FLAGS_att_output_dir);
  if (!att_output_dir.has_value()) {
    std::cerr << "--att_output_dir not specified\n";
    return 2;
  }
  if (int32_t simd = absl::GetFlag(FLAGS_simd); simd < -1 || simd > 3) {
    std::cerr << "--simd must be in [0,3], or -1 for all SIMDs; got " << simd
              << "\n";
    return 2;
  }
  const std::string duration_check_text =
      absl::GetFlag(FLAGS_verify_trace_duration);
  std::optional<my_rocperf_tool::DurationCheck> duration_check =
      parse_duration_check(duration_check_text);
  if (!duration_check) {
    std::cerr << "--verify_trace_duration must be off, warn or error; got '"
              << duration_check_text << "'\n";
    return 2;
  }
  my_rocperf_tool::WaveAnalysisOptions options;
  options.simd = absl::GetFlag(FLAGS_simd);
  options.detect_loops = absl::GetFlag(FLAGS_detect_loops);
  options.mfma_coexec = absl::GetFlag(FLAGS_mfma_coexec);
  // Only the loop contents of per-wave records print a wave's rows.
  options.wave_loop_rows = absl::GetFlag(FLAGS_dump_loop_contents) &&
                           !absl::GetFlag(FLAGS_aggregate_wave_only);
  options.duration_check = *duration_check;
  // After absl is done with argv: on some platforms InitLLVM rewrites it.
  my_rocperf_tool::InitLLVM init_llvm(argc, argv);
  // Outlives run_main so its TOTAL scope has closed by the time we write.
  PHASE_TIMER_SESSION();
  return run_main(*att_output_dir, options);
}
