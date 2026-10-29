#include "my_rocperf_tool/att_output_dir.h"
#include "my_rocperf_tool/disassembler.h"
#include "my_rocperf_tool/loop_detection.h"
#include "my_rocperf_tool/loop_profile.h"
#include "my_rocperf_tool/loop_report.h"
#include "my_rocperf_tool/wave_analysis.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "trace_decoder_api.h"
#include "trace_decoder_types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
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

// The loop report's options, but for show_prefix, which depends on the
// directory.
my_rocperf_tool::LoopOutputConfig loop_output_config_from_flags() {
  my_rocperf_tool::LoopOutputConfig cfg;
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
    cfg.format = my_rocperf_tool::LoopOutputFormat::Json;
  else if (fmt == "csv")
    cfg.format = my_rocperf_tool::LoopOutputFormat::Csv;
  else
    cfg.format = my_rocperf_tool::LoopOutputFormat::Text;
  return cfg;
}

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
// needs the loop's body. The report's rows view the bodies' text until it is
// rendered. Listing a body may grow the map, which moves the other bodies, but
// moving a vector leaves its elements, and so their text, where they are.
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

// The report's groups, in print order: a group per wave with loops unless
// --aggregate_wave_only, then the aggregate under --aggregate_wave. Each
// record is resolved here, against the profile and its loop's body, so
// rendering reads plain data.
std::vector<my_rocperf_tool::LoopGroup> build_loop_groups(
    const std::vector<my_rocperf_tool::WaveLoopInfo>& wave_loops,
    const my_rocperf_tool::LoopProfile& profile, LoopBodies& bodies,
    const my_rocperf_tool::LoopOutputConfig& cfg) {
  auto resolve = [&](const my_rocperf_tool::LoopRecord& record) {
    return my_rocperf_tool::ReportedLoop{
        record, profile.metrics(record, bodies.of(record.back_edge))};
  };
  std::vector<my_rocperf_tool::LoopGroup> groups;
  if (!cfg.aggregate_wave_only) {
    for (const auto& wl : wave_loops) {
      my_rocperf_tool::LoopGroup g{false, wl.cu, wl.simd, wl.wave_id, {}};
      g.loops.reserve(wl.loops.size());
      for (const auto& dl : wl.loops)
        g.loops.push_back(resolve({dl.back_edge, dl.stats, 1, wl.wave_index}));
      groups.push_back(std::move(g));
    }
  }
  if (cfg.aggregate_wave) {
    my_rocperf_tool::LoopGroup g{true, 0, 0, 0, {}};
    for (const auto& record : aggregate_loops_by_back_edge(wave_loops))
      g.loops.push_back(resolve(record));
    groups.push_back(std::move(g));
  }
  return groups;
}

// fmt::format_to for the stderr text, which is rare enough that how it is
// appended does not matter.
template <typename... T>
inline void writef(std::string& out, fmt::format_string<T...> f, T&&... args) {
  fmt::format_to(std::back_inserter(out), f, std::forward<T>(args)...);
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
  // traces have none, and an empty string, unlike a LoopReportChunk,
  // reserves nothing. Printed only once every worker has joined: an abort in
  // a later trace loses the lines buffered so far, and under
  // --verify_trace_duration=warn every warning stays in memory until the end.
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
void run_trace(my_rocperf_tool::LoopReportChunk& out, TraceSummary& summary,
               const my_rocperf_tool::AttOutputDir& out_dir,
               const my_rocperf_tool::AttPath& att_path,
               my_rocperf_tool::Disassembler* disas,
               const my_rocperf_tool::LoopOutputConfig& cfg,
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

  // Read only by an assert.
  [[maybe_unused]] rocprofiler_thread_trace_decoder_status_t parse_res;
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
    PHASE_TIMER("loop report");
    if (options.detect_loops) {
      LoopBodies bodies(*disas);
      my_rocperf_tool::TraceLoopReport report;
      report.prefix = att_path.prefix;
      report.dispatch_id = summary.dispatch_id;
      report.se_id = att_path.se_id;
      report.counts = result.counts;
      report.simd_filter = options.simd;
      report.groups =
          build_loop_groups(result.wave_loops, result.profile, bodies, cfg);
      my_rocperf_tool::render_loop_report(out, cfg, report);
    }
  }
}

// `cfg` lacks only show_prefix, which run_main sets from the directory.
int run_main(const std::string& att_output_dir_path,
             my_rocperf_tool::LoopOutputConfig cfg,
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
  // and holding only that run's code objects. A Disassembler cannot move (its
  // MC objects keep the address of its MC options), so it is built in place by
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
  std::vector<my_rocperf_tool::LoopReportChunk> chunks(n_traces);
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
  //    first time it classifies an instruction that is neither a v_ one nor
  //    a branch (text containing "branch"): s_endpgm, for instance, which
  //    nearly every wave runs. Concurrent fills race on its std::unordered_map
  //    writes — UB. Once a trace has filled it, std::thread construction
  //    publishes those writes to workers, which then only read. A trace that
  //    is unreadable, or whose run's code objects did not all load, never
  //    reaches the decoder, hence "until one has been decoded" rather than
  //    just trace[0]. A decoded trace can still end the warm-up with the
  //    trie unfilled: one without a single wave, or one whose PCs name only
  //    code objects its run lacks, since the decoder classifies no
  //    instruction the ISA callback fails.
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

  my_rocperf_tool::write_loop_report_document(
      cfg, chunks, [](std::string_view piece) {
        std::fwrite(piece.data(), 1, piece.size(), stdout);
      });

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
      switch (s.failure) {
        case TraceSummary::Failure::None:
          break;
        case TraceSummary::Failure::Unreadable:
          unreadable_traces++;
          break;
        case TraceSummary::Failure::CodeObjectLoad:
          code_object_load_traces++;
          break;
        case TraceSummary::Failure::UnknownCodeObject:
          unknown_code_object_traces++;
          break;
        case TraceSummary::Failure::DurationBelowStall:
          rejected_traces++;
          break;
      }
      // Never reported, so not an empty capture: it adds nothing to the
      // totals, but leaves its dispatch's verdict open.
      if (s.failure != TraceSummary::Failure::None) {
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
          std::string ids;
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
    std::cerr << "--verify_trace_duration must be off, warn or error (true "
                 "and false are accepted as warn and off); got '"
              << duration_check_text << "'\n";
    return 2;
  }
  const auto cfg = loop_output_config_from_flags();
  my_rocperf_tool::WaveAnalysisOptions options;
  options.simd = absl::GetFlag(FLAGS_simd);
  options.detect_loops = absl::GetFlag(FLAGS_detect_loops);
  options.mfma_coexec = absl::GetFlag(FLAGS_mfma_coexec);
  // Only the loop contents of per-wave records print a wave's rows. Taken
  // from the report's options, so the analysis keeps what they print.
  options.wave_loop_rows = cfg.show_contents && !cfg.aggregate_wave_only;
  options.duration_check = *duration_check;
  // For crash handlers and stack traces; a Disassembler registers its target
  // itself. After absl is done with argv: on some platforms InitLLVM rewrites
  // it.
  llvm::InitLLVM init_llvm(argc, argv);
  // Outlives run_main so its TOTAL scope has closed by the time we write.
  PHASE_TIMER_SESSION();
  return run_main(*att_output_dir, cfg, options);
}
