#include "my_rocperf_tool/att_output_dir.h"
#include "my_rocperf_tool/disassembler.h"
#include "my_rocperf_tool/init_llvm.h"
#include "my_rocperf_tool/inst_statistics.h"
#include "my_rocperf_tool/loop_detection.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#if ROCPROF_TRACE_DECODER_NEW
#include "trace_decoder_api.h"
#else
#include "rocprof_trace_decoder.h"
#endif

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInst.h"

#include "fmt/format.h"

#include <atomic>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifdef ENABLE_TIMERS
#include <chrono>
struct PhaseTimer {
  using clock = std::chrono::steady_clock;
  std::chrono::time_point<clock> start;
  const char *name;
  PhaseTimer(const char *n) : name(n), start(clock::now()) {}
  ~PhaseTimer() {
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                  clock::now() - start)
                  .count();
    // Format then single fwrite so output stays line-atomic when many
    // worker threads emit timers concurrently. glibc holds a per-FILE
    // lock around fwrite; the line is well under PIPE_BUF/BUFSIZ.
    fmt::basic_memory_buffer<char, 128> buf;
    fmt::format_to(fmt::appender(buf),
                   "[TIMER] {:<35} {}.{:03} ms\n", name, us / 1000,
                   us % 1000);
    std::fwrite(buf.data(), 1, buf.size(), stderr);
  }
};
#define PHASE_TIMER_CONCAT_(a, b) a##b
#define PHASE_TIMER_CONCAT(a, b) PHASE_TIMER_CONCAT_(a, b)
#define PHASE_TIMER(name) PhaseTimer PHASE_TIMER_CONCAT(_timer_, __LINE__)(name)
#else
#define PHASE_TIMER(name) ((void)0)
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
ABSL_FLAG(bool, verify_trace_duration, false,
          "verify duration >= stall for each trace record");
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

using PerPCMap =
    llvm::DenseMap<rocprofiler_thread_trace_decoder_pc_t, uint64_t>;
using PerLoopMap = llvm::DenseMap<my_rocperf_tool::BackEdge, PerPCMap>;

static const PerPCMap empty_pc_map;

enum class LoopOutputFormat { Text, Csv, Json };

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
    if (cfg.aggregate_wave_only)
      cfg.aggregate_wave = true;
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

struct trace_decoder_context {
  bool first_run;
  std::unique_ptr<char[]> att_file_content;
  size_t att_file_size;
  uint64_t dispatch_id;
  my_rocperf_tool::Disassembler &disas;
  my_rocperf_tool::InstStatistics stats;
  bool detect_loops_flag;
  bool verify_duration;
  bool mfma_coexec_flag;
  std::vector<my_rocperf_tool::WaveLoopInfo> wave_loops;
  llvm::DenseMap<rocprofiler_thread_trace_decoder_pc_t, bool> mfma_cache;
  PerLoopMap bubble_totals;
  PerLoopMap rel_issue_totals;

  explicit trace_decoder_context(my_rocperf_tool::Disassembler &d) : disas(d) {}
};

std::optional<uint64_t> parse_dispatch_id_from_path(const std::string &path) {
  auto name = std::filesystem::path(path).filename().string();
  const std::string marker = "_dispatch_";
  auto marker_pos = name.rfind(marker);
  if (marker_pos == std::string::npos)
    return std::nullopt;
  auto digit_pos = marker_pos + marker.size();
  if (digit_pos >= name.size() || !std::isdigit(static_cast<unsigned char>(name[digit_pos])))
    return std::nullopt;
  return std::stoull(name.substr(digit_pos));
}

uint64_t att_decoder_se_data_callback(uint8_t **buffer, uint64_t *buffer_size,
                                      void *userdata) {
  auto &ctx = *reinterpret_cast<trace_decoder_context *>(userdata);
  if (!ctx.first_run) {
    return 0;
  }
  ctx.first_run = false;
  uint64_t data_size = ctx.att_file_size;
  *buffer_size = data_size;
  *buffer = (uint8_t *)ctx.att_file_content.get();
  return data_size;
}

rocprofiler_thread_trace_decoder_status_t att_decoder_isa_callback(
    char *isa_instruction, uint64_t *isa_memory_size, uint64_t *isa_size,
    rocprofiler_thread_trace_decoder_pc_t pc, void *userdata) {
  assert(pc.code_object_id != 0);
  auto &ctx = *reinterpret_cast<trace_decoder_context *>(userdata);
  auto &obj_file = ctx.disas.get_object_file_by_id(pc.code_object_id);
  uint64_t inst_size;
  auto mc_inst = obj_file.decode_at(pc.address, inst_size);
  obj_file.streamer->emitInstruction(mc_inst, *obj_file.sub_target);
  llvm::StringRef inst_ref = llvm::StringRef(obj_file.inst_str).trim();
  auto str_size = inst_ref.size();
  if (*isa_size < str_size) {
    *isa_size = str_size;
    obj_file.inst_str.clear();
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  memcpy(isa_instruction, inst_ref.data(), str_size);
  *isa_size = str_size;
  *isa_memory_size = inst_size;
  obj_file.inst_str.clear();
  return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

void analyze_mfma_coexec(
    trace_decoder_context &ctx,
    const rocprofiler_thread_trace_decoder_wave_t &wave,
    const my_rocperf_tool::BackEdge &edge) {
  auto &obj = ctx.disas.get_object_file_by_id(edge.code_object_id);
  int64_t good_interval = 0;
  int64_t last_mfma_coexec_end = 0;

  for (size_t i = 0; i < wave.instructions_size; i++) {
    const auto &inst = wave.instructions_array[i];
    if (my_rocperf_tool::is_null_pc(inst))
      continue;
    if (inst.pc.code_object_id != edge.code_object_id ||
        inst.pc.address < edge.target_addr ||
        inst.pc.address > edge.source_addr)
      continue;

    // Check if this instruction is MFMA (cached per PC).
    auto [cache_it, inserted] = ctx.mfma_cache.try_emplace(inst.pc, false);
    if (inserted) {
      uint64_t inst_size;
      auto mc_inst = obj.decode_at(inst.pc.address, inst_size);
      cache_it->second =
          obj.mc_instr_info->getName(mc_inst.getOpcode())
              .starts_with("V_MFMA");
    }

    if (cache_it->second) {
      // MFMA is productive work — update shadow, don't count as bubble.
      int64_t issue_point = inst.time + inst.stall;
      // Verify no two MFMAs issue within each other's shadow.
      if (last_mfma_coexec_end > 0 && issue_point < last_mfma_coexec_end) {
        fmt::print(stderr,
                   "WARNING: MFMA shadow overlap at PC 0x{:x} code_object={}: "
                   "issues at {}, previous shadow ends at {} "
                   "(overlap={} cycles)\n",
                   inst.pc.address, inst.pc.code_object_id, issue_point,
                   last_mfma_coexec_end,
                   last_mfma_coexec_end - issue_point);
      }
      good_interval = issue_point + 16;
      last_mfma_coexec_end = good_interval;
    } else {
      // Non-MFMA: compute bubble (portion not covered by co-exec shadow).
      int64_t end_time = inst.time + inst.duration;
      if (end_time > good_interval) {
        int64_t uncovered = end_time - good_interval;
        int64_t bubble =
            std::min(static_cast<int64_t>(inst.duration), uncovered);
        if (bubble > 0)
          ctx.bubble_totals[edge][inst.pc] += static_cast<uint64_t>(bubble);
      }
    }
  }
}

void compute_relative_issue(
    trace_decoder_context &ctx,
    const rocprofiler_thread_trace_decoder_wave_t &wave,
    const my_rocperf_tool::BackEdge &edge) {
  int64_t iter_start = -1;
  for (size_t i = 0; i < wave.instructions_size; i++) {
    const auto &inst = wave.instructions_array[i];
    if (my_rocperf_tool::is_null_pc(inst))
      continue;
    if (inst.pc.code_object_id != edge.code_object_id ||
        inst.pc.address < edge.target_addr ||
        inst.pc.address > edge.source_addr)
      continue;
    // New iteration starts when we hit the loop header.
    if (inst.pc.address == edge.target_addr)
      iter_start = inst.time + inst.stall;
    if (iter_start >= 0) {
      int64_t rel = (inst.time + inst.stall) - iter_start;
      if (rel >= 0)
        ctx.rel_issue_totals[edge][inst.pc] += static_cast<uint64_t>(rel);
    }
  }
}

rocprofiler_thread_trace_decoder_status_t att_decoder_trace_callback(
    rocprofiler_thread_trace_decoder_record_type_t record_type_id,
    void *trace_events, uint64_t trace_size, void *userdata) {
  auto &ctx = *reinterpret_cast<trace_decoder_context *>(userdata);
  switch (record_type_id) {
  default:
    break;
  case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE:
    auto *wave_events =
        static_cast<rocprofiler_thread_trace_decoder_wave_t *>(trace_events);
    for (size_t wave_n = 0; wave_n < trace_size; wave_n++) {
      const auto &wave = wave_events[wave_n];

      int64_t last_time = wave.begin_time;
      for (size_t j = 0; j < wave.instructions_size; j++) {
        const auto &inst = wave.instructions_array[j];
        if (my_rocperf_tool::is_null_pc(inst))
          continue;
        assert(inst.pc.code_object_id != 0);
        uint32_t idle = my_rocperf_tool::compute_idle(inst, last_time);
        if (ctx.verify_duration && inst.duration < static_cast<int32_t>(inst.stall)) {
          fmt::print(stderr,
                     "WARNING: duration < stall at PC 0x{:x} "
                     "code_object={}: duration={} stall={}\n",
                     inst.pc.address, inst.pc.code_object_id, inst.duration,
                     inst.stall);
        }
        auto res = ctx.stats.add_inst(inst, idle);
        assert(res);
      }

      if (ctx.detect_loops_flag) {
        auto loop_info = my_rocperf_tool::detect_loops(wave);
        if (!loop_info.loops.empty()) {
          for (const auto &dl : loop_info.loops) {
            compute_relative_issue(ctx, wave, dl.back_edge);
            if (ctx.mfma_coexec_flag)
              analyze_mfma_coexec(ctx, wave, dl.back_edge);
          }
          ctx.wave_loops.push_back(std::move(loop_info));
        }
      }
    }
  }
  return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

void dump_states(my_rocperf_tool::Disassembler &disas,
                 my_rocperf_tool::InstStatistics &stats,
                 bool decode_all) {
  for (const auto &[pc, inst_vec] : stats.getInsts()) {
    for (const auto &stat : inst_vec) {
      assert(stat.stall <= stat.duration);
    }
  }
  if (decode_all) {
    for (const auto &[obj_id, obj] : disas.get_object_files())
      obj.decode_all_sections();
  }
}

struct LoopInstInfo {
  uint64_t addr;
  std::string inst_str;
  size_t n;
  uint64_t dur, stall, idle, issue;
  uint64_t bubble;
  uint64_t rel_issue;
  bool has_stats;
};

std::vector<LoopInstInfo>
collect_loop_insts(
    my_rocperf_tool::Disassembler &disas,
    my_rocperf_tool::InstStatistics &stats,
    const my_rocperf_tool::BackEdge &edge,
    const PerLoopMap &all_bubbles,
    const PerLoopMap &all_rel_issues) {
  auto bi = all_bubbles.find(edge);
  const auto &bubbles = bi != all_bubbles.end() ? bi->second : empty_pc_map;
  auto ri = all_rel_issues.find(edge);
  const auto &rel_issues = ri != all_rel_issues.end() ? ri->second : empty_pc_map;
  std::vector<LoopInstInfo> result;
  auto &obj = disas.get_object_file_by_id(edge.code_object_id);
  uint64_t start_offset = edge.target_addr - obj.text_sec_address;
  uint64_t end_offset = edge.source_addr - obj.text_sec_address;
  if (start_offset >= obj.text_sec_size || end_offset >= obj.text_sec_size)
    return result;

  auto buffer = obj.memory_buffer->getBuffer();
  auto section_buffer =
      buffer.drop_front(obj.text_sec_offset).take_front(obj.text_sec_size);
  auto block_buffer = section_buffer.drop_front(start_offset);
  llvm::ArrayRef<unsigned char> bytes{block_buffer.bytes_begin(),
                                      block_buffer.size()};

  uint64_t virt_addr = obj.load_base + start_offset;
  uint64_t sec_offset = start_offset;

  while (sec_offset <= end_offset && !bytes.empty()) {
    uint64_t inst_size;
    auto mc_inst = obj.decode_at(sec_offset + obj.text_sec_address, inst_size);
    obj.streamer->emitInstruction(mc_inst, *obj.sub_target);
    std::string inst_str = llvm::StringRef(obj.inst_str).trim().str();
    obj.inst_str.clear();

    uint64_t addr = sec_offset + obj.text_sec_address;
    rocprofiler_thread_trace_decoder_pc_t pc{addr, edge.code_object_id};
    auto stats_range = stats.get_inst_at(pc);

    LoopInstInfo info{addr, std::move(inst_str), 0, 0, 0, 0, 0, 0, 0,
                      !stats_range.empty()};
    if (info.has_stats) {
      for (const auto &s : stats_range) {
        info.dur += s.duration;
        info.stall += s.stall;
        info.idle += s.idle;
        int32_t iss = s.duration - static_cast<int32_t>(s.stall);
        info.issue += iss > 0 ? static_cast<uint64_t>(iss) : 0;
      }
      info.n = stats_range.size();
      auto bi_it = bubbles.find(pc);
      if (bi_it != bubbles.end())
        info.bubble = bi_it->second;
      auto ri_it = rel_issues.find(pc);
      if (ri_it != rel_issues.end())
        info.rel_issue = ri_it->second;
    }
    result.push_back(std::move(info));

    sec_offset += inst_size;
    virt_addr += inst_size;
    bytes = bytes.drop_front(inst_size);
  }
  return result;
}

// Escape a string for CSV output (RFC 4180: double embedded quotes)
std::string csv_escape(const std::string &s) {
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
std::string json_escape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (unsigned char c : s) {
    switch (c) {
    case '"':  out += "\\\""; break;
    case '\\': out += "\\\\"; break;
    case '\b': out += "\\b"; break;
    case '\f': out += "\\f"; break;
    case '\n': out += "\\n"; break;
    case '\r': out += "\\r"; break;
    case '\t': out += "\\t"; break;
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
const char *bubble_color(const LoopInstInfo &ii, uint64_t max_bubble) {
  if (!ii.has_stats || ii.n == 0 || ii.bubble == 0 || max_bubble == 0)
    return "\033[32m"; // green: fully covered
  double ratio = static_cast<double>(ii.bubble) / max_bubble;
  if (ratio <= 0.25)
    return "\033[33m"; // yellow: low
  if (ratio <= 0.75)
    return "\033[31m"; // red: medium
  return "\033[1;31m"; // bright red: high
}

struct LoopSummary {
  double avg_dur, avg_stall, avg_idle, avg_bubble;
  double mfma_util; // mfma_time / avg_dur
  uint32_t mfma_count;
  uint32_t mfma_time; // mfma_count * 16
};

LoopSummary compute_loop_summary(const my_rocperf_tool::LoopStats &s,
                                 const std::vector<LoopInstInfo> &insts) {
  LoopSummary sum{};
  uint64_t total_bubble = 0;
  // ii.bubble and ii.dur/stall/idle are accumulated globally across
  // all waves, so use ii.n (total observations) for per-iteration averages.
  size_t total_n = 0;
  for (const auto &ii : insts) {
    total_bubble += ii.bubble;
    if (ii.has_stats && ii.n > total_n)
      total_n = ii.n;
    if (llvm::StringRef(ii.inst_str).starts_with("v_mfma"))
      sum.mfma_count++;
  }
  if (s.iteration_count > 0) {
    sum.avg_dur = static_cast<double>(s.total_duration) / s.iteration_count;
    sum.avg_stall = static_cast<double>(s.total_stall) / s.iteration_count;
    sum.avg_idle = static_cast<double>(s.total_idle) / s.iteration_count;
  }
  if (total_n > 0)
    sum.avg_bubble = static_cast<double>(total_bubble) / total_n;
  sum.mfma_time = sum.mfma_count * 16;
  if (sum.avg_dur > 0)
    sum.mfma_util = static_cast<double>(sum.mfma_time) / sum.avg_dur;
  return sum;
}

struct LoopRecord {
  my_rocperf_tool::BackEdge back_edge;
  my_rocperf_tool::LoopStats stats;
  uint32_t wave_count; // 1 for per-wave, N for aggregate
};

std::vector<LoopRecord> aggregate_loops_by_back_edge(
    const std::vector<my_rocperf_tool::WaveLoopInfo> &wave_loops) {
  std::vector<LoopRecord> result;
  llvm::DenseMap<my_rocperf_tool::BackEdge, size_t> index_by_edge;
  for (const auto &wl : wave_loops) {
    for (const auto &dl : wl.loops) {
      const auto &edge = dl.back_edge;
      auto it = index_by_edge.find(edge);
      if (it == index_by_edge.end()) {
        it = index_by_edge.try_emplace(edge, result.size()).first;
        result.push_back(LoopRecord{edge, {}, 0});
      }

      auto &agg = result[it->second];
      agg.wave_count++;
      agg.stats.iteration_count += dl.stats.iteration_count;
      agg.stats.total_inst_count += dl.stats.total_inst_count;
      agg.stats.total_duration += dl.stats.total_duration;
      agg.stats.total_stall += dl.stats.total_stall;
      agg.stats.total_idle += dl.stats.total_idle;
    }
  }
  std::sort(result.begin(), result.end(),
            [](const LoopRecord &a, const LoopRecord &b) {
              return std::tie(a.back_edge.code_object_id,
                              a.back_edge.target_addr,
                              a.back_edge.source_addr) <
                     std::tie(b.back_edge.code_object_id,
                              b.back_edge.target_addr,
                              b.back_edge.source_addr);
            });
  return result;
}

struct LoopGroup {
  bool is_aggregate;
  uint8_t cu = 0;     // valid only when !is_aggregate
  uint8_t simd = 0;   // valid only when !is_aggregate
  uint32_t wave_id = 0; // valid only when !is_aggregate
  std::vector<LoopRecord> loops;
};

using LoopInstCache =
    llvm::DenseMap<my_rocperf_tool::BackEdge, std::vector<LoopInstInfo>>;

std::vector<LoopGroup>
build_loop_groups(const std::vector<my_rocperf_tool::WaveLoopInfo> &wave_loops,
                  const LoopOutputConfig &cfg) {
  std::vector<LoopGroup> groups;
  if (!cfg.aggregate_wave_only) {
    for (const auto &wl : wave_loops) {
      LoopGroup g{false, wl.cu, wl.simd, wl.wave_id, {}};
      g.loops.reserve(wl.loops.size());
      for (const auto &dl : wl.loops)
        g.loops.push_back({dl.back_edge, dl.stats, 1});
      groups.push_back(std::move(g));
    }
  }
  if (cfg.aggregate_wave) {
    LoopGroup g{true, 0, 0, 0, aggregate_loops_by_back_edge(wave_loops)};
    groups.push_back(std::move(g));
  }
  return groups;
}

const std::vector<LoopInstInfo> &get_loop_insts(
    LoopInstCache &cache, my_rocperf_tool::Disassembler &disas,
    my_rocperf_tool::InstStatistics &stats,
    const my_rocperf_tool::BackEdge &edge, const PerLoopMap &bubbles,
    const PerLoopMap &rel_issues) {
  auto it = cache.find(edge);
  if (it != cache.end())
    return it->second;
  auto insts = collect_loop_insts(disas, stats, edge, bubbles, rel_issues);
  return cache.try_emplace(edge, std::move(insts)).first->second;
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
inline void writef(TraceBuf &out, fmt::format_string<T...> f, T &&...args) {
  fmt::format_to(fmt::appender(out), f, std::forward<T>(args)...);
}

void dump_loop_json(
    TraceBuf &out, const my_rocperf_tool::BackEdge &e,
    const my_rocperf_tool::LoopStats &s,
    const std::vector<LoopInstInfo> &insts, const LoopOutputConfig &cfg,
    std::optional<uint32_t> wave_count = std::nullopt) {
  auto sum = compute_loop_summary(s, insts);
  writef(out,
         "    {{\"target_addr\":\"0x{:x}\",\"source_addr\":\"0x{:x}\","
         "\"code_object\":{}",
         e.target_addr, e.source_addr, e.code_object_id);
  if (wave_count.has_value())
    writef(out, ",\"wave_count\":{}", *wave_count);
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
    for (const auto &ii : insts) {
      if (!first_inst) writef(out, ",\n");
      first_inst = false;
      writef(out, "      {{\"addr\":\"0x{:x}\",\"inst\":\"{}\"", ii.addr,
             json_escape(ii.inst_str));
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
          writef(out,
                 ",\"dur\":{},\"stall\":{},\"idle\":{},\"issue\":{}", ii.dur,
                 ii.stall, ii.idle, ii.issue);
          if (cfg.show_mfma_coexec)
            writef(out, ",\"bubble\":{}", ii.bubble);
          writef(out, ",\"rel_issue\":{}", ii.rel_issue);
        }
      }
      writef(out, "}}");
    }
    writef(out, "\n    ]");
  }
  writef(out, "}}");
}

void dump_csv_header(TraceBuf &out, const LoopOutputConfig &cfg) {
  if (cfg.aggregate_wave)
    writef(out, "record_type,");
  writef(out, "dispatch_id,cu,simd,wave_id,");
  if (cfg.aggregate_wave)
    writef(out, "wave_count,");
  writef(out, "target_addr,source_addr,code_object,"
              "iterations,total_duration,total_stall,total_idle,"
              "loop_avg_dur,loop_avg_stall,loop_avg_idle,");
  if (cfg.show_mfma_coexec)
    writef(out, "loop_avg_bubble,");
  writef(out, "mfma_count,mfma_time,mfma_util");
  if (cfg.show_contents) {
    writef(out, ",inst_addr,instruction,n,");
    if (cfg.show_avg) {
      writef(out, "avg_dur,avg_stall,avg_idle,avg_issue");
      if (cfg.show_mfma_coexec)
        writef(out, ",avg_bubble");
      writef(out, ",avg_cycle");
    } else {
      writef(out, "dur,stall,idle,issue");
      if (cfg.show_mfma_coexec)
        writef(out, ",bubble");
      writef(out, ",rel_issue");
    }
  }
  writef(out, "\n");
}

void dump_csv_row_prefix(TraceBuf &out, const LoopOutputConfig &cfg,
                         uint64_t dispatch_id, const LoopGroup &g,
                         uint32_t wave_count) {
  if (cfg.aggregate_wave) {
    const char *record_type = g.is_aggregate ? "aggregate" : "wave";
    writef(out, "{},{},", record_type, dispatch_id);
    if (g.is_aggregate)
      writef(out, ",,,");
    else
      writef(out, "{},{},{},", g.cu, g.simd, g.wave_id);
    writef(out, "{},", wave_count);
  } else {
    writef(out, "{},{},{},{},", dispatch_id, g.cu, g.simd, g.wave_id);
  }
}

void dump_csv_loop_rows(
    TraceBuf &out, const LoopOutputConfig &cfg, uint64_t dispatch_id,
    const LoopGroup &g, const LoopRecord &lr,
    const std::vector<LoopInstInfo> &insts) {
  const auto &e = lr.back_edge;
  const auto &s = lr.stats;
  auto sum = compute_loop_summary(s, insts);
  auto print_loop_fields = [&]() {
    writef(out, "0x{:x},0x{:x},{},{},{},{},{},{:.2f},{:.2f},{:.2f},",
           e.target_addr, e.source_addr, e.code_object_id, s.iteration_count,
           s.total_duration, s.total_stall, s.total_idle, sum.avg_dur,
           sum.avg_stall, sum.avg_idle);
    if (cfg.show_mfma_coexec)
      writef(out, "{:.2f},", sum.avg_bubble);
    writef(out, "{},{},{:.4f}", sum.mfma_count, sum.mfma_time, sum.mfma_util);
  };

  if (!cfg.show_contents) {
    dump_csv_row_prefix(out, cfg, dispatch_id, g, lr.wave_count);
    print_loop_fields();
    writef(out, "\n");
    return;
  }

  for (const auto &ii : insts) {
    dump_csv_row_prefix(out, cfg, dispatch_id, g, lr.wave_count);
    print_loop_fields();
    writef(out, ",0x{:x},\"{}\",{},", ii.addr, csv_escape(ii.inst_str), ii.n);
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
        writef(out, ",,,,");
        if (cfg.show_mfma_coexec)
          writef(out, ",");
        writef(out, ",");
      }
      writef(out, "\n");
    } else {
      writef(out, "{},{},{},{}", ii.dur, ii.stall, ii.idle, ii.issue);
      if (cfg.show_mfma_coexec)
        writef(out, ",{}", ii.bubble);
      writef(out, ",{}\n", ii.rel_issue);
    }
  }
}

// Emits per-trace dispatch entries separated by ",\n" with no leading or
// trailing comma. Caller is responsible for the surrounding "[\n" / "\n]\n"
// and for inserting ",\n" between non-empty per-trace bodies.
// Per-trace dump state, threaded through dump_loops*.  Holds references —
// owners are run_trace (cache, out) and trace_decoder_context (others).
struct DumpCtx {
  TraceBuf &out;
  my_rocperf_tool::Disassembler &disas;
  my_rocperf_tool::InstStatistics &stats;
  const LoopOutputConfig &cfg;
  const PerLoopMap &bubbles;
  const PerLoopMap &rel_issues;
  LoopInstCache &cache;
  uint64_t dispatch_id;
};

// Emits per-trace dispatch entries separated by ",\n" with no leading or
// trailing comma. Caller is responsible for the surrounding "[\n" / "\n]\n"
// and for inserting ",\n" between non-empty per-trace bodies.
void dump_loops_json(const DumpCtx &ctx,
                     const std::vector<LoopGroup> &groups) {
  bool first_entry = true;
  for (const auto &g : groups) {
    if (!first_entry) writef(ctx.out, ",\n");
    first_entry = false;
    if (g.is_aggregate)
      writef(ctx.out,
             "  {{\"dispatch_id\":{},\"aggregate\":true,\"loops\":[\n",
             ctx.dispatch_id);
    else
      writef(ctx.out,
             "  {{\"dispatch_id\":{},\"cu\":{},\"simd\":{},\"wave_id\":{},"
             "\"loops\":[\n",
             ctx.dispatch_id, g.cu, g.simd, g.wave_id);
    bool first_loop = true;
    for (const auto &lr : g.loops) {
      if (!first_loop) writef(ctx.out, ",\n");
      first_loop = false;
      const auto &insts = get_loop_insts(ctx.cache, ctx.disas, ctx.stats,
                                         lr.back_edge, ctx.bubbles,
                                         ctx.rel_issues);
      auto wave_count = g.is_aggregate
                            ? std::optional<uint32_t>(lr.wave_count)
                            : std::nullopt;
      dump_loop_json(ctx.out, lr.back_edge, lr.stats, insts, ctx.cfg,
                     wave_count);
    }
    writef(ctx.out, "\n  ]}}");
  }
}

void dump_loops_csv(const DumpCtx &ctx,
                    const std::vector<LoopGroup> &groups) {
  for (const auto &g : groups) {
    for (const auto &lr : g.loops) {
      const auto &insts = get_loop_insts(ctx.cache, ctx.disas, ctx.stats,
                                         lr.back_edge, ctx.bubbles,
                                         ctx.rel_issues);
      dump_csv_loop_rows(ctx.out, ctx.cfg, ctx.dispatch_id, g, lr, insts);
    }
  }
}

void dump_loop_text_stats(TraceBuf &out,
                          const my_rocperf_tool::LoopStats &s,
                          const std::vector<LoopInstInfo> &insts,
                          const LoopOutputConfig &cfg) {
  auto sum = compute_loop_summary(s, insts);
  writef(out, "    avg/iter: dur={:.2f} stall={:.2f} idle={:.2f}", sum.avg_dur,
         sum.avg_stall, sum.avg_idle);
  if (cfg.show_mfma_coexec)
    writef(out, " bubble={:.2f}", sum.avg_bubble);
  writef(out, " mfma_time={} ({} mfma * 16) mfma_util={:.1f}%\n",
         sum.mfma_time, sum.mfma_count, sum.mfma_util * 100.0);
  if (!cfg.show_contents)
    return;

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
  if (cfg.show_addr)
    writef(out, "{:<14} ", "addr");
  writef(out, "instruction\n");

  uint64_t max_bubble = 0;
  if (cfg.color_mfma_bubble) {
    for (const auto &ii : insts)
      if (ii.bubble > max_bubble)
        max_bubble = ii.bubble;
  }
  for (const auto &ii : insts) {
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
      if (cfg.show_mfma_coexec)
        writef(out, "{:<10} ", "");
      writef(out, "{:<10} ", "");
    }
    if (cfg.show_addr)
      writef(out, "0x{:x}:  ", ii.addr);
    writef(out, "{}", ii.inst_str);
    if (cfg.color_mfma_bubble && ii.has_stats)
      writef(out, "\033[0m");
    writef(out, "\n");
  }
}

void dump_loops_text(const DumpCtx &ctx,
                     const std::vector<LoopGroup> &groups) {
  for (const auto &g : groups) {
    if (g.is_aggregate)
      writef(ctx.out, "=== Aggregate Across Waves (DispatchID={}) ===\n",
             ctx.dispatch_id);
    else
      writef(ctx.out,
             "=== Wave (DispatchID={}, CU={}, SIMD={}, WaveID={}) ===\n",
             ctx.dispatch_id, g.cu, g.simd, g.wave_id);
    for (const auto &lr : g.loops) {
      const auto &e = lr.back_edge;
      const auto &s = lr.stats;
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
      const auto &insts = get_loop_insts(ctx.cache, ctx.disas, ctx.stats, e,
                                         ctx.bubbles, ctx.rel_issues);
      dump_loop_text_stats(ctx.out, s, insts, ctx.cfg);
    }
  }
}

void dump_loops(
    const DumpCtx &ctx,
    const std::vector<my_rocperf_tool::WaveLoopInfo> &wave_loops) {
  auto groups = build_loop_groups(wave_loops, ctx.cfg);
  switch (ctx.cfg.format) {
  case LoopOutputFormat::Json: dump_loops_json(ctx, groups); break;
  case LoopOutputFormat::Csv:  dump_loops_csv(ctx, groups);  break;
  case LoopOutputFormat::Text: dump_loops_text(ctx, groups); break;
  }
}

int run_trace(TraceBuf &out,
              const my_rocperf_tool::AttOutputDir &out_dir,
              const my_rocperf_tool::AttPath &att_path,
              my_rocperf_tool::Disassembler &disas,
              const LoopOutputConfig &cfg,
              std::optional<uint64_t> dispatch_id_override) {
  trace_decoder_context decoder_ctx(disas);
  {
    PHASE_TIMER("trace setup + I/O");
    auto [att_file_content, att_file_size] = out_dir.read_att_data(att_path);
    decoder_ctx.first_run = true;
    decoder_ctx.att_file_content = std::move(att_file_content);
    decoder_ctx.att_file_size = att_file_size;
    decoder_ctx.dispatch_id = dispatch_id_override.value_or(att_path.dispatch_id);
    decoder_ctx.detect_loops_flag = absl::GetFlag(FLAGS_detect_loops);
    decoder_ctx.verify_duration = absl::GetFlag(FLAGS_verify_trace_duration);
    decoder_ctx.mfma_coexec_flag = absl::GetFlag(FLAGS_mfma_coexec);
  }

  {
    PHASE_TIMER("trace decode");
    auto parse_res = rocprof_trace_decoder_parse_data(
        att_decoder_se_data_callback, att_decoder_trace_callback,
        att_decoder_isa_callback, &decoder_ctx);
    assert(parse_res == ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);
  }

  {
    PHASE_TIMER("dump_states (verify + decode_all)");
    dump_states(decoder_ctx.disas, decoder_ctx.stats,
               absl::GetFlag(FLAGS_decode_all_insts));
  }

  {
    PHASE_TIMER("sort wave_loops");
    std::sort(decoder_ctx.wave_loops.begin(), decoder_ctx.wave_loops.end(),
              [](const my_rocperf_tool::WaveLoopInfo &a,
                 const my_rocperf_tool::WaveLoopInfo &b) {
                return std::tie(a.cu, a.simd, a.wave_id) <
                       std::tie(b.cu, b.simd, b.wave_id);
              });
  }

  {
    PHASE_TIMER("dump_loops");
    if (decoder_ctx.detect_loops_flag) {
      LoopInstCache cache;
      DumpCtx dump_ctx{out, decoder_ctx.disas, decoder_ctx.stats, cfg,
                       decoder_ctx.bubble_totals,
                       decoder_ctx.rel_issue_totals, cache,
                       decoder_ctx.dispatch_id};
      dump_loops(dump_ctx, decoder_ctx.wave_loops);
    }
  }
  return 0;
}

int run_main(const std::string &att_output_dir_path) {
  PHASE_TIMER("TOTAL");

  my_rocperf_tool::AttOutputDir out_dir(att_output_dir_path);
  if (out_dir.att_paths.empty()) {
    std::cerr << "no ATT trace files found in " << att_output_dir_path << "\n";
    return 2;
  }

  auto object_load_bases = out_dir.read_load_bases();
  auto cfg = LoopOutputConfig::from_flags();
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

  auto load_code_objects = [&](my_rocperf_tool::Disassembler &d) {
    for (const auto &obj_file : out_dir.code_objects) {
      auto load_base_it =
          object_load_bases.find(static_cast<int>(obj_file.id));
      assert(load_base_it != object_load_bases.end());
      d.addCodeObject(obj_file.id, obj_file.path, load_base_it->second);
    }
  };

  // Per-trace output buffers. Indexed in dispatch_id order; workers may
  // complete out of order. fmt::memory_buffer owns its storage (RAII), so
  // exceptions between fill and emit don't leak.
  // Memory note: chunks hold all per-trace output until emit. For
  // aggregate_wave_only this is small; --dump_loop_contents over thousands
  // of traces could reach hundreds of MB.
  std::vector<TraceBuf> chunks(n_traces);

  auto process_one = [&](my_rocperf_tool::Disassembler &disas, size_t i) {
    run_trace(chunks[i], out_dir, out_dir.att_paths[i], disas, cfg,
              dispatch_id_override);
  };

  // Always process trace[0] serially first. Two reasons:
  // 1. The decoder library lazy-inits a process-wide singleton trie
  //    (Trie::root_trie in rocprof-trace-decoder/source/trie.cpp) on its
  //    first call, guarded only by a non-atomic bool. Concurrent first
  //    calls race on its std::unordered_map writes — UB. After the first
  //    call returns, std::thread construction publishes those writes to
  //    workers, which then only read.
  // 2. The serial Disassembler then becomes the main thread's worker, so
  //    no per-worker setup work is wasted.
  my_rocperf_tool::Disassembler main_disas;
  {
    PHASE_TIMER("load code objects + warmup");
    load_code_objects(main_disas);
    process_one(main_disas, 0);
  }

  if (n_traces > 1) {
    PHASE_TIMER("parallel decode");
    std::atomic<size_t> next{1};
    auto worker_loop = [&](my_rocperf_tool::Disassembler &disas) {
      while (true) {
        size_t i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= n_traces) break;
        process_one(disas, i);
      }
    };
    std::vector<std::thread> workers;
    workers.reserve(jobs - 1);
    for (size_t w = 1; w < jobs; w++) {
      workers.emplace_back([&]() {
        my_rocperf_tool::Disassembler local_disas;
        load_code_objects(local_disas);
        worker_loop(local_disas);
      });
    }
    worker_loop(main_disas);
    for (auto &t : workers) t.join();
  }

  // Emit prologue + chunks (in order) + epilogue.
  if (cfg.format == LoopOutputFormat::Json)
    std::fputs("[\n", stdout);
  else if (cfg.format == LoopOutputFormat::Csv) {
    TraceBuf header;
    dump_csv_header(header, cfg);
    std::fwrite(header.data(), 1, header.size(), stdout);
  }
  bool first_nonempty = true;
  for (const auto &c : chunks) {
    if (c.size() == 0) continue;
    if (cfg.format == LoopOutputFormat::Json && !first_nonempty)
      std::fputs(",\n", stdout);
    std::fwrite(c.data(), 1, c.size(), stdout);
    first_nonempty = false;
  }
  if (cfg.format == LoopOutputFormat::Json)
    std::fputs("\n]\n", stdout);
  return 0;
}

int main(int argc, char *argv[]) {
  absl::ParseCommandLine(argc, argv);
  std::optional<std::string> att_output_dir =
      absl::GetFlag(FLAGS_att_output_dir);
  if (!att_output_dir.has_value()) {
    std::cerr << "--att_output_dir not specified\n";
    return 2;
  }
  int LLVMArgc = 1;
  const char **LLVMArgv = const_cast<const char **>(argv);
  my_rocperf_tool::InitLLVM X(LLVMArgc, LLVMArgv);
  return run_main(*att_output_dir);
}
