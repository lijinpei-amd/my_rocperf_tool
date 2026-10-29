#include "my_rocperf_tool/loop_report.h"

#include "my_rocperf_tool/inst_kind.h"
#include "my_rocperf_tool/loop_detection.h"
#include "my_rocperf_tool/loop_profile.h"

#include "gtest/gtest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace my_rocperf_tool {
namespace {

constexpr BackEdge kEdge{/*code_object_id=*/1, /*source_addr=*/0x110,
                         /*target_addr=*/0x100};
// The body instruction every pass branches over.
constexpr uint64_t kSkipped = 0x10c;

// A loop body as LLVM prints it: operands separated by commas, which CSV has
// to quote.
LoopBody list_loop() {
  return {
      {0x100, "v_mfma_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7], v[0:3]",
       InstKind::Mfma},
      {0x108, "s_cbranch_scc1 1", InstKind::Other},
      {kSkipped, "s_add_u32 s0, s0, 1", InstKind::Other},
      {0x110, "s_cbranch_scc0 65531", InstKind::Other},
  };
}

// A record of the loop over `iterations` passes, an even number, so that the
// per-iteration averages have a fraction: 32.5 cycles, 6.5 of them stalled.
ReportedLoop loop_record(const LoopBody& body, uint32_t iterations,
                         uint32_t wave_count, std::optional<uint64_t> wave) {
  ReportedLoop loop;
  loop.record.back_edge = kEdge;
  loop.record.stats.iteration_count = iterations;
  loop.record.stats.total_inst_count = 3 * iterations;
  loop.record.stats.total_duration = 65 * iterations / 2;
  loop.record.stats.total_stall = 13 * iterations / 2;
  loop.record.stats.total_idle = 5 * iterations / 2;
  loop.record.wave_count = wave_count;
  loop.record.wave = wave;
  for (const auto& inst : body) {
    const uint64_t n = inst.vaddr == kSkipped ? 0 : iterations;
    loop.metrics.rows.push_back({inst.vaddr, inst.text, inst.kind, n,
                                 /*dur=*/9 * n, /*stall=*/2 * n, /*idle=*/n,
                                 /*issue=*/7 * n, /*bubble=*/3 * n,
                                 /*rel_issue=*/5 * n, /*has_stats=*/n > 0});
  }
  loop.metrics.summary = {
      /*avg_dur=*/32.5,        /*avg_stall=*/6.5,
      /*avg_idle=*/2.5,        /*avg_bubble=*/9,
      /*mfma_util=*/16 / 32.5, /*mfma_count=*/1,
      /*mfma_time=*/16};
  return loop;
}

// Two waves that ran the loop and, with `aggregate`, the aggregate over them.
// The prefix needs escaping in CSV and JSON.
TraceLoopReport trace_report(const LoopBody& body, bool aggregate,
                             uint64_t dispatch_id) {
  TraceLoopReport report;
  report.prefix = R"(run "1",a)";
  report.dispatch_id = dispatch_id;
  report.se_id = 2;
  report.groups.push_back({false, 1, 0, 2, {loop_record(body, 4, 1, 0)}});
  report.groups.push_back({false, 3, 1, 0, {loop_record(body, 4, 1, 1)}});
  if (aggregate)
    report.groups.push_back(
        {true, 0, 0, 0, {loop_record(body, 8, 2, std::nullopt)}});
  return report;
}

LoopReportChunk render(const LoopOutputConfig& cfg,
                       const TraceLoopReport& report) {
  LoopReportChunk chunk;
  render_loop_report(chunk, cfg, report);
  return chunk;
}

std::string to_string(const LoopReportChunk& chunk) {
  return std::string(chunk.data(), chunk.size());
}

std::string document(const LoopOutputConfig& cfg,
                     llvm::ArrayRef<LoopReportChunk> chunks) {
  std::string doc;
  write_loop_report_document(cfg, chunks,
                             [&](std::string_view piece) { doc += piece; });
  return doc;
}

// Every combination of the options that add or drop columns and keys.
std::vector<LoopOutputConfig> option_matrix(LoopOutputFormat format) {
  std::vector<LoopOutputConfig> configs;
  for (unsigned bits = 0; bits < 32; bits++) {
    LoopOutputConfig cfg;
    cfg.format = format;
    cfg.show_avg = bits & 1;
    cfg.show_mfma_coexec = bits & 2;
    cfg.show_contents = bits & 4;
    cfg.aggregate_wave = bits & 8;
    cfg.show_prefix = bits & 16;
    configs.push_back(cfg);
  }
  return configs;
}

std::string describe(const LoopOutputConfig& cfg) {
  return "show_avg=" + std::to_string(cfg.show_avg) +
         " show_mfma_coexec=" + std::to_string(cfg.show_mfma_coexec) +
         " show_contents=" + std::to_string(cfg.show_contents) +
         " aggregate_wave=" + std::to_string(cfg.aggregate_wave) +
         " show_prefix=" + std::to_string(cfg.show_prefix);
}

// The lines of a document that ends each line with '\n'.
std::vector<std::string_view> lines(std::string_view doc) {
  std::vector<std::string_view> result;
  while (!doc.empty()) {
    size_t end = doc.find('\n');
    EXPECT_NE(end, std::string_view::npos) << "unterminated line: " << doc;
    if (end == std::string_view::npos) end = doc.size() - 1;
    result.push_back(doc.substr(0, end));
    doc.remove_prefix(end + 1);
  }
  return result;
}

// Just enough of RFC 4180: a quoted field may hold commas, and "" in it is a
// quote.
std::vector<std::string> split_csv(std::string_view line) {
  std::vector<std::string> fields(1);
  bool quoted = false;
  for (size_t i = 0; i < line.size(); i++) {
    const char c = line[i];
    if (quoted) {
      if (c != '"') {
        fields.back() += c;
      } else if (i + 1 < line.size() && line[i + 1] == '"') {
        fields.back() += '"';
        i++;
      } else {
        quoted = false;
      }
    } else if (c == '"') {
      quoted = true;
    } else if (c == ',') {
      fields.emplace_back();
    } else {
      fields.back() += c;
    }
  }
  EXPECT_FALSE(quoted) << "unterminated quote: " << line;
  return fields;
}

size_t column(const std::vector<std::string>& header, std::string_view name) {
  for (size_t i = 0; i < header.size(); i++)
    if (header[i] == name) return i;
  ADD_FAILURE() << "no column " << name;
  return 0;
}

TEST(LoopReportTest, CsvRowsHaveTheHeadersColumns) {
  auto body = list_loop();
  for (const auto& cfg : option_matrix(LoopOutputFormat::Csv)) {
    SCOPED_TRACE(describe(cfg));
    std::vector<LoopReportChunk> chunks;
    chunks.push_back(render(cfg, trace_report(body, cfg.aggregate_wave, 7)));

    auto doc = document(cfg, chunks);

    auto rows = lines(doc);
    ASSERT_FALSE(rows.empty());
    const auto header = split_csv(rows[0]);
    const size_t records = cfg.aggregate_wave ? 3 : 2;
    const size_t rows_per_record = cfg.show_contents ? body.size() : 1;
    ASSERT_EQ(rows.size(), 1 + records * rows_per_record);
    for (size_t i = 1; i < rows.size(); i++) {
      SCOPED_TRACE(rows[i]);
      const auto fields = split_csv(rows[i]);
      ASSERT_EQ(fields.size(), header.size());
      if (cfg.show_prefix)
        EXPECT_EQ(fields[column(header, "prefix")], R"(run "1",a)");
      if (!cfg.show_contents) continue;
      const auto& inst = body[(i - 1) % rows_per_record];
      EXPECT_EQ(fields[column(header, "instruction")], inst.text);
      if (inst.vaddr != kSkipped) continue;
      // Never executed: no average to give, but its sums are 0.
      const size_t n = column(header, "n");
      EXPECT_EQ(fields[n], "0");
      for (size_t f = n + 1; f < fields.size(); f++)
        EXPECT_EQ(fields[f], cfg.show_avg ? "" : "0") << header[f];
    }
  }
}

TEST(LoopReportTest, JsonDocumentParses) {
  auto body = list_loop();
  for (const auto& cfg : option_matrix(LoopOutputFormat::Json)) {
    SCOPED_TRACE(describe(cfg));
    std::vector<LoopReportChunk> chunks;
    chunks.push_back(render(cfg, trace_report(body, cfg.aggregate_wave, 7)));
    // A trace that printed nothing, between two that did.
    chunks.push_back(render(cfg, TraceLoopReport{}));
    ASSERT_EQ(chunks.back().size(), 0u);
    chunks.push_back(render(cfg, trace_report(body, cfg.aggregate_wave, 8)));

    auto doc = document(cfg, chunks);

    auto parsed = llvm::json::parse(doc);
    ASSERT_TRUE(static_cast<bool>(parsed))
        << llvm::toString(parsed.takeError()) << "\n"
        << doc;
    const auto* entries = parsed->getAsArray();
    ASSERT_NE(entries, nullptr);
    ASSERT_EQ(entries->size(), cfg.aggregate_wave ? 6u : 4u);
    for (const auto& entry : *entries) {
      const auto* loops = entry.getAsObject()->getArray("loops");
      ASSERT_NE(loops, nullptr);
      ASSERT_EQ(loops->size(), 1u);
      const auto* insts = (*loops)[0].getAsObject()->getArray("instructions");
      ASSERT_EQ(insts != nullptr, cfg.show_contents);
      if (!insts) continue;
      ASSERT_EQ(insts->size(), body.size());
      for (size_t i = 0; i < body.size(); i++) {
        const auto* inst = (*insts)[i].getAsObject();
        EXPECT_EQ(inst->getString("inst").value_or("").str(), body[i].text);
        // A never-executed instruction has no metrics to give.
        EXPECT_EQ(inst->get("n") != nullptr, body[i].vaddr != kSkipped);
      }
    }

    auto no_entries = llvm::json::parse(document(cfg, {}));
    ASSERT_TRUE(static_cast<bool>(no_entries))
        << llvm::toString(no_entries.takeError());
    ASSERT_NE(no_entries->getAsArray(), nullptr);
    EXPECT_TRUE(no_entries->getAsArray()->empty());
  }
}

TEST(LoopReportTest, TextHasAHeadingPerGroup) {
  auto body = list_loop();
  for (const auto& cfg : option_matrix(LoopOutputFormat::Text)) {
    SCOPED_TRACE(describe(cfg));
    auto report = trace_report(body, cfg.aggregate_wave, 7);

    auto text = to_string(render(cfg, report));

    size_t headings = 0;
    for (auto line : lines(text))
      if (line.substr(0, 4) == "=== ") headings++;
    EXPECT_EQ(headings, report.groups.size());
    EXPECT_EQ(text.find("no loops detected"), std::string::npos);
  }
}

TEST(LoopReportTest, TextMatchesGolden) {
  auto body = list_loop();
  LoopOutputConfig cfg;
  cfg.show_contents = true;
  auto report = trace_report(body, /*aggregate=*/false, 7);
  report.groups.resize(1);

  auto text = to_string(render(cfg, report));

  // The loop line truncates the averages the avg/iter line shows in full.
  EXPECT_EQ(text, R"(=== Wave (DispatchID=7, CU=1, SIMD=0, WaveID=2) ===
  Loop [0x100 -> 0x110] code_object=1: 4 iterations, 130 total cycles (avg 32/iter), stall 26 (avg 6/iter), idle 10
    avg/iter: dur=32.50 stall=6.50 idle=2.50 mfma_time=16 (1 mfma * 16) mfma_util=49.2%
    n      dur        stall      idle       issue    avg_cycle  addr           instruction
    4      36         8          4          28       5.00       0x100:  v_mfma_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7], v[0:3]
    4      36         8          4          28       5.00       0x108:  s_cbranch_scc1 1
                                                                0x10c:  s_add_u32 s0, s0, 1
    4      36         8          4          28       5.00       0x110:  s_cbranch_scc0 65531
)");
}

// A report as a trace without loops resolves it: no group, unless
// --aggregate_wave adds the aggregate's.
TraceLoopReport loop_free_report() {
  TraceLoopReport report;
  report.prefix = "run_a";
  report.dispatch_id = 7;
  report.se_id = 2;
  report.counts.waves = 5;
  report.counts.insts = 120;
  report.counts.waves_filtered = 3;
  return report;
}

TEST(LoopReportTest, TextBannersATraceWithoutGroups) {
  LoopOutputConfig cfg;
  auto report = loop_free_report();

  EXPECT_EQ(to_string(render(cfg, report)),
            "=== DispatchID=7, SE=2: no loops detected (5 waves, 120 "
            "instructions) ===\n");

  cfg.show_prefix = true;
  report.simd = 1;
  EXPECT_EQ(to_string(render(cfg, report)),
            "=== Prefix=run_a, DispatchID=7, SE=2: no loops detected (5 "
            "waves, 120 instructions; 3 waves on other SIMDs skipped by "
            "--simd=1) ===\n");
}

TEST(LoopReportTest, EmptyAggregateHeadingSuppressesTheBanner) {
  LoopOutputConfig cfg;
  cfg.aggregate_wave = true;
  auto report = loop_free_report();
  report.groups.push_back({true, 0, 0, 0, {}});

  EXPECT_EQ(to_string(render(cfg, report)),
            "=== Aggregate Across Waves (DispatchID=7) ===\n");
}

TEST(LoopReportTest, NoBannerOutsideText) {
  for (auto format : {LoopOutputFormat::Csv, LoopOutputFormat::Json}) {
    LoopOutputConfig cfg;
    cfg.format = format;

    EXPECT_EQ(render(cfg, loop_free_report()).size(), 0u);
  }
}

}  // namespace
}  // namespace my_rocperf_tool
