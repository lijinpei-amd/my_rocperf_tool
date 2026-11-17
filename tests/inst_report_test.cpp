#include "my_rocperf_tool/inst_report.h"
#include "my_rocperf_tool/report_document.h"
#include "report_test_util.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <array>

namespace my_rocperf_tool {
namespace {

constexpr std::string_view kMfmaText =
    "v_mfma_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7], v[0:3]";
constexpr std::string_view kAddText = "s_add_u32 s0, s0, 1";

TraceInstReport trace_inst_report(AggregateScope scope, bool dynamic = false) {
  TraceInstReport r;
  r.prefix = R"(run "1",a)";
  r.dispatch_id = 7;
  r.se_id = 2;
  InstGroup g{scope, {1, 0, 2}, 0, {1, 4, 16, 4, 18, 3, 1, 1.0}, {}};
  g.rows = {{{0x100, 1},
             dynamic ? 1u : 0u,
             InstKind::mfma(16),
             kMfmaText,
             {2, 2, 8, 2, 0, 6, 0, 0}},
            {{0x104, 1},
             dynamic ? 2u : 0u,
             InstKind{},
             kAddText,
             {2, 1, 10, 4, 12, 6, 3, 25}}};
  if (scope == AggregateScope::All) {
    g.slot = {};
    g.summary.waves = 12;
  } else if (scope == AggregateScope::SimdSlot)
    g.summary.waves = 3;
  r.groups.push_back(g);
  if (scope != AggregateScope::All) {
    g.slot = {3, 1, 0};
    g.wave_index = 1;
    r.groups.push_back(g);
  }
  return r;
}

ReportChunk render(const ReportConfig& cfg, const TraceInstReport& report) {
  ReportChunk out;
  render_inst_report(out, cfg, report);
  return out;
}

std::string text(const ReportConfig& cfg, const TraceInstReport& report) {
  auto chunk = render(cfg, report);
  return std::string(chunk.data(), chunk.size());
}

std::string document(const ReportConfig& cfg,
                     llvm::ArrayRef<ReportChunk> chunks) {
  std::string out;
  write_report_document(cfg, chunks,
                        [&](std::string_view piece) { out += piece; });
  return out;
}

std::vector<ReportConfig> option_matrix(ReportFormat format) {
  std::vector<ReportConfig> result;
  for (auto report : {ReportKind::StaticInst, ReportKind::DynamicInst})
    for (auto scope :
         {AggregateScope::Wave, AggregateScope::SimdSlot, AggregateScope::All})
      for (unsigned bits = 0; bits < 8; bits++) {
        ReportConfig cfg;
        cfg.format = format;
        cfg.report = report;
        cfg.aggregate = scope;
        cfg.show_avg = bits & 1;
        cfg.show_mfma_coexec = bits & 2;
        cfg.show_prefix = bits & 4;
        result.push_back(cfg);
      }
  return result;
}

TEST(InstReportTest, TextStaticMatchesGolden) {
  ReportConfig cfg;
  cfg.report = ReportKind::StaticInst;
  auto report = trace_inst_report(AggregateScope::Wave);
  report.groups.resize(1);
  EXPECT_EQ(text(cfg, report),
            R"(=== Wave (DispatchID=7, SE=2, CU=1, SIMD=0, WaveID=2) ===
  avg/wave: records=4.00 dur=16.00 stall=4.00 idle=18.00 mfma=1.00 mfma_util=100.0%
    n      waves  dur        stall      idle       issue    avg_cycle  addr           instruction
    2      2      8          2          0          6        0.00       0x100:  v_mfma_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7], v[0:3]
    2      1      10         4          12         6        12.50      0x104:  s_add_u32 s0, s0, 1
)");
}

TEST(InstReportTest, TextDynamicAddsTheOccurrenceColumn) {
  ReportConfig cfg;
  cfg.report = ReportKind::DynamicInst;
  auto report = trace_inst_report(AggregateScope::Wave, true);
  report.groups.resize(1);
  EXPECT_EQ(text(cfg, report),
            R"(=== Wave (DispatchID=7, SE=2, CU=1, SIMD=0, WaveID=2) ===
  avg/wave: records=4.00 dur=16.00 stall=4.00 idle=18.00 mfma=1.00 mfma_util=100.0%
    n      waves  dur        stall      idle       issue    avg_cycle  occ   addr           instruction
    2      2      8          2          0          6        0.00       1     0x100:  v_mfma_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7], v[0:3]
    2      1      10         4          12         6        12.50      2     0x104:  s_add_u32 s0, s0, 1
)");
}

TEST(InstReportTest, TextHeadingsPerScope) {
  const std::array<std::string, 3> headings{
      "=== Wave (DispatchID=7, SE=2, CU=1, SIMD=0, WaveID=2) ===",
      "=== SIMD Slot (DispatchID=7, SE=2, CU=1, SIMD=0, WaveID=2, 3 waves) ===",
      "=== Aggregate Across Waves (DispatchID=7, SE=2, 12 waves) ==="};
  size_t i = 0;
  for (auto scope :
       {AggregateScope::Wave, AggregateScope::SimdSlot, AggregateScope::All}) {
    ReportConfig cfg;
    cfg.report = ReportKind::StaticInst;
    cfg.aggregate = scope;
    auto report = trace_inst_report(scope);
    auto out = text(cfg, report);
    EXPECT_EQ(lines(out)[0], headings[i]);
    cfg.show_prefix = true;
    out = text(cfg, report);
    auto expected = headings[i];
    expected.insert(expected.find("DispatchID"), R"(Prefix=run "1",a, )");
    EXPECT_EQ(lines(out)[0], expected);
    i++;
  }
}

TEST(InstReportTest, TextBannersATraceWithoutGroups) {
  ReportConfig cfg;
  cfg.report = ReportKind::StaticInst;
  TraceInstReport r;
  r.dispatch_id = 7;
  r.se_id = 2;
  EXPECT_EQ(text(cfg, r), "=== DispatchID=7, SE=2: no waves analyzed ===\n");
  r.simd_filter = 1;
  r.counts.waves_filtered = 3;
  EXPECT_EQ(text(cfg, r),
            "=== DispatchID=7, SE=2: no waves analyzed (3 waves on other SIMDs "
            "skipped by --simd=1) ===\n");
  for (auto format : {ReportFormat::Csv, ReportFormat::Json}) {
    cfg.format = format;
    EXPECT_EQ(render(cfg, r).size(), 0u);
  }
}

TEST(InstReportTest, CsvRowsHaveTheHeadersColumns) {
  for (const auto& cfg : option_matrix(ReportFormat::Csv)) {
    bool dynamic = cfg.report == ReportKind::DynamicInst;
    auto report = trace_inst_report(cfg.aggregate, dynamic);
    std::vector<ReportChunk> chunks;
    chunks.push_back(render(cfg, report));
    auto doc = document(cfg, chunks);
    auto rows = lines(doc);
    auto header = split_csv(rows[0]);
    ASSERT_EQ(rows.size(), 1 + 2 * report.groups.size());
    EXPECT_EQ(
        std::find(header.begin(), header.end(), "occurrence") != header.end(),
        dynamic);
    EXPECT_EQ(
        std::find(header.begin(), header.end(), "record_type") != header.end(),
        cfg.aggregate != AggregateScope::Wave);
    EXPECT_EQ(
        std::find(header.begin(), header.end(), "avg_bubble") != header.end(),
        cfg.show_avg && cfg.show_mfma_coexec);
    EXPECT_EQ(std::find(header.begin(), header.end(), "bubble") != header.end(),
              !cfg.show_avg && cfg.show_mfma_coexec);
    for (size_t i = 1; i < rows.size(); i++) {
      const auto fields = split_csv(rows[i]);
      ASSERT_EQ(fields.size(), header.size());
      EXPECT_EQ(fields[column(header, "se_id")], "2");
      EXPECT_EQ(fields[column(header, "instruction")],
                i % 2 ? kMfmaText : kAddText);
      EXPECT_EQ(fields[column(header, "n")], "2");
      EXPECT_EQ(fields[column(header, "exec_waves")], i % 2 ? "2" : "1");
      EXPECT_EQ(fields[column(header, "code_object")], "1");
      if (cfg.show_prefix)
        EXPECT_EQ(fields[column(header, "prefix")], R"(run "1",a)");
      if (dynamic)
        EXPECT_EQ(fields[column(header, "occurrence")], i % 2 ? "1" : "2");
      if (cfg.aggregate != AggregateScope::Wave) {
        EXPECT_EQ(
            fields[column(header, "record_type")],
            cfg.aggregate == AggregateScope::All ? "aggregate" : "simd_slot");
        EXPECT_EQ(fields[column(header, "wave_count")],
                  cfg.aggregate == AggregateScope::All ? "12" : "3");
      }
      if (cfg.aggregate == AggregateScope::All) {
        for (auto name : {"cu", "simd", "wave_id"})
          EXPECT_EQ(fields[column(header, name)], "");
      }
      if (cfg.show_avg) {
        EXPECT_EQ(fields[column(header, "avg_cycle")],
                  i % 2 ? "0.00" : "12.50");
        EXPECT_EQ(fields[column(header, "avg_dur")], i % 2 ? "4.00" : "5.00");
        if (cfg.show_mfma_coexec)
          EXPECT_EQ(fields[column(header, "avg_bubble")],
                    i % 2 ? "0.00" : "1.50");
      } else {
        EXPECT_EQ(fields[column(header, "rel_issue")], i % 2 ? "0" : "25");
        EXPECT_EQ(fields[column(header, "dur")], i % 2 ? "8" : "10");
      }
    }
  }
}

TEST(InstReportTest, JsonDocumentParses) {
  for (const auto& cfg : option_matrix(ReportFormat::Json)) {
    bool dynamic = cfg.report == ReportKind::DynamicInst;
    auto report = trace_inst_report(cfg.aggregate, dynamic);
    std::vector<ReportChunk> chunks;
    chunks.push_back(render(cfg, report));
    chunks.push_back(render(cfg, TraceInstReport{}));
    EXPECT_EQ(chunks.back().size(), 0u);
    report.dispatch_id = 8;
    chunks.push_back(render(cfg, report));
    auto doc = document(cfg, chunks);
    auto parsed = llvm::json::parse(doc);
    ASSERT_TRUE(static_cast<bool>(parsed))
        << llvm::toString(parsed.takeError()) << doc;
    auto* entries = parsed->getAsArray();
    ASSERT_NE(entries, nullptr);
    ASSERT_EQ(entries->size(), 2 * report.groups.size());
    for (const auto& entry : *entries) {
      auto* g = entry.getAsObject();
      EXPECT_EQ(g->getInteger("se_id"), 2);
      EXPECT_EQ(g->get("aggregate") != nullptr,
                cfg.aggregate == AggregateScope::All);
      EXPECT_EQ(g->get("simd_slot") != nullptr,
                cfg.aggregate == AggregateScope::SimdSlot);
      if (cfg.show_prefix)
        EXPECT_EQ(g->getString("prefix").value_or("").str(), R"(run "1",a)");
      auto* sum = g->getObject("summary");
      ASSERT_NE(sum, nullptr);
      EXPECT_EQ(sum->getInteger("waves"),
                cfg.aggregate == AggregateScope::All        ? 12
                : cfg.aggregate == AggregateScope::SimdSlot ? 3
                                                            : 1);
      EXPECT_EQ(sum->getNumber("avg_dur"), 16);
      EXPECT_EQ(sum->get("avg_bubble") != nullptr, cfg.show_mfma_coexec);
      auto* insts = g->getArray("instructions");
      ASSERT_NE(insts, nullptr);
      ASSERT_EQ(insts->size(), 2u);
      for (size_t i = 0; i < 2; i++) {
        auto* inst = (*insts)[i].getAsObject();
        EXPECT_EQ(inst->get("occurrence") != nullptr, dynamic);
        EXPECT_EQ(inst->getInteger("n"), 2);
        EXPECT_EQ(inst->getInteger("exec_waves"), i == 0 ? 2 : 1);
        EXPECT_EQ(inst->getString("inst").value_or("").str(),
                  i == 0 ? kMfmaText : kAddText);
        if (cfg.show_avg)
          EXPECT_EQ(inst->getNumber("avg_cycle"), i == 0 ? 0 : 12.5);
        else
          EXPECT_EQ(inst->getInteger("rel_issue"), i == 0 ? 0 : 25);
      }
    }
  }
}

TEST(InstReportTest, RowFlagsControlStatsAndTextAddresses) {
  ReportConfig cfg;
  cfg.report = ReportKind::DynamicInst;
  cfg.show_stats = false;
  cfg.show_addr = false;
  auto report = trace_inst_report(AggregateScope::Wave, true);
  report.groups.resize(1);
  auto out = text(cfg, report);
  EXPECT_EQ(out.find("avg_cycle"), std::string::npos);
  EXPECT_EQ(out.find("0x100"), std::string::npos);
  EXPECT_NE(out.find("occ"), std::string::npos);
  EXPECT_NE(out.find(kMfmaText), std::string::npos);
  cfg.format = ReportFormat::Json;
  std::vector<ReportChunk> chunks;
  chunks.push_back(render(cfg, report));
  auto parsed = llvm::json::parse(document(cfg, chunks));
  ASSERT_TRUE(static_cast<bool>(parsed)) << llvm::toString(parsed.takeError());
  auto* row =
      (*(*parsed->getAsArray())[0].getAsObject()->getArray("instructions"))[0]
          .getAsObject();
  EXPECT_EQ(row->get("n"), nullptr);
  EXPECT_EQ(row->get("exec_waves"), nullptr);
  EXPECT_EQ(row->getString("addr").value_or("").str(), "0x100");
  EXPECT_EQ(row->getInteger("occurrence"), 1);
  cfg.format = ReportFormat::Csv;
  chunks.clear();
  chunks.push_back(render(cfg, report));
  out = document(cfg, chunks);
  auto header = split_csv(lines(out)[0]);
  EXPECT_NE(std::find(header.begin(), header.end(), "n"), header.end());
  EXPECT_NE(std::find(header.begin(), header.end(), "inst_addr"), header.end());
}

}  // namespace
}  // namespace my_rocperf_tool
