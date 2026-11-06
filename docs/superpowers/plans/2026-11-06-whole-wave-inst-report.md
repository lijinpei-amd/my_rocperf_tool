# Whole-wave Instruction Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add static and dynamic whole-wave instruction reports to `load_issue_stall`, selected by `--report`, grouped by `--aggregate=wave|simd_slot|all` for every report.

**Architecture:** A new `InstProfile` module accumulates per-PC or per-(PC, occurrence) sums over whole waves, fed by `WaveAnalysis` like `LoopProfile`; a new `InstReport` module renders it. Loop and instruction reports share one `ReportConfig`, one document writer and private formatting helpers. `LoopProfile` gains per-slot records so `--aggregate=simd_slot` works for loops too.

**Tech Stack:** C++17, LLVM ADT, fmt, abseil flags, googletest; CMake + Ninja.

**Spec:** `docs/superpowers/specs/2026-11-06-whole-wave-inst-report-design.md`

## Global Constraints

- C++17, `-fPIC -fno-rtti`, everything in namespace `my_rocperf_tool` (`DenseMapInfo` specializations in `llvm`).
- clang-format via pre-commit: run `pre-commit run --all-files` before each commit.
- Core-library modules take plain data and print nothing; only `tools/load_issue_stall.cpp` prints.
- Waves are streamed: nothing of a wave is kept past `add_wave` but running sums.
- Loop output must stay byte-identical for: plain run = `--report=loop_summary --aggregate=wave`; `--dump_loop_contents` = `--report=loop_detail --aggregate=wave`; `--aggregate_wave_only` = `--aggregate=all` (other flags renamed).
- No deprecated flag aliases. Unknown `--report` or `--aggregate` value: message on stderr, exit 2.
- Output must not depend on `--jobs`.
- Build: `cmake --build build`. Tests: `build/tests/my_rocperf_tool_tests --gtest_filter=<Suite>.*`, all: `ctest --test-dir build --output-on-failure`.
- Real traces for end-to-end checks (read-only; never write into them):
  - `T1=/raid/jinpli/workspace/home/jinpli/att`: 45 traces, 316 waves, 4609976 instructions, every dispatch has loops; ~0.3 s per run.
  - `T2=/raid/jinpli/workspace/home01/jinpli/development/bench_upstream_20261106/pro_a4w4_tokens/att`: 24 traces, 10524 waves, 51203324 instructions, two runs (prefix labels print), 3 of 6 dispatches loop-free (`out_fly/out:63284`, `64624`, `64628`: the unrolled-kernel case this feature is for); ~2 s per run. Under `--aggregate=wave` the instruction reports are ~50M rows (GBs, all held in memory until written): don't run that scope on T2.

## Review Focus

1. A wave whose records are all null-PC under the new reports: its group counts the wave, has no rows, averages 0, no division by zero. Test: Task 4 `WaveWithoutAttributedRecordsHasAnEmptyGroup`.
2. Several waves in one slot under `--aggregate=wave`: separate groups, ordered `(cu, simd, wave_id)` then decode order. Tests: Task 4 `GroupsFollowTheScope`; loops: Task 3 baseline diff.
3. Records with stall > duration (kept under `--verify_trace_duration=warn`): issue 0, `rel` from `time + stall`. Test: Task 4 `IssueIsZeroWhereStallExceedsDuration`.
4. `--simd=N` with `--aggregate=simd_slot`: only SIMD N's slots appear. Test: Task 5 `SimdFilterDropsWavesBeforeTheInstProfile`.
5. Multi-run directories (`show_prefix`) in the new reports' three formats. Test: Task 6 matrices include `show_prefix`.

---

### Task 0: Branch and baseline

**Files:** none changed. The branch `whole-wave-inst-report` already exists, with the spec and this plan committed on it.

- [ ] **Step 1: Switch to the branch and build**

```bash
git switch whole-wave-inst-report
cmake --build build && ctest --test-dir build --output-on-failure
```
Expected: build succeeds, all tests pass.

- [ ] **Step 2: Capture the old binary's loop output**

```bash
B=build/tools/load_issue_stall; O=/tmp/wwir-baseline
mkdir -p $O && cp $B $O/load_issue_stall.old
for t in 1 2; do
  T=$([ $t = 1 ] && echo /raid/jinpli/workspace/home/jinpli/att || echo /raid/jinpli/workspace/home01/jinpli/development/bench_upstream_20261106/pro_a4w4_tokens/att)
  for f in text csv json; do
    $B --att_output_dir=$T --loop_output_format=$f >$O/t$t.plain.$f 2>$O/t$t.plain.$f.err
    $B --att_output_dir=$T --loop_output_format=$f --dump_loop_contents --mfma_coexec >$O/t$t.detail.$f 2>$O/t$t.detail.$f.err
    $B --att_output_dir=$T --loop_output_format=$f --aggregate_wave_only --dump_loop_contents --loop_inst_avg >$O/t$t.all.$f 2>$O/t$t.all.$f.err
  done
done
ls -s $O
```
Expected: 36 non-empty files plus the binary copy.

---

### Task 1: Extract the MFMA shadow and issue arithmetic

Pure refactor: `LoopProfile` keeps its behaviour; `InstProfile` (Task 4) reuses it.

**Files:**
- Create: `include/my_rocperf_tool/mfma_shadow.h`, `lib/mfma_shadow.cpp`
- Modify: `include/my_rocperf_tool/loop_profile.h` (move constants and `MfmaShadowOverlap` out, include the new header), `lib/loop_profile.cpp:84-141` (`add_passes` uses the new helpers), `lib/CMakeLists.txt` (add `mfma_shadow.cpp`)

**Interfaces:**
- Produces, in `mfma_shadow.h`:
  - `kMfmaShadowCycles` (16), `kAttTimestampGranule` (4), `struct MfmaShadowOverlap`: moved verbatim from `loop_profile.h`, comments included.
  - `inline uint64_t issue_cycles(const rocprofiler_thread_trace_decoder_inst_t& inst)`: `max(duration - stall, 0)`.
  - `class MfmaShadow { public: uint64_t step(const rocprofiler_thread_trace_decoder_inst_t& inst, bool is_mfma, std::vector<MfmaShadowOverlap>& overlaps); private: int64_t end = 0; };` For an MFMA: appends an overlap if `issue_point < end - kAttTimestampGranule` (and `end > 0`), sets `end = max(end, issue_point + kMfmaShadowCycles)`, returns 0. Otherwise returns `min(duration, time + duration - end)` if positive, else 0. Exactly the logic now inline in `add_passes`.

- [ ] **Step 1: Move the code and switch `add_passes` to `issue_cycles` and one `MfmaShadow` per loop walk**

- [ ] **Step 2: Run the existing tests**

Run: `cmake --build build && ctest --test-dir build --output-on-failure`
Expected: all pass (the `WaveAnalysisMfmaTest.*` and `LoopProfileTest.*` cases pin the arithmetic).

- [ ] **Step 3: Commit**

```bash
git add include/my_rocperf_tool/mfma_shadow.h lib/mfma_shadow.cpp include/my_rocperf_tool/loop_profile.h lib/loop_profile.cpp lib/CMakeLists.txt
git commit -m "Extract the MFMA shadow walk from LoopProfile"
```

---

### Task 2: Per-slot loop records in LoopProfile

**Files:**
- Create: `include/my_rocperf_tool/aggregate_scope.h`
- Modify: `include/my_rocperf_tool/loop_profile.h`, `lib/loop_profile.cpp`, `tests/synthetic_wave.h` (`WaveBuilder` takes a wave slot)
- Test: `tests/loop_profile_test.cpp`

**Interfaces:**
- Produces, in `aggregate_scope.h`:
  - `enum class AggregateScope { Wave, SimdSlot, All };`
  - `struct WaveSlot { uint8_t cu = 0; uint8_t simd = 0; uint8_t wave_id = 0; bool operator==(const WaveSlot&) const; bool operator<(const WaveSlot&) const; };` ordered by `(cu, simd, wave_id)`; plus `llvm::DenseMapInfo<my_rocperf_tool::WaveSlot>`.
- Produces, in `loop_profile.h`: `LoopRecord` gains a last member `std::optional<WaveSlot> slot;`: a per-slot record's slot, nullopt otherwise; at most one of `wave` and `slot` is set (assert in `metrics`). `LoopProfile` keeps, besides today's totals, `llvm::DenseMap<std::pair<WaveSlot, BackEdge>, RecordTotals> slot_totals`, always with rows; `add_wave` files each loop under the slot `{loops.cu, loops.simd, loops.wave_id}`. `metrics` answers a slot record from it.
- Produces, in `synthetic_wave.h`: `explicit WaveBuilder(uint8_t simd = 0, uint8_t wave_id = 0)`, which sets `wave.wave_id`.

- [ ] **Step 1: Write the failing test**

```cpp
// Slot (0, 0, 1) runs waves 0, 1 and 3; the loop is detected in 0 (2 passes)
// and 1 (3 passes), not in 3 (1 pass). Slot (0, 1, 0) runs wave 2 (2 passes).
TEST(LoopProfileTest, SlotRecordSumsItsSlotsWavesWithTheLoop) {
  KindTable table{{{kLoopHeader, InstKind::Mfma}}};
  LoopProfile profile(/*mfma_coexec=*/true);
  auto passes = [](WaveBuilder& b, int n) {
    for (int i = 0; i < n; i++) b.iteration(/*branch_taken=*/i + 1 < n);
  };
  WaveBuilder w0(0, 1), w1(0, 1), w2(1, 0), w3(0, 1);
  passes(w0, 2); passes(w1, 3); passes(w2, 2); passes(w3, 1);
  auto l0 = add_wave(profile, w0, table, 0);
  auto l1 = add_wave(profile, w1, table, 1);
  auto l2 = add_wave(profile, w2, table, 2);
  auto l3 = add_wave(profile, w3, table, 3);
  ASSERT_TRUE(l3.loops.empty());
  auto body = list_body(kLoop, table.kinds);

  LoopStats slot_stats = l0.loops.at(0).stats;
  slot_stats += l1.loops.at(0).stats;
  auto slot = profile.metrics(
      {kLoop, slot_stats, 2, std::nullopt, WaveSlot{0, 0, 1}}, body);
  auto other = profile.metrics(
      {kLoop, l2.loops.at(0).stats, 1, std::nullopt, WaveSlot{0, 1, 0}}, body);
  auto all = profile.metrics({kLoop, {}, 3, std::nullopt}, body);

  EXPECT_EQ(row_at(slot, kLoopHeader).n, 5u);
  EXPECT_EQ(row_at(slot, kLoopBranch).n, 5u);
  // Per pass the MFMA at t (stall 1) covers to t+17, so kLoopHeader + 8
  // (t+20..t+24) has 4 bubble cycles: 5 passes in the slot.
  EXPECT_EQ(row_at(slot, kLoopHeader + 8).bubble, 20u);
  EXPECT_EQ(row_at(other, kLoopHeader).n, 2u);
  EXPECT_EQ(row_at(all, kLoopHeader).n, 7u);
}
```

- [ ] **Step 2: Run it to see it fail**

Run: `cmake --build build`
Expected: compile error: `WaveSlot` and the 5-member `LoopRecord` don't exist yet.

- [ ] **Step 3: Implement `aggregate_scope.h`, `LoopRecord::slot`, `slot_totals`, `WaveBuilder(simd, wave_id)`**

- [ ] **Step 4: Run the tests**

Run: `build/tests/my_rocperf_tool_tests --gtest_filter='LoopProfileTest.*:WaveAnalysis*'`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add include/my_rocperf_tool/aggregate_scope.h include/my_rocperf_tool/loop_profile.h lib/loop_profile.cpp tests/synthetic_wave.h tests/loop_profile_test.cpp
git commit -m "Keep a loop record per SIMD wave slot"
```

---

### Task 3: ReportConfig, `--report` and `--aggregate` for loop reports

Moves the loop report onto the shared config and the new flags. Static and dynamic values come in Task 7.

**Files:**
- Create: `include/my_rocperf_tool/report_config.h`, `include/my_rocperf_tool/report_document.h`, `lib/report_document.cpp`, `lib/report_writer.h` (private helpers)
- Modify: `include/my_rocperf_tool/loop_report.h`, `lib/loop_report.cpp`, `lib/CMakeLists.txt`, `tools/load_issue_stall.cpp`
- Test: `tests/loop_report_test.cpp`

**Interfaces:**
- Consumes: `AggregateScope`, `WaveSlot`, `LoopRecord::slot` (Task 2).
- Produces, in `report_config.h`:
  - `enum class ReportKind { LoopSummary, LoopDetail, StaticInst, DynamicInst, None };`
  - `enum class ReportFormat { Text, Csv, Json };` (renamed from `LoopOutputFormat`)
  - `struct ReportConfig` (renamed from `LoopOutputConfig`): `format`, `ReportKind report = ReportKind::LoopSummary`, `AggregateScope aggregate = AggregateScope::Wave`, `show_stats`, `show_addr`, `show_avg`, `show_mfma_coexec`, `color_mfma_bubble`, `show_prefix`. `show_contents`, `aggregate_wave`, `aggregate_wave_only` are gone: loop rows print iff `report == ReportKind::LoopDetail`.
  - `using ReportChunk = fmt::basic_memory_buffer<char, 256>;` (renamed from `LoopReportChunk`)
- Produces, in `report_document.h`: `void write_report_document(const ReportConfig& cfg, llvm::ArrayRef<ReportChunk> chunks, llvm::function_ref<void(std::string_view)> write);` (replaces `write_loop_report_document`). `ReportKind::None` writes nothing in any format. CSV header: `write_loop_csv_header` for loop reports (Task 6 adds the instruction header).
- Produces, in `loop_report.h`: `struct LoopGroup { AggregateScope scope = AggregateScope::Wave; WaveSlot slot; std::vector<ReportedLoop> loops; };` (`slot` unused under `All`); `void write_loop_csv_header(ReportChunk& out, const ReportConfig& cfg);`; `render_loop_report(ReportChunk&, const ReportConfig&, const TraceLoopReport&)`.
- Produces, in `lib/report_writer.h` (inline or with definitions in `lib/report_document.cpp`): `writef(ReportChunk&, ...)`, `csv_escape`, `json_escape`, `write_dispatch_label(ReportChunk& out, const ReportConfig& cfg, std::string_view prefix, uint64_t dispatch_id)`, `const char* bubble_color(uint64_t bubble, uint64_t max_bubble)`: moved from `loop_report.cpp`.

**Loop report output per scope** (`Wave` and `All` exactly as today's per-wave and aggregate):

| | `Wave` | `SimdSlot` | `All` |
|---|---|---|---|
| Text heading | `=== Wave (…, CU=c, SIMD=s, WaveID=w) ===` | `=== SIMD Slot (DispatchID=7, CU=1, SIMD=0, WaveID=2) ===` | `=== Aggregate Across Waves (…) ===` |
| Text loop line | per-wave form | aggregate form (`N waves, …`) | aggregate form |
| CSV `record_type`, `wave_count` columns | absent | `simd_slot`, record's wave count | `aggregate`, record's wave count |
| CSV `cu`,`simd`,`wave_id` | filled | filled | blank |
| JSON group keys | `cu`,`simd`,`wave_id` | `"simd_slot":true`,`cu`,`simd`,`wave_id`; each loop has `wave_count` | `"aggregate":true`; each loop has `wave_count` |

**CLI changes** (`tools/load_issue_stall.cpp`):
- Remove flags `detect_loops`, `dump_loop_contents`, `aggregate_wave`, `aggregate_wave_only`. Rename `loop_output_format`→`output_format`, `loop_inst_stats`→`inst_stats`, `loop_inst_addr`→`inst_addr`, `loop_inst_avg`→`inst_avg`.
- Add `ABSL_FLAG(std::string, report, "loop_summary", …)` and `ABSL_FLAG(std::string, aggregate, "wave", …)`; `std::optional<ReportKind> parse_report_kind(const std::string&)` (accepts `loop_summary`, `loop_detail`, `none` in this task) and `std::optional<AggregateScope> parse_aggregate_scope(const std::string&)` (`wave`, `simd_slot`, `all`). In `main`, a bad value prints `--report must be one of …; got '<v>'` (likewise `--aggregate`) and returns 2.
- `report_config_from_flags()` replaces `loop_output_config_from_flags()`. Options: `detect_loops = report is LoopSummary or LoopDetail`; `wave_loop_rows = report == LoopDetail && aggregate == Wave`.
- `build_loop_groups` by `cfg.aggregate`: `Wave` = today's per-wave groups; `SimdSlot` = one group per slot with loops, slots in `WaveSlot` order, each loop's record `{edge, Σ stats over the slot's waves with it, that count, nullopt, slot}`, loops sorted like the aggregate's; `All` = today's aggregate group (printed even when empty). Generalize `aggregate_loops_by_back_edge` with a `std::optional<WaveSlot>` filter instead of copying it.
- The `wave_loops` sort breaks ties by `wave_index` (decode order).
- The first `[summary]` line keeps its loop clause under loop reports; under other reports it ends after `instructions decoded`. The loop-free lines print only under loop reports.
- `write_report_document` replaces `write_loop_report_document`.

- [ ] **Step 1: Update `tests/loop_report_test.cpp` to the new API and write the new tests**

Rename types; `option_matrix` sets `cfg.report = bits & 4 ? ReportKind::LoopDetail : ReportKind::LoopSummary` and loops over the three scopes instead of the `aggregate_wave` bit. `trace_report(body, AggregateScope scope, dispatch_id)` builds: `Wave` → groups `{Wave, {1,0,2}, {loop_record(body,4,1,0)}}`, `{Wave, {3,1,0}, {loop_record(body,4,1,1)}}`; `SimdSlot` → `{SimdSlot, {1,0,2}, {loop_record(body,8,2,nullopt)}}`, `{SimdSlot, {3,1,0}, {loop_record(body,4,1,nullopt)}}` (records' `slot` set to the group's); `All` → `{All, {}, {loop_record(body,8,2,nullopt)}}`. Expected entry counts follow from that (2, 2, 1 groups). `EmptyAggregateHeadingSuppressesTheBanner` uses `aggregate = All`. New tests:

```cpp
TEST(LoopReportTest, CsvSlotAndAggregateRowsNameTheirRecordType) {
  // SimdSlot: record_type=simd_slot, cu=1, simd=0, wave_id=2, wave_count=2 on row 1.
  // All: record_type=aggregate, cu="" on row 1.
  // Wave: header has no record_type and no wave_count column.
}
TEST(LoopReportTest, JsonSlotGroupsAreMarked) {
  // SimdSlot entry 0: getBoolean("simd_slot") == true, getInteger("cu") == 1,
  // no "aggregate" key, loops[0].getInteger("wave_count") == 2.
}
TEST(LoopReportTest, TextSlotHeadingAndLoopLine) {
  // One SimdSlot group {1,0,2} with loop_record(body, 4, 2, nullopt), LoopSummary:
  EXPECT_EQ(text, R"(=== SIMD Slot (DispatchID=7, CU=1, SIMD=0, WaveID=2) ===
  Loop [0x100 -> 0x110] code_object=1: 2 waves, 4 iterations, 130 total cycles (avg 32/iter), stall 26 (avg 6/iter), idle 10
    avg/iter: dur=32.50 stall=6.50 idle=2.50 mfma_time=16 (1 mfma * 16) mfma_util=49.2%
)");
}
TEST(LoopReportTest, ReportNoneWritesNothing) {
  // For each ReportFormat, cfg.report = None: document(cfg, {chunk}) == "".
}
```
Write each test's body from its comment; the assertions named there are the test.

- [ ] **Step 2: Run them to see them fail**

Run: `cmake --build build`
Expected: compile errors (`ReportConfig`, `AggregateScope` in `LoopGroup`, `write_report_document` missing).

- [ ] **Step 3: Implement the library side (`report_config.h`, `report_document`, `report_writer.h`, `loop_report`)**

- [ ] **Step 4: Run the loop report tests**

Run: `build/tests/my_rocperf_tool_tests --gtest_filter='LoopReportTest.*'`
Expected: all pass, including the unchanged `TextMatchesGolden`.

- [ ] **Step 5: Implement the CLI changes**

- [ ] **Step 6: Diff against the baseline**

```bash
N=build/tools/load_issue_stall; O=/tmp/wwir-baseline
for t in 1 2; do
  T=$([ $t = 1 ] && echo /raid/jinpli/workspace/home/jinpli/att || echo /raid/jinpli/workspace/home01/jinpli/development/bench_upstream_20261106/pro_a4w4_tokens/att)
  for f in text csv json; do
    $N --att_output_dir=$T --output_format=$f --report=loop_summary >/tmp/new.plain.$f 2>/tmp/new.plain.$f.err
    $N --att_output_dir=$T --output_format=$f --report=loop_detail --mfma_coexec >/tmp/new.detail.$f 2>/tmp/new.detail.$f.err
    $N --att_output_dir=$T --output_format=$f --aggregate=all --report=loop_detail --inst_avg >/tmp/new.all.$f 2>/tmp/new.all.$f.err
    for k in plain detail all; do cmp $O/t$t.$k.$f /tmp/new.$k.$f && cmp $O/t$t.$k.$f.err /tmp/new.$k.$f.err; done
  done
done
T=/raid/jinpli/workspace/home/jinpli/att
$N --att_output_dir=$T --report=bogus; echo "exit=$?"
$N --att_output_dir=$T --aggregate=bogus; echo "exit=$?"
$N --att_output_dir=$T --output_format=csv --aggregate=simd_slot | head -3
```
Expected: every `cmp` silent; both bogus runs print the flag error and `exit=2`; the slot CSV shows `record_type` and `simd_slot` rows. A diff that only reorders same-label `=== Wave` groups is the tie-break; anything else is a regression to fix.

- [ ] **Step 7: Commit**

```bash
git add -A include lib tools tests
git commit -m "Select loop reports with --report and group them with --aggregate"
```

---

### Task 4: InstProfile

**Files:**
- Create: `include/my_rocperf_tool/inst_profile.h`, `lib/inst_profile.cpp`, `tests/inst_profile_test.cpp`
- Modify: `lib/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `MfmaShadow`, `issue_cycles`, `MfmaShadowOverlap` (Task 1); `AggregateScope`, `WaveSlot` (Task 2); `compute_idle`, `is_null_pc` (`loop_detection.h`); `DenseMapInfo` for PCs (`trace_pc.h`).
- Produces, in `inst_profile.h`:

```cpp
enum class InstProfileKind { Static, Dynamic };

// One key's sums over a group's executions of it.
struct InstSums {
  uint64_t n = 0;           // executions
  uint64_t exec_waves = 0;  // waves with at least one execution
  uint64_t dur = 0, stall = 0, idle = 0, issue = 0, bubble = 0, rel_issue = 0;
};

struct InstRow {
  rocprofiler_thread_trace_decoder_pc_t pc;
  uint32_t occurrence;    // 1-based in a dynamic profile, 0 in a static one
  InstKind kind;
  std::string_view text;  // empty from the profile; the CLI fills it in
  InstSums sums;
};

// Per-wave averages over a group's waves; all 0 for a group of no records.
struct InstSummary {
  uint64_t waves = 0;
  double avg_records = 0, avg_dur = 0, avg_stall = 0, avg_idle = 0,
         avg_bubble = 0;
  double avg_mfma = 0;   // MFMA executions per wave
  double mfma_util = 0;  // avg_mfma * kMfmaShadowCycles / avg_dur, 0 if avg_dur is 0
};

struct InstGroup {
  AggregateScope scope;
  WaveSlot slot;          // unused under All
  uint64_t wave_index;    // the wave, under Wave only
  InstSummary summary;
  std::vector<InstRow> rows;
};

class InstProfile {
 public:
  using pc_t = rocprofiler_thread_trace_decoder_pc_t;
  using wave_t = rocprofiler_thread_trace_decoder_wave_t;
  using KindLookup = llvm::function_ref<InstKind(const pc_t&)>;

  InstProfile(InstProfileKind kind, AggregateScope scope, bool mfma_coexec);
  // `wave` has its s_barrier token pairs coalesced. kind_at is asked for
  // every attributed record (WaveAnalysis caches it per PC). Returns the
  // wave's MFMA shadow overlaps, under mfma_coexec.
  std::vector<MfmaShadowOverlap> add_wave(const wave_t& wave,
                                          uint64_t wave_index,
                                          KindLookup kind_at);
  // Consumes the profile. In report order: Wave groups by (slot, wave_index);
  // SimdSlot groups by slot; one All group, or none if no wave was added.
  std::vector<InstGroup> groups() &&;
};
```

Rules (from the spec): walk every attributed record in order; `idle` via `compute_idle` from `wave.begin_time`; `rel = time + stall - (time + stall of the wave's first attributed record)`; one `MfmaShadow` per wave, bubbles only under `mfma_coexec`; occurrence = per-wave count of the PC so far. Sum a wave into a per-wave map first, then merge it into its group, adding 1 to `exec_waves` per key. Static rows sort by `(code_object_id, vaddr)`; dynamic rows by `rel_issue / n`, ties by `(code_object_id, vaddr, occurrence)`.

- [ ] **Step 1: Write the failing tests**

All use `synthetic_wave.h`: `inst` records start every 10 cycles from 0, last 4, stall 1, so idle is 6 after the first and `rel` is `10 * k`.

```cpp
const InstRow& row_at(const InstGroup& g, uint64_t addr, uint32_t occurrence = 0);  // ADD_FAILURE if missing

TEST(InstProfileTest, StaticRowsSumEachPcOverWaves) {
  // Two waves, each inst(0xf0,kSalu).inst(0x100,kValu).inst(0x104,kValu).inst(0x110,kSalu); Static, All.
  ASSERT_EQ(groups.size(), 1u);
  // rows in address order 0xf0, 0x100, 0x104, 0x110; each n=2, exec_waves=2, dur=8, stall=2, issue=6
  EXPECT_EQ(row_at(g, 0xf0).sums.idle, 0u);    EXPECT_EQ(row_at(g, 0x100).sums.idle, 12u);
  EXPECT_EQ(row_at(g, 0xf0).sums.rel_issue, 0u); EXPECT_EQ(row_at(g, 0x100).sums.rel_issue, 20u);
  EXPECT_EQ(row_at(g, 0x104).sums.rel_issue, 40u); EXPECT_EQ(row_at(g, 0x110).sums.rel_issue, 60u);
  EXPECT_EQ(g.summary.waves, 2u); EXPECT_DOUBLE_EQ(g.summary.avg_records, 4);
  EXPECT_DOUBLE_EQ(g.summary.avg_dur, 16); EXPECT_DOUBLE_EQ(g.summary.avg_stall, 4);
  EXPECT_DOUBLE_EQ(g.summary.avg_idle, 18);
}
TEST(InstProfileTest, RelIsAnchoredAtTheFirstRecordAndIdleAtBeginTime) {
  // wait(100).inst(0xf0).inst(0x100): row 0xf0 idle=100 rel_issue=0; row 0x100 idle=6 rel_issue=10.
}
TEST(InstProfileTest, MfmaShadowRunsAcrossTheWholeWave) {
  // mfma_coexec; 0xf0 is Mfma. at(0xf0,kValu,0,4,0), at(0x100,kValu,10,4,0), at(0x104,kValu,20,4,0):
  // bubble 0xf0=0, 0x100=0 (ends at 14, shadow to 16), 0x104=4; summary.avg_bubble=4.
}
TEST(InstProfileTest, WithoutMfmaCoexecNoBubblesButMfmasCount) {
  // Same wave, no mfma_coexec: every bubble 0; avg_mfma=1; mfma_util=16.0/12.0.
}
TEST(InstProfileTest, SkipsNullPcRecords) {
  // null_pc().inst(0xf0).null_pc().inst(0x100): 2 rows; avg_records=2; row 0xf0 rel_issue=0, stall=1.
}
TEST(InstProfileTest, IssueIsZeroWhereStallExceedsDuration) {
  // at(0xf0,kValu,0,/*dur*/2,/*stall*/5), at(0x100,kValu,10,4,1): row 0xf0 issue=0; row 0x100 rel_issue=6.
}
TEST(InstProfileTest, DynamicRowsKeyByPcAndOccurrence) {
  // Wave A: 0x100,0x104,0x100,0x108. Wave B: 0x100,0x108. Dynamic, All.
  // Row (addr, occurrence, n) sequence: (0x100,1,2), (0x104,1,1), (0x100,2,1), (0x108,1,2);
  // (0x100,2) and (0x108,1) tie at avg 20, broken by address. exec_waves == n in every row.
}
TEST(InstProfileTest, StaticRowsCountEveryExecution) {
  // Same two waves, Static: 0x100 n=3 exec_waves=2; 0x104 n=1 exec_waves=1; 0x108 n=2 exec_waves=2; occurrence 0.
}
TEST(InstProfileTest, GroupsFollowTheScope) {
  // Waves 0..3 on slots (simd,wave_id) (0,1), (1,0), (0,1), (0,0), each inst(0xf0).
  // Wave: wave_index sequence {3, 0, 2, 1}. SimdSlot: slots (0,0),(0,1),(1,0), waves {1,2,1},
  // row 0xf0 n=2 in (0,1). All: one group, waves=4, row 0xf0 n=4.
}
TEST(InstProfileTest, NoWaveNoGroup) {
  // For both kinds and all three scopes: groups() is empty.
}
TEST(InstProfileTest, WaveWithoutAttributedRecordsHasAnEmptyGroup) {
  // One wave of null_pc() only, Wave scope: 1 group, no rows, waves=1, every average 0.
}
TEST(InstProfileTest, ReportsMfmaShadowOverlaps) {
  // mfma_coexec; 0xf0, 0x100 Mfma; at(0xf0,kValu,0,4,0), at(0x100,kValu,8,4,0):
  // add_wave returns one overlap {pc 0x100, issue_point 8, shadow_end 16}.
}
```
Write each body from its comment; the values named there are the assertions.

- [ ] **Step 2: Run them to see them fail**

Run: `cmake --build build`
Expected: compile error, `inst_profile.h` not found.

- [ ] **Step 3: Implement `InstProfile` in `lib/inst_profile.cpp`**

- [ ] **Step 4: Run the tests**

Run: `build/tests/my_rocperf_tool_tests --gtest_filter='InstProfileTest.*'`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add include/my_rocperf_tool/inst_profile.h lib/inst_profile.cpp lib/CMakeLists.txt tests/inst_profile_test.cpp tests/CMakeLists.txt
git commit -m "Add InstProfile: whole-wave per-instruction sums"
```

---

### Task 5: Route waves to the InstProfile

**Files:**
- Modify: `include/my_rocperf_tool/wave_analysis.h`, `lib/wave_analysis.cpp`
- Test: `tests/wave_analysis_test.cpp`

**Interfaces:**
- Consumes: `InstProfile`, `InstProfileKind` (Task 4).
- Produces: `WaveAnalysisOptions` gains `std::optional<InstProfileKind> inst_profile;` (nullopt: no instruction profile) and `AggregateScope aggregate = AggregateScope::Wave;`. `WaveAnalysisResult` gains `std::optional<InstProfile> inst_profile;`, set iff `options.inst_profile`. `add_wave` feeds the coalesced wave, after the duration check, to it with the wave's `wave_index` and the cached `kind_at`; its overlaps go to `diagnostics`.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(WaveAnalysisTest, InstProfileSeesTheCoalescedWholeWave) {
  // inst_profile=Static, aggregate=All, detect_loops=false; 0xf0 is ClassicBarrier.
  // Wave: MESSAGE+IMMED token pair at 0xf0 (built as in
  // CoalescesBarrierTokenPairIntoOneRecord), then iteration(true), iteration(false).
  // std::move(*result.inst_profile).groups()[0]: row 0xf0 n=1; row kLoopHeader n=2; result.wave_loops empty.
}
TEST(WaveAnalysisTest, InstProfileOverlapsBecomeDiagnostics) {
  // inst_profile=Dynamic, mfma_coexec; the overlap wave of ReportsMfmaShadowOverlaps:
  // result.diagnostics holds exactly one MfmaShadowOverlap.
}
TEST(WaveAnalysisTest, SimdFilterDropsWavesBeforeTheInstProfile) {
  // simd=1, aggregate=SimdSlot; waves on (simd 0, wave 0) and (simd 1, wave 2):
  // one group, slot.simd=1, slot.wave_id=2, summary.waves=1.
}
TEST(WaveAnalysisTest, NoInstProfileUnlessAsked) {
  // Default options: !result.inst_profile.has_value().
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `cmake --build build`
Expected: compile error, no `inst_profile` member.

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run the tests**

Run: `build/tests/my_rocperf_tool_tests --gtest_filter='WaveAnalysis*'`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add include/my_rocperf_tool/wave_analysis.h lib/wave_analysis.cpp tests/wave_analysis_test.cpp
git commit -m "Feed analyzed waves to an InstProfile when asked"
```

---

### Task 6: InstReport rendering

**Files:**
- Create: `include/my_rocperf_tool/inst_report.h`, `lib/inst_report.cpp`, `tests/inst_report_test.cpp`
- Modify: `lib/report_document.cpp` (CSV header for instruction reports), `lib/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `ReportConfig`, `ReportChunk`, `write_report_document`, `lib/report_writer.h` (Task 3); `InstGroup`, `InstRow`, `InstSummary` (Task 4); `WaveCounts` (`wave_analysis.h`).
- Produces, in `inst_report.h`:

```cpp
struct TraceInstReport {
  std::string prefix;
  uint64_t dispatch_id = 0;
  uint64_t se_id = 0;
  WaveCounts counts;
  int32_t simd_filter = -1;
  std::vector<InstGroup> groups;  // in print order; rows' text must outlive the report
};
// cfg.report is StaticInst or DynamicInst; DynamicInst adds the occurrence column/key.
void render_inst_report(ReportChunk& out, const ReportConfig& cfg, const TraceInstReport& report);
void write_inst_csv_header(ReportChunk& out, const ReportConfig& cfg);
```

**Formats.** Stats follow the loop rows' conventions: sums by default, averages per execution under `show_avg`, bubble columns only under `show_mfma_coexec`, `show_stats=false` drops stats in text and JSON (CSV keeps them), `color_mfma_bubble` colours text rows by bubble relative to the group's largest.

| | Text | CSV | JSON |
|---|---|---|---|
| Group label | heading, below | `[record_type,][prefix,]dispatch_id,se_id,cu,simd,wave_id[,wave_count]` (`record_type`/`wave_count` iff scope ≠ Wave; `cu,simd,wave_id` blank under All) | `[prefix,]dispatch_id,se_id`, then Wave: `cu,simd,wave_id`; SimdSlot: `"simd_slot":true,cu,simd,wave_id,wave_count`; All: `"aggregate":true,wave_count` |
| Summary | `  avg/wave: records= dur= stall= idle=[ bubble=] mfma= mfma_util=N.N%` (`{:.2f}`, util `{:.1f}%`) | none | `"summary":{"waves","avg_records","avg_dur","avg_stall","avg_idle"[,"avg_bubble"],"avg_mfma","mfma_util"}` |
| Row identity | `[occ]` column, `addr`, text | `code_object,inst_addr[,occurrence],instruction` | `"code_object","addr"[,"occurrence"],"inst"` |
| Row stats | `n waves` then loop-row stats | `n,exec_waves,` then `dur,stall,idle,issue[,bubble],rel_issue` or `avg_dur,avg_stall,avg_idle,avg_issue[,avg_bubble],avg_cycle` | same names as CSV |

Text headings (prefix via `write_dispatch_label`):
- Wave: `=== Wave (DispatchID=7, SE=2, CU=1, SIMD=0, WaveID=2) ===`
- SimdSlot: `=== SIMD Slot (DispatchID=7, SE=2, CU=1, SIMD=0, WaveID=2, 3 waves) ===`
- All: `=== Aggregate Across Waves (DispatchID=7, SE=2, 12 waves) ===`
- No group (text only): `=== DispatchID=7, SE=2: no waves analyzed ===`, or with a SIMD filter `=== DispatchID=7, SE=2: no waves analyzed (3 waves on other SIMDs skipped by --simd=1) ===`

- [ ] **Step 1: Write the failing tests**

Fixture rows (static: occurrence 0; dynamic: occurrences 1 and 2):
row A `{pc {0x100,1}, Mfma, "v_mfma_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7], v[0:3]", n=2, exec_waves=2, dur=8, stall=2, idle=0, issue=6, bubble=0, rel_issue=0}`,
row B `{pc {0x104,1}, Other, "s_add_u32 s0, s0, 1", n=2, exec_waves=1, dur=10, stall=4, idle=12, issue=6, bubble=3, rel_issue=25}`;
summary `{waves=1, avg_records=4, avg_dur=16, avg_stall=4, avg_idle=18, avg_bubble=3, avg_mfma=1, mfma_util=1.0}`.
`trace_inst_report(scope)`: dispatch 7, SE 2, prefix `R"(run "1",a)"`; Wave → groups on slots {1,0,2} and {3,1,0}; SimdSlot → the same two slots with `summary.waves` 3; All → one group with `summary.waves` 12. Every group has rows A and B.

```cpp
TEST(InstReportTest, TextStaticMatchesGolden) {
  // Wave scope, one group (slot {1,0,2}), default cfg with report=StaticInst:
  EXPECT_EQ(text, R"(=== Wave (DispatchID=7, SE=2, CU=1, SIMD=0, WaveID=2) ===
  avg/wave: records=4.00 dur=16.00 stall=4.00 idle=18.00 mfma=1.00 mfma_util=100.0%
    n      waves  dur        stall      idle       issue    avg_cycle  addr           instruction
    2      2      8          2          0          6        0.00       0x100:  v_mfma_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7], v[0:3]
    2      1      10         4          12         6        12.50      0x104:  s_add_u32 s0, s0, 1
)");
}
TEST(InstReportTest, TextDynamicAddsTheOccurrenceColumn) {
  // Same, report=DynamicInst, occurrences 1 and 2: the last three lines are
  //    n      waves  dur        stall      idle       issue    avg_cycle  occ   addr           instruction
  //    2      2      8          2          0          6        0.00       1     0x100:  v_mfma_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7], v[0:3]
  //    2      1      10         4          12         6        12.50      2     0x104:  s_add_u32 s0, s0, 1
}
TEST(InstReportTest, TextHeadingsPerScope) {
  // First lines of the SimdSlot and All reports equal the headings in the table above;
  // with show_prefix, "Prefix=run "1",a, " precedes "DispatchID".
}
TEST(InstReportTest, TextBannersATraceWithoutGroups) {
  // Both banner strings above; CSV and JSON render nothing for it.
}
TEST(InstReportTest, CsvRowsHaveTheHeadersColumns) {
  // For every config of the matrix (kind x scope x show_avg x show_mfma_coexec x show_prefix):
  // 1 + 2 * groups lines; each row splits (split_csv) into header.size() fields;
  // "occurrence" column iff DynamicInst; "record_type" iff scope != Wave, value
  // "simd_slot"/"aggregate"; se_id == "2"; instruction == the row's text;
  // with show_prefix, prefix == R"(run "1",a)".
}
TEST(InstReportTest, JsonDocumentParses) {
  // For every config of the matrix: two traces' chunks and an empty one between parse as one
  // array of 2 * groups entries; each has "summary" (with "waves") and "instructions" of size 2;
  // "occurrence" key iff DynamicInst; "aggregate" iff All; "simd_slot" iff SimdSlot.
}
```
Copy `lines`, `split_csv`, `column` from `loop_report_test.cpp` (or move them into a shared `tests/report_test_util.h`, which the loop report test then includes too).

- [ ] **Step 2: Run them to see them fail**

Run: `cmake --build build`
Expected: compile error, `inst_report.h` not found.

- [ ] **Step 3: Implement `inst_report.cpp` and the CSV header branch in `write_report_document`**

Build the CSV header and rows from one column list, as `csv_columns` does for loops.

- [ ] **Step 4: Run the tests**

Run: `build/tests/my_rocperf_tool_tests --gtest_filter='InstReportTest.*:LoopReportTest.*'`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add include/my_rocperf_tool/inst_report.h lib/inst_report.cpp lib/report_document.cpp lib/CMakeLists.txt tests
git commit -m "Render static and dynamic instruction reports"
```

---

### Task 7: CLI: `--report=static_inst|dynamic_inst`

**Files:**
- Modify: `tools/load_issue_stall.cpp`

**Interfaces:**
- Consumes: `InstProfileKind`, `WaveAnalysisOptions::inst_profile/aggregate`, `WaveAnalysisResult::inst_profile` (Tasks 4–5); `TraceInstReport`, `render_inst_report` (Task 6).
- Produces:
  - `parse_report_kind` also accepts `static_inst`, `dynamic_inst`.
  - `main` sets `options.inst_profile` (Static/Dynamic) and `options.aggregate` from the flags.
  - `class InstTexts { explicit InstTexts(const Disassembler&); std::string_view of(const rocprofiler_thread_trace_decoder_pc_t& pc); };`: each PC's `text_at` once per trace, in a node-based map (`std::map<std::pair<uint64_t, uint64_t>, std::string>`) so the views stay valid.
  - In `run_trace`, under an instruction report: `groups = std::move(*result.inst_profile).groups()` consumes the profile, each row's `text = texts.of(row.pc)`, then a `TraceInstReport` (prefix, dispatch id, SE, counts, SIMD filter) through `render_inst_report`.
  - `--report` help text names the five values, says `dynamic_inst` memory grows with the longest wave's records times the groups, and that waves caught mid-run (lossy captures) are included, which skews their `rel` and occurrences.

- [ ] **Step 1: Implement**

- [ ] **Step 2: Check the reports against the run's own totals**

```bash
T1=/raid/jinpli/workspace/home/jinpli/att
T2=/raid/jinpli/workspace/home01/jinpli/development/bench_upstream_20261106/pro_a4w4_tokens/att
N=build/tools/load_issue_stall
sum_n() { python3 -I -c 'import csv,sys; r=csv.DictReader(sys.stdin); print(sum(int(x["n"]) for x in r))'; }
for k in static_inst dynamic_inst; do
  for a in all simd_slot wave; do echo "T1 $k $a: $($N --att_output_dir=$T1 --report=$k --aggregate=$a --output_format=csv 2>/dev/null | sum_n)"; done
  for a in all simd_slot; do echo "T2 $k $a: $($N --att_output_dir=$T2 --report=$k --aggregate=$a --output_format=csv 2>/dev/null | sum_n)"; done
done
$N --att_output_dir=$T1 --report=none | wc -c
$N --att_output_dir=$T1 --report=dynamic_inst --aggregate=all --mfma_coexec --inst_avg | head -8
# A loop-free (unrolled) dispatch: the case this feature exists for.
$N --att_output_dir=$T2 --report=static_inst --aggregate=all --mfma_coexec --inst_avg | grep -A4 'Prefix=out_fly/out, DispatchID=63284' | head -12
```
Expected: every T1 sum prints 4609976 and every T2 sum 51203324 (each attributed record counted exactly once per scope); `--report=none` prints 0 bytes to stdout; the dynamic text shows an `=== Aggregate Across Waves (…, SE=…, N waves) ===` heading and an `occ` column; dispatch 63284 shows `Aggregate Across Waves` headings with `Prefix=out_fly/out`, an `avg/wave:` line with a bubble value, and instruction rows.

- [ ] **Step 3: Check `--jobs` independence and JSON validity**

```bash
for j in 1 0; do $N --att_output_dir=$T2 --report=static_inst --aggregate=simd_slot --output_format=json --jobs=$j > /tmp/j$j.json 2>/tmp/j$j.err; done
cmp /tmp/j1.json /tmp/j0.json && cmp /tmp/j1.err /tmp/j0.err && python3 -I -m json.tool /tmp/j0.json > /dev/null && echo OK
```
Expected: `OK`.

- [ ] **Step 4: Re-run the Task 3 baseline diff and the full test suite**

Run: the Task 3 Step 6 loop, then `ctest --test-dir build --output-on-failure`.
Expected: no diff; all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/load_issue_stall.cpp
git commit -m "Add --report=static_inst and dynamic_inst to load_issue_stall"
```

---

### Task 8: Documentation

**Files:**
- Modify: `CLAUDE.md`, `CONTEXT.md`

- [ ] **Step 1: Update `CLAUDE.md`**

Architecture: add `mfma_shadow.h`, `aggregate_scope.h`, `report_config.h` / `report_document`, **InstProfile** and **InstReport** entries (with their test files); update **LoopProfile** (slot records, constants moved), **LoopReport** (`ReportConfig`, three scopes), **WaveAnalysis** (`inst_profile`, `aggregate`) and the CLI paragraph (`--report`, `--aggregate`, renamed flags, `none` writes no stdout). Replace every old flag name.

- [ ] **Step 2: Update `CONTEXT.md`**

Add terms: **Static instruction**, **Dynamic instruction** / **Occurrence**, **Slot** and **Aggregate scope** (wave / simd_slot / all), **Relative issue** in whole-wave reports (anchored at the wave's first attributed record). Update **Loop record** (per-slot scope) and **Duration check** / other entries naming old flags.

- [ ] **Step 3: Check no old flag name survives**

Run: `grep -rnE 'dump_loop_contents|detect_loops|aggregate_wave|loop_output_format|loop_inst_(stats|addr|avg)|LoopOutputConfig|LoopReportChunk' --include='*.h' --include='*.cpp' --include='*.md' . | grep -v '^./3rd_party' | grep -v '^./docs/superpowers'`
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md CONTEXT.md
git commit -m "Document whole-wave instruction reports and the new flags"
```
