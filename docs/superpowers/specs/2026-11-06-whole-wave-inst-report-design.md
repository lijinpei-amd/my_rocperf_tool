# Whole-wave instruction reports (static and dynamic)

Status: draft for review. Date: 2026-11-06.

## Why

A kernel whose main loop the compiler fully unrolled has no back-edge, so
`load_issue_stall` finds no loop and prints only
`no loops detected (N waves, M instructions)`. Before the unroll,
`--dump_loop_contents --mfma_coexec` showed, per body instruction, where the
cycles went (dur, stall, idle, issue) and where the MFMA pipe sat idle
(bubble). The user wants that same view back for straight-line code: the
whole wave, every instruction it executed.

Success: on an unrolled kernel's capture, the user can read per-instruction
stall and bubble numbers, averaged over waves, in text, CSV and JSON, both
across all waves of a trace and per (SIMD, wave slot).

## Goals

1. A new option, `--report`, selects exactly one report per run:
   - **loop summary** and **loop detail**: today's loop report without and
     with `--dump_loop_contents`, output unchanged; loop summary is the
     default;
   - **static instruction report**: one row per instruction (PC) the waves
     executed;
   - **dynamic instruction report**: one row per dynamic instruction,
     `(PC, occurrence)`, averaged over the waves that executed it.
2. Both new reports cover the whole wave, from its first record to its last,
   not a guessed region. They work on any kernel, loops or not.
3. A new option, `--aggregate`, selects one scope (how records are grouped)
   for every report, loop reports included:
   - **`all`**: every analyzed wave of the trace, aggregated together;
   - **`simd_slot`**: the waves that ran in one `(simd, wave_id)` slot, one
     after another, aggregated together;
   - **`wave`**: each wave on its own.
4. All three formats: text, CSV, JSON.

## Non-goals

- No per-wave generation number (the n-th wave a slot ran). The user
  decided it is not needed; a slot group is named by `(simd, wave_id)` alone.
- No `--anchor_pc` or other user-chosen origin for relative issue time.
- No change to what the loop report prints. Only the flags that select and
  shape it change (see Options).
- No cross-trace (cross-SE) aggregate: each trace reports on its own, as the
  loop report does today.

## Anti-goals (would make it a failure even if it runs)

- A static or dynamic row that mixes different instructions together.
- Changing what the loop report prints for the old flag combinations that
  still have an equivalent, given the same other flags (renamed):

  | Old | New, same output |
  |---|---|
  | (plain run) | `--report=loop_summary --aggregate=wave` |
  | `--dump_loop_contents` | `--report=loop_detail --aggregate=wave` |
  | `--aggregate_wave_only` | `--aggregate=all` |

  The old `--aggregate_wave` (per-wave groups followed by the aggregate in
  one run) has no equivalent: `--aggregate` takes one value.
- Output that depends on `--jobs` (ordering must stay deterministic).

## Definitions (decided with the user)

All existing record-level definitions in `CONTEXT.md` still hold: duration,
stall, issue (`max(duration - stall, 0)`), idle, bubble, MFMA shadow,
coalesced s_barrier token pair. Records with a null PC are skipped, as
everywhere else.

- **Scope of a wave's walk**: every attributed record of the wave, in time
  order. The MFMA shadow (under `--mfma_coexec`) starts at 0 at the wave's
  start and runs across the whole wave; it is not reset anywhere. A bubble is
  a non-MFMA record's cycles past the current shadow end, exactly as in
  `LoopProfile::add_passes`.
- **Idle**: as today: from the end of the wave's previous attributed record,
  or from `wave.begin_time` for the first record.
- **Relative issue (`rel`)**: a record's issue point `time + stall` minus the
  issue point of the **wave's first attributed record**. The first record's
  `rel` is 0. Not `wave.begin_time`: the gap from launch to the first
  instruction is launch latency, and the first record's `idle` already shows
  it.
  - Static row: `rel_issue = Σ rel` over the row's executions,
    `avg_cycle = rel_issue / n`.
  - Dynamic row: same sums over the waves that executed it.
- **Occurrence**: the 1-based count of how many times a wave has executed a
  given PC so far. The k-th execution of PC p in a wave is dynamic instruction
  `(p, k)`. Keyed by (PC, occurrence), not by position in the wave, so that
  rows stay aligned after waves take different branches. In fully unrolled
  code every occurrence is 1, and the dynamic report holds the same numbers
  as the static one, in a different order.
- **Slot**: `(simd, wave_id)` of the decoded wave. `cu` is always the trace's
  target CU (WGP on gfx10+), so within a trace `(simd, wave_id)` names the
  slot; groups still print `cu` like the loop report does.

## Report contents

### Static instruction report

One row per executed PC, per group. Rows in `(code_object_id, vaddr)` order.
Only executed PCs get rows (listing all of `.text` would print other kernels
in the same code object).

Row fields: `code_object`, `addr`, `instruction` text, `n` (executions),
`wave_count` (waves that executed it at least once), and the sums `dur`,
`stall`, `idle`, `issue`, `bubble` (under `--mfma_coexec` only), `rel_issue`,
or under the averages option their per-execution averages `avg_dur`,
`avg_stall`, `avg_idle`, `avg_issue`, `avg_bubble`, `avg_cycle`.

### Dynamic instruction report

One row per `(PC, occurrence)`, per group. Rows ordered by `avg_cycle`
(average `rel`), ties broken by `(code_object_id, vaddr, occurrence)`, so the
rows read as the average wave's timeline.

Row fields: the static row's, plus `occurrence`. `n` here is the number of
waves in the group that executed that occurrence.

### Groups (`--aggregate`)

The same three scopes apply to every report. A group's rows count the
executions of its waves only.

| `--aggregate=` | Groups per trace | Group order | Group label |
|---|---|---|---|
| `all` | one, every analyzed wave | — | the aggregate, with its wave count |
| `simd_slot` | one per `(simd, wave_id)` slot that analyzed a wave | `(cu, simd, wave_id)` | cu, simd, wave_id, wave count |
| `wave` (default) | one per analyzed wave | `(cu, simd, wave_id)`, then decode order | cu, simd, wave_id |

- **Loop reports:** `wave` and `all` are today's per-wave records and
  aggregate record. `simd_slot` is new for loops: one loop record per (slot,
  loop), summing the waves of that slot in which `detect_loops` found the
  loop, with `wave_count` counting them; like the aggregate, a slot wave that
  ran the body without the loop being detected adds nothing. As today, the
  loop report lists only groups that have a loop (except the `all` group,
  which prints its heading as today).
- **Static and dynamic reports under `wave`:** one wave's rows. In a dynamic
  row `n` is then 1, and its numbers are that one execution's.
- **Memory:** per-wave static and dynamic groups keep a row set per wave until
  the trace renders, so memory grows with waves × executed PCs (static) or
  waves × records (dynamic). Acceptable: the user selects it knowingly.

*Author's call:* default `wave`, so a run without `--aggregate` prints what
it printed before. For the new reports the user will usually pass
`--aggregate=all` or `simd_slot`.

### Shared summary line per group

`waves`, total records, and averages per wave of dur, stall, idle and, under
`--mfma_coexec`, bubble; MFMA count per wave and MFMA utilization
(`mfma_count * kMfmaShadowCycles / avg wave duration`), mirroring the loop
summary. Builder may trim this if it does not fit naturally; the rows are the
point.

*Author's call:* "wave duration" here is the sum of the wave's record
durations, as the loop summary's `avg_dur` sums record durations per pass
(`lib/loop_detection.cpp`), not the wave's wall-clock span.

### Edge cases (author's calls)

- **No waves analyzed:** under the static and dynamic reports, a trace that
  analyzed no wave gets no group, even under `--aggregate=all`, and text
  prints the banner. Unlike the loop
  report, which prints an empty aggregate heading in that case.
- **Partial waves:** the decoder emits waves still running when the trace
  stopped, and rebuilds waves whose start packet it missed. Their first
  attributed record may not be the kernel's first instruction, which skews
  `rel` and occurrence alignment. They are included like any other wave (only
  lossy captures have them); the flag's help text mentions it.

## Options (CLI)

The user is the tool's only user, so flags are renamed and removed outright:
no deprecated aliases.

- **New: `--report=loop_summary|loop_detail|static_inst|dynamic_inst|none`**,
  default `loop_summary`. An unknown value is a flag error (exit 2).

  | Value | Prints | Replaces |
  |---|---|---|
  | `loop_summary` | each loop record's summary, no instruction rows | a plain run |
  | `loop_detail` | each loop record's summary and its body's instruction rows | `--dump_loop_contents` |
  | `static_inst` | the static instruction report | new |
  | `dynamic_inst` | the dynamic instruction report | new |
  | `none` | no report; traces still decode and the stderr `[summary]` prints | `--detect_loops=false` |

- **Removed: `--detect_loops`.** `--report` decides whether loops are
  detected: only `loop_summary` and `loop_detail` detect them.
- **Removed: `--dump_loop_contents`.** Replaced by `--report=loop_detail`.
  Wherever the code read it (e.g. keeping per-wave loop rows,
  `WaveAnalysisOptions::wave_loop_rows`), read `--report=loop_detail`
  instead.
- **Renamed** (same meaning, now applying to every report):

  | Old | New |
  |---|---|
  | `--loop_output_format` | `--output_format` |
  | `--loop_inst_stats` | `--inst_stats` |
  | `--loop_inst_addr` | `--inst_addr` |
  | `--loop_inst_avg` | `--inst_avg` |

- **New: `--aggregate=all|simd_slot|wave`**, default `wave`; one value, for
  every report. An unknown value is a flag error (exit 2). See Groups.
- **Removed: `--aggregate_wave` and `--aggregate_wave_only`.** Replaced by
  `--aggregate`.
- **CSV and JSON labels:** `all` keeps what `--aggregate_wave_only` printed
  (CSV `record_type=aggregate` and `wave_count` columns, blank wave fields;
  JSON `"aggregate":true`). `simd_slot` follows the same convention with
  `record_type=simd_slot`, the slot's `cu`, `simd`, `wave_id` filled in and
  its `wave_count`. `wave` keeps the plain run's columns (no `record_type`,
  no `wave_count`).
- **Unchanged, apply to every report:** `--mfma_coexec`,
  `--color_mfma_bubble`, `--simd`, `--verify_trace_duration`, `--jobs`.
- The instruction-row flags (`--inst_stats`, `--inst_addr`, `--inst_avg`,
  `--color_mfma_bubble`) shape rows wherever rows print: `loop_detail`,
  `static_inst`, `dynamic_inst`. Under `loop_summary` they change nothing,
  as today without `--dump_loop_contents`.
- Rows show sums by default and averages under `--inst_avg`, as loop rows do.
- Update every mention of the old flag names (help text, `CLAUDE.md`,
  `CONTEXT.md`, code comments).

## Output framing and diagnostics

- Same document framing as the loop report: CSV header once, JSON array of
  per-group objects, text headings per group, chunks rendered per trace on the
  workers and written in trace order.
- Text: a trace with no group (no waves analyzed) prints a one-line banner,
  like today's "no loops detected" banner, naming waves and SIMD-filter skips.
- MFMA shadow overlap warnings: produced by the whole-wave walk, once per
  offending MFMA, same stderr format as today.
- `[summary]` on stderr: keeps totals and failure counts; the loop-free
  dispatch line prints only under `loop_summary` and `loop_detail` (no other
  report looks for loops).
- Exit codes unchanged.

## Constraints

- C++17, `-fno-rtti`, project namespace, clang-format via pre-commit.
- Waves are streamed: the decoder owns a wave only during its callback, so
  nothing of a wave is kept but running sums.
- Memory: static grows with distinct PCs × groups; dynamic with distinct
  `(PC, occurrence)` keys × groups. A kernel that still has a long loop makes
  the dynamic report large (one key per executed record of the longest wave,
  per group). Acceptable; note it in the flag's help text.
- Core-library modules take plain data and print nothing; the CLI wires them.
  Follow the existing split (profile accumulates, report renders plain data
  resolved up front), so rendering stays parallel per trace.
- Unit tests in `tests/` with `synthetic_wave.h`, covering at least: shadow
  carried across the whole wave, `rel` anchored at the first record,
  occurrence keying across waves that branch differently, the three
  `--aggregate` scopes for every report (loop `simd_slot` records included),
  and each format's rows. The CLI stays untested end-to-end, as today.
- Update `CLAUDE.md` (Architecture) and `CONTEXT.md` (new terms: static /
  dynamic instruction, occurrence, slot group).

## Left to the builder

- Module structure (a new profile module beside `LoopProfile`, a new report
  module or an extension of `LoopReport`), names, and how `WaveAnalysis`
  routes waves to the selected profile.
- The new flags' exact help text.
- Exact text column layout and CSV/JSON field names beyond those listed, as
  long as they follow the loop report's conventions (JSON omits missing keys,
  CSV leaves blanks, text pads).
- Whether to share code with `LoopProfile::add_passes` for the per-record
  arithmetic (encouraged, if it keeps both readable).
