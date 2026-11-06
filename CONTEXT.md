# my_rocperf_tool

Loop and whole-wave timing analysis of AMD GPU thread traces (ATT): how each executed instruction spent its cycles. Terms use the names the code uses.

## Capture

**Run** (output-file prefix):
The files one rocprofv3 run wrote, named by the output-file prefix that leads each file name (the profiled process' pid unless `--output-file` set it), qualified by the file's subdirectory under the scanned directory. Code object ids and dispatch ids are unique only within a run.

**Dispatch**:
One traced kernel dispatch, named by the dispatch id in its trace files' names and unique only within its run. It has a trace per shader engine, so the summary's loop-free verdict is per dispatch, over all its traces. When the directory holds a single trace, a `_dispatch_<n>` in the last component of the `--att_output_dir` path overrides the dispatch id in that trace's name.

**Trace** (shader engine trace, `.att`):
The thread trace of one shader engine (SE) during one dispatch: one `<prefix>_<agent>_shader_engine_<se>_<dispatch>.att` file. The CLI decodes, analyzes and reports each trace separately, and a trace that fails costs only its own report.
_Avoid_: capture, which the code uses both for one trace and for the whole directory of traces and code objects.

**Code object**:
An AMDGPU ELF a run loaded, saved as `<prefix>_<arch>_code_object_id_<id>.out`. Its `code_object_id` names it within the run (ids restart every run); 0 marks a PC the decoder could not attribute.

**ELF vaddr**:
An address in a code object's ELF section address space, not where the run loaded it. The decoder resolves each hardware PC to a `{code_object_id, ELF vaddr}` pair, and that pair is the PC everywhere in the tool.
_Avoid_: load address, runtime address.

## Waves and records

**Wave**:
One wavefront's decoded records in time order, with the `cu` (WGP on gfx10+), `simd` and `wave_id` slot it ran in. Waves reuse slots, so `wave_index`, the wave's position among those its trace analyzed, tells them apart. Waves caught mid-run in lossy traces are included; a missing beginning skews their relative issue and occurrence alignment.

**Record**:
One entry of a wave: one execution of the instruction at its PC, with a category (`JUMP` is a taken branch), `time`, `stall` and `duration`. A record with a null PC ({0, 0}) is unattributed and left out of every count.
_Avoid_: token, except for the two records a classic s_barrier leaves.

**Coalesced s_barrier token pair**:
A classic `s_barrier` (not gfx12's split barrier) leaves two back-to-back records at its PC: a `MESSAGE` token (the signal), then an `IMMED` token (the wait) starting where the first ends. WaveAnalysis merges them into one record, durations and stalls summed (saturating), before anything counts it.

## Instruction reports and groups

**Static instruction**:
An executed PC in a static report, combining all executions of that `{code_object_id, ELF vaddr}` within its group. Rows follow code-object and address order; unexecuted PCs have no row.

**Occurrence**:
The 1-based count of how many times a wave has executed a particular PC so far. Counts restart for each PC in each wave.

**Dynamic instruction**:
An executed `(PC, occurrence)` in a dynamic report, aligning the same occurrence across waves even when they take different branches. Rows follow average relative issue time, with ties ordered by code object, address and occurrence.

**Executing waves** (`exec_waves`):
The waves in a group that executed a row's key at least once. A static row's `n` counts all executions; a dynamic row's `n` equals its executing-wave count.

**Slot** (`WaveSlot`):
The `(cu, simd, wave_id)` location where a decoded wave ran; successive waves may reuse it. A trace targets one CU (WGP on gfx10+), so `(simd, wave_id)` distinguishes its slots.

**Aggregate scope** (`--aggregate`):
The waves grouped together within one trace: `wave` names each wave separately, `simd_slot` combines waves that reused a slot, and `all` combines every analyzed wave. Slot groups sort by `(cu, simd, wave_id)`; same-slot wave groups follow decode order.

**Instruction group summary**:
Per-wave averages of records, summed record duration, stall, idle, optional bubble and MFMA executions over every analyzed wave in the group, including waves with no attributed records. Summed record duration differs from the wave's wall-clock span.

## Loops

**Back-edge**:
A taken branch record whose next record lies at or below it in the same code object. Its `source_addr` is the branch, its `target_addr` the loop header.

**Loop**:
A back-edge that has 2 or more iterations in a wave. Its body is every instruction from header to branch inclusive, so an outer loop's body holds any inner loop.

**Iteration** (pass):
One pass through a loop's body: each taken back-edge ends one, and the body records after the last make a final one. A back-edge taken once is therefore 2 iterations.

**Loop record** (wave, SIMD-slot or trace-wide scope):
A loop's iterations from one wave, the waves of one slot, or every wave of its trace in which the loop was detected; `wave_count` counts those waves. Every metric covers those executions only, so a wave that ran the body without a detected loop contributes to no record of that loop.

## Timing

**Duration**:
A record's cycles from `time`, when the wave first tried to execute it, to completion: its stall plus its issue time (gfx9) or execution time (gfx10+).

**Stall**:
A record's cycles from `time` until it issued; `time + stall` is its issue point.

**Issue**:
A record's duration minus its stall, clamped at 0: its issue (gfx9) or execution (gfx10+) time.

**Idle**:
The cycles from the end (`time + duration`) of the wave's previous attributed record, or from the wave's begin time, to a record's `time`; 0 if they overlap.

**Relative issue** (`rel_issue`, `avg_cycle`):
In loop reports, an issue point relative to the latest loop-header issue; in whole-wave reports, relative to the wave's first attributed issue. Rows sum it as `rel_issue`, and `avg_cycle` is that sum per execution; the first attributed record's relative issue is zero, while its idle includes launch latency.

**Timestamp granule** (`kAttTimestampGranule`):
4 cycles: ATT timestamps land on a 4-cycle grid and the decoder spreads the cycles between two timestamp packets over the records in between, so one record's duration can be off by a granule.

**Duration check** (`--verify_trace_duration`, `DurationCheck`):
Whether each record's duration is at least its stall, as the decoder's definition of duration requires. `off` checks nothing; `warn` reports each offending record and keeps it, issuing in 0 cycles; `error`, the default, rejects the trace, reporting nothing for it, and the run exits 1.

## MFMA

**MFMA**:
A matrix fused multiply-add: any `v_mfma_*` instruction, but not the sparse `v_smfmac_*` family.

**MFMA shadow** (`kMfmaShadowCycles`):
The 16 cycles from an MFMA's issue point during which it is assumed to keep the MAI pipe busy, so a co-issued non-MFMA executes for free. Under `--mfma_coexec`, the shadow end moves only forward, independently for each loop walk or continuously across the entire wave in instruction reports.

**MFMA shadow overlap** (`MfmaShadowOverlap`):
An MFMA that issues more than a timestamp granule inside the previous shadow, reported on stderr as a warning: a sign that `kMfmaShadowCycles` is wrong for the ISA.

**MFMA utilization** (`mfma_time`, `mfma_util`):
For loops, the body's MFMA count times `kMfmaShadowCycles` gives `mfma_time`, divided by average iteration duration for `mfma_util`. For whole-wave reports, utilization is MFMA executions per wave times the shadow length divided by summed record duration per wave; both reports give it with or without `--mfma_coexec`.

**Bubble**:
Computed only under `--mfma_coexec`: the part of a non-MFMA record's span, `time` to `time + duration`, past the current shadow's end: cycles no MFMA covered. Rows sum it per instruction; `avg_bubble` is the loop record's total per iteration or the instruction group's total per wave.
