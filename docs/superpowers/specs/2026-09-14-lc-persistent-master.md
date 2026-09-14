# LC-first persistent research master, version 1

## Approval and scope

The user approved the LC-first three-book design, asked how a persistent master
learns, then explicitly requested planning followed by completion. Implement and
exercise an offline research version, not an always-running live trader or model
weight training. Keep the master system-wide curriculum, but LC is its first
specialist assignment. Existing seventeen production archetypes remain unchanged.

## Fixed experiment

Source window: [2026-01-01T00:00:00Z,2026-04-01T00:00:00Z), monthly independent
30-day warm-ups, original Binance BTCUSDT minute archive SHA256
5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035.
This is retrospectively exposed development history, not a pristine holdout.
Do not extend dates because a rule has few passes or poor outcomes.

Retain every long LC native pre-winner emission from the unchanged signal replay.
Do NOT use the old hourly_eligible collector, which already filters on H2.
Preserve per-candidate native diagnostics and previous features even if missing.
This is pre-dedup native eligibility, not full runner entry/management parity.
All17 signal definitions and original per-archetype cooldowns remain intact.

Three independently replayed LC books: native immediate; native plus the
deterministic nested-child reference; identical reference plus agent judgment.
Use common fixed $50,000 notional,12bps round-trip cost,90s processing,zero routing,
15min exclusive entry expiry,stop=candidate close-2.7*ATR14,actual-entry 2R target,
absolute candidate-decision+1440min deadline. Existing causal occupancy/scoring
rules apply. Missing/invalid required evidence or invalid assessments is unknown,
not a winning rejection. Defined structural failure is a no-order rejection.
No combined portfolio, funded-account or native-management claims.

The full source population supports two deterministic books. Agent comparison is
a separately labelled bounded subset: first four previously unassessed native
LC candidates in (decision_time,candidate_id) order, without structural or outcome
selection. Merge existing prior-case registries and exclude unidentified cases
by matching decision clock conservatively. Shortages remain shortages.
At most four fresh specialists and four independent critics, no retries or answer
repair. One source-only Astra master design has already run; one bounded master
brief using the frozen curriculum may run before specialist requests. Requested
role models/snapshots and observable usage must be recorded honestly.

All source, policy, curriculum, input packets and role instructions are frozen
before market roles. Lock raw responses/actual delivery captures before review;
lock critiques and grades before revealing prices. No post-reveal policy edits.

## Nested structure reference: lc_nested_child_rejection_v1

This source-only Astra-authored hypothesis is not asserted to be trader doctrine
or the sole valid LC thesis. It avoids forcing every LC trade to sweep a parent
floor. The previous completed hourly candle is a minimal compression-box proxy,
not proof of a developed auction range.

Let t be the candidate hour, p the immediately preceding completed hour,
C=[p.low,p.high], and P the causal4H/N3 parent strictly available before t opens.
Evaluate separate pass/fail/unknown facts:

1. Native: captured LC native_signal.direction is long.
2. Evidence: current/previous candle clocks and required numeric fields match
   completed same-stream OHLCV; reconstruction has locked source/code/formula
   provenance; P has valid anchors and continuous lineage through decision.
3. Compression: p.bb_width<=0.06.
4. Nesting: finite positive P.low<=C.low<C.high<=P.high, P.low<P.high.
5. Rejection: t.low<C.low<t.close<P.high.

Known false dominates unknown; otherwise any unavailable operand yields unknown.
Future data is invalid/unknown research input and must never authorize entry.
Native source numeric truth is distinguished from authentic original live receipt
truth: verified deterministic reconstruction is enough for THIS offline reference,
but never described as receipt-authenticated. Missing historical macro/derivatives
are explicitly unavailable; not filled with current information or defaults.

Daily context, internal levels and completed15m/5m/1m sequence inform judgment,
not extra undeclared hard gates. Parent high is not automatically a destination.
Stop and structural invalidation remain separate. No new thresholds or optimized
parameter search in this version.

## Agent interface

The code owns named condition states and immutable evidence groups. Agents do not
echo arithmetic or attach a pass/fail status to every natural-language sentence.
Packet contains six completed timeframes, feature provenance, current/previous
LC inputs, both parent views, condition states, executable finite menu and the
approved curriculum snapshot. Retain supplied evidence with stable IDs.

Specialist exact response: case_id,packet_sha256,memory_sha256,interpretation,
plan_id,supporting,opposing,unknowns,structural_invalidation.
Interpretation is support|oppose|uncertain. Supporting/opposing/unknowns are lists
of {text,evidence_ids}; evidence IDs refer to packet-owned groups and curriculum
records; citations can be shared and condition facts referenced as groups.
No sentence-status polarity labels and no calibrated confidence requirement.
Support maps to one legal enter/wait menu plan; oppose maps to reject; uncertain
maps to null. Structural fail permits reject only; structural unknown permits null
only. Code never silently converts an invalid positive answer into rejection.
Both evidence and contrary/missing-evidence consideration must be explicit; empty
lists are valid when accompanied by uncertainty/interpretation supported elsewhere.

Critic exact response: case_id,reviewed_sha256,complete,material_errors,notes.
Each error has {category,evidence_ids,explanation}; categories factual,chronology,
missing_required,unsupported_plan. Mere discretionary disagreement and stylistic
preferences are notes, not material errors. References may support the response
as a whole. Review is bound to exact request and raw answer. Code checks schemas,
IDs, parent/condition authorization and menu ownership; critic checks interpretation.
Invalid transports, bindings or material errors leave validated agent plan null.
No raw-choice economic credit is needed in this new experiment.

## Persistent master memory

Use a local stdlib SQLite store: immutable proposal records, append-only review
records and content-addressed snapshots. No external database or paid dependency.
Records contain kind,author,content,source_refs,tags,available_at,event_end,case_ids.
Kinds: doctrine,code_map,hypothesis,lesson,market_state. Doctrine/code_map/hypothesis
can be timeless methodology (available_at null only with no case_ids/event_end);
case-bearing records require dated availability and event_end. This is a curation
attestation, not semantic proof that prose is free of future information.
Proposals never enter approved retrieval without a separate named reviewer and
nonempty rationale. Review author must differ from proposal author; this enforces
declared separation, not cryptographic identity/authentication.

Snapshot pins approved record bytes, decision cutoff, training_end and exclusion
IDs. Date-bearing content must be available by cutoff and have event_end strictly
before training_end; matching excluded case IDs are forbidden. Approval and
recording clocks are research clocks, never falsely substituted for market times.
Freeze snapshots for evaluation; new lessons cannot alter old snapshots. Retrieval
checks content hashes; duplicates are idempotent, changed ID content is rejected.
Preserve supersession through new records/reviews, never overwrite old decisions.

Separate immutable job records support request export, raw-answer capture, critic
binding, grading and outcome-lock stages across process restart. Runtime uses
externally supplied role outputs (CLI/orchestrator transport); it does not create
unbounded model loops. No browser/network/order tools are granted to market roles.
Store model/role identifiers and actual captures where available; do not invent
token use, independence, attention, latency or model snapshot certification.

The LC curriculum uses source-linked methodology and the source-only master brief,
not the entire MEMORY file or reports containing known outcomes. Outcome-bearing
lessons are proposed only after reveal and remain unapproved until independently
reviewed. Demonstrate reopening the database, stable snapshot identity, exclusion
of a newly proposed lesson, and persistence of a completed job.

## Deliverables and completion

1. Reusable tested source-only LC collector, preserving all native candidates.
2. Tested durable memory and restartable research jobs with explicit trust limits.
3. Tested LC structural facts and revised judgment/review adapters using unchanged
   conditional menus and isolated replay, no old frozen-module edits.
4. Actual Q1 source reconstruction and frozen bounded master/specialist/critic run;
   code books plus separately labelled matched agent subset, all results locked.
5. Independent quant/code review, report, CLI handoff and clear remaining limits.

If source coverage/role validity prevents an economic result, report that exact
failure rather than substituting a profitable case or claiming completion of the
blocked experiment. No live deployment, pushes, PRs, fine-tuning, perpetual daemon,
autonomous knowledge promotion or all17 performance certification in this version.
