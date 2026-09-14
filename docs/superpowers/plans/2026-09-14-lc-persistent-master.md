# LC Persistent Research Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Deliver and exercise a persistent offline LC research desk with a frozen nested-structure versus agent comparison.

**Architecture:** New source collection, memory and LC judgment modules reuse existing causal evidence, conditional menus and isolated books. Frozen old experiment modules are never edited. Local source/model artifacts remain separate from public code and documentation.

**Tech Stack:** Existing Python/pandas/TA-Lib, stdlib SQLite/JSON/hashlib, pytest, bounded orchestrator model roles. No new dependency.

**Spec:** `docs/superpowers/specs/2026-09-14-lc-persistent-master.md`

## Global Constraints

- Offline only; no live/config/fusion/archetype changes, new dependencies, push, PR or fine-tuning.
- Do not edit existing frozen research modules or artifacts. Add new modules/tests; preserve unrelated user changes.
- Source window [2026-01-01T00:00:00Z,2026-04-01T00:00:00Z), monthly independent30day seeds; development data, not pristine holdout.
- Retain native pre-winner long LC emissions before any H2/nested-structure filtering; never call hourly_eligible for collection.
- Common $50000 notional,12bps,90s processing,zero routing,15min exclusive entry expiry,close-2.7ATR stop,actual-entry2R,decision+1440min deadline.
- Independent LC-native,LC-structure,LC-structure-agent books; full code cohort distinct from four-case agent subset.
- First four previously unassessed native cases by decision_time/candidate_id; no replacements or outcome selection; four specialists/four critics maximum plus one curriculum master brief.
- Freeze sources/policy/memory/requests before roles, raw responses before critics, grades before outcomes; null is not reject or avoided loss.
- Model/capture metadata are observed or explicitly unknown; declared reviewer identity is not authentication.
- Write tests before code, run focused tests during development and full tests/research before each implementation commit.

### Task 1: Source-only native LC population

**Files:** Create `scripts/research/lc_source_population.py`; create `tests/research/test_lc_source_population.py`.

**Interfaces:**

```python
def collect_native_lc(rows, start, end):
    """Return every native long LC row with current/previous features and diagnostics."""
def prepare_month(month, out):
    """Replay one fixed Q1 month; save immutable source-only source.json."""
```

Input rows follow run_signal_replay output. Current features and prior features
are copied, not mutated. Native eligibility is precisely
row.output.engine_signal.archetypes.liquidity_compression.native_signal.direction
== 'long', regardless of selected, H2, current numeric permission or other winners.
Require valid unique ordered decision clocks; missing previous features remains
None on retained candidates. Return candidate_id='hourly-lc:'+UTC ISO decision,
track='hourly',decision_time,setup_open,features,previous_features,native_diagnostic,
native_emitted=selected,feature_available_at and observation identity/timing fields.

- [ ] Write failing collector tests using hand-built replay rows:

```python
def test_native_long_survives_prior_compression_failure():
    rows = [row('2026-01-02T00:00:00Z', bb_width=.2),
            row('2026-01-02T01:00:00Z', direction='long', selected=False)]
    result = collect_native_lc(rows, '2026-01-01T00:00:00Z', '2026-02-01T00:00:00Z')
    assert len(result) == 1
    assert result[0]['previous_features']['bb_width'] == .2
    assert result[0]['native_emitted'] is False
```

Implement fixture row in tests matching the consumed output fields; test missing
previous, nonlong exclusion, start inclusive/end exclusive, chronological errors,
deep-copy isolation and an absent feature row interrupting previous-hour continuity.
- [ ] Run focused tests red, implement collector, rerun green.
- [ ] Add month/source CLI with `--month` in2026-01/02/03 and `--out` new directory.
  Reuse unchanged run_signal_replay, complete-hour aggregation, side_effect_guard,
  source hash/TA-Lib checks and build_parent_ledger4H/N3,daily/N3 from the old private
  preparer. Reuse old private helpers only lazily inside CLI. Preserve their exact
  source/runtime identity and report this local reproduction dependency.
  Do not run the old minute detector/selector or import modules that auto-score.
  Save per-month source_path/hash, seed/start/end, hourly_input_hash, all candidates,
  parent ledgers, source/code/config manifests, runtime and missing-input limits.
  Verify source/code hashes before and after replay. Emit progress every120hours.
  Immutable writes reject unequal overwrite; reject existing output before running.
- [ ] Test month parsing, source-only output boundary and immutable-output handling
  without running a long historical replay in the unit suite.
- [ ] Run full research suite and commit only the two files. Controller reviews,
  then launches three source jobs with at most two concurrent processes.

### Task 2: Durable reviewed memory and frozen snapshots

**Files:** Create `scripts/research/master_memory.py`; create `tests/research/test_master_memory.py`.

**Interfaces:**

```python
class ResearchMemory:
    def __init__(self, path): ...
    def propose(self, record): ...  # returns content hash ID
    def review(self, record_id, reviewer, verdict, rationale): ...
    def snapshot(self, *, as_of, training_end, excluded_case_ids=(), tags=()): ...
    def load_snapshot(self, snapshot_id): ...
    def close(self): ...
```

Record exact fields/kinds and chronology follow spec. available_at/event_end null
only for case-free methodology kinds doctrine/code_map/hypothesis; lesson and
market_state require dated clocks and nonempty case_ids. event_end>=available_at
is NOT required (a lesson can become available after its event ends); both clocks
must be timezone aware and <=snapshot cutoff; event_end<training_end. training_end
must be <=as_of. Require nonempty author, content object, source_refs list of
nonempty strings, unique tags/case IDs. Reject NaN/Infinity/duplicate identity.
Review verdict approve|reject; latest review controls future snapshots, existing
snapshots unchanged. Reviewer differs from author; store rationale and actual
recording clock. Proposal IDs hash canonical record bytes. Review log append-only.
Snapshot body includes exact selected records, review IDs, cutoffs/exclusions/tags,
not a mutable query pointer; exclude records whose tags do not intersect requested
tags (empty requested tags means all). Verify stored record/review/snapshot hashes
on reads. Reopening must preserve all identities. SQLite transactions protect
atomic writes; no unsafe pickle or SQL string interpolation.

- [ ] Write failing real tmp_path SQLite tests:

```python
def test_unreviewed_memory_does_not_enter_snapshot(tmp_path):
    db = ResearchMemory(tmp_path/'memory.sqlite')
    rid = db.propose(doctrine_fixture())
    first = db.snapshot(as_of='2026-01-01T00:00:00Z', training_end='2026-01-01T00:00:00Z')
    assert first['records'] == []
    db.review(rid, 'independent-reviewer', 'approve', 'Checked against named source')
    second = db.snapshot(as_of='2026-01-01T00:00:00Z', training_end='2026-01-01T00:00:00Z')
    assert [r['id'] for r in second['records']] == [rid]
    db.close()
```

Test self-review refusal, future availability, overlapping labels, excludedcase,
unchanged frozen snapshot after later rejection, reopen, tampering detection,
duplicate idempotence, invalid record schema and tag filtering. Tests must derive
expected eligibility independently, not invoke the filter to construct expectation.
- [ ] Run red, implement minimal store, run green.
- [ ] Add `propose --db --record`, `review --db --id --reviewer --verdict --rationale`,
  `snapshot --db --as-of --training-end` CLI with JSON stdout and clear errors.
- [ ] Full research suite; commit only memory module/tests; controller task review.

### Task 3: LC structural facts, judgment and restartable jobs

**Files:** Create `scripts/research/lc_master_assessment.py`, `scripts/research/master_research_jobs.py`; create corresponding tests under `tests/research/`.

**Interfaces:**

```python
def build_lc_packet(raw, minute_bars, ledgers, provenance, case_id): ...
def evaluate_reference(packet): ...  # {conditions:{id:{status,evidence_ids}},status}
def build_lc_request(packet, memory_snapshot, master_brief, settings): ...
def grade_lc_choice(request, choice): ...  # list[str]
def build_lc_review_request(request, choice): ...
def grade_lc_review(review_request, response): ...  # list[str]
def gate_lc_choice(request, choice, review): ...  # {status,research_plan,...}
class ResearchJob:
    def __init__(self, directory): ...
    def prepare(self, request): ...
    def capture(self, role, raw_response, provenance): ...
    def lock_grade(self, grade): ...
    def authorize_reveal(self): ...
    def save_outcome(self, outcome): ...
```

Source-only packet construction reuses completed_candles six layers, parent_asof
strict-before and ledger transitions (may reuse private parent_view lazily;
public pure validators must not require ignored files). Include exact current and
previous hourly OHLC matching source, native diagnostic, feature availability,
source manifest digest and reconstruction verification. Distinguish absent parent
(fail) from incomplete history (unknown). Require exact prior-hour continuity.
Check source stream/instrument matches both parents and native inputs; reject
future observations. Validate bound parent anchor availability and hourly lineage
continuity through decision. Any known break invalidates reference. Daily geometry
is descriptive, not an extra gate. Use spec inequalities exactly; known-fail
dominates unknown. No teacher attribution for new numeric geometry.

Use existing compile_menu on new packets without changing old module prompts.
New request replaces legacy instructions entirely; source material is data.
Evidence group IDs reference complete groups: current,previous,native,parent4h,
parent1d,1m,5m,15m,1h,4h,1d,conditions,economics,limitations and approved curriculum.
Include canonical sealed packet, snapshot and menu hashes. Master brief is source-
only versioned content; freeze its hash. Text fields <=1200characters; lists <=8;
supporting/opposing/unknowns evidence IDs nonempty/unique/known, text nonempty.
All exact response and critic fields/mappings are in spec. Reject bool numbers,
unknown keys, stale hash/case/memory, illegal plans, invalid positive authorization.
Require at least one grounded supporting item for support and opposing item for
oppose; uncertain requires unknowns. structural_invalidation is a grounded item
{text,evidence_ids}. Critic complete must be bool; any material_errors blocks;
notes are nonblocking. Keep outcome interpretation separate from schema quality.

Job filesystem is new directory, immutable JSON/raw artifacts, never overwrite
unequal bytes. Prepare pins request; specialist capture pins raw string+declared
transport provenance; reviewer capture allowed only after specialist capture;
grade after both captures; reveal authorization only after grade; outcome only
after authorization. Every transition verifies all existing hashes. Equal replay
is idempotent; changed data rejected. Do not certify truth/identity from caller
attestations. Input model metadata has exact role,agent_id,requested_model and
actual_model(nullable),capture_sha256(nullable),transport_valid bool; real workflow
must independently validate captured chunk returns before setting transport_valid.
Snapshot request contains approved records so no uncontrolled lookup during role.

- [ ] Write failing tests with literal nested fixture P=[90,120],C=[100,110],
  t.low=99,t.close=105,priorBB=.04: pass. Change t.low=100 -> fail; pBB=.07 ->fail;
  C.high=121 ->fail; missingparenthistory ->unknown; brokenlineage ->fail;
  prioravailableat>setup ->unknown; mutate future minute tail ->same packet.
- [ ] Add request/choice/critic tests for all mappings, stale bindings, citations,
  unsupported plan, null versus reject, shared evidence support and discretionary
  critic notes. Use real conditional menu and existing resolver for a hand-built
  enter/wait fixture, not duplicated accounting.
- [ ] Add filesystem job tests: restart between every state, out-of-order reveal
  refused, changed response refused, tampered earlier artifact refused, unknown
  transport cannot authorize a valid research plan. Outcome saving never changes
  pinned request or snapshot.
- [ ] Run red, implement minimally, run green/full research suite; commit new files
  only; controller task review before real market requests.

### Task 4: Actual frozen experiment and master handoff

**Files:** New ignored `results/lc_persistent_master_2026_09_14/`; public `docs/knowledge/lc_persistent_master_2026_09_14.md`; update `docs/knowledge/MEMORY.md`.

- [ ] Verify all3 monthly source artifacts and original source/code hashes. Preserve
  every native candidate, missing-input field and exposure reason; no reclassification
  based on prices after entry. Freeze first4 previously unassessed candidates before
  market role calls. Freeze source, policy, settings, code, role instructions and
  exact source-only curriculum, including master-author source recommendations.
- [ ] Populate actual memory with curated doctrine/code-map/hypothesis records,
  independent reviewer approvals and source references; build a Q1 frozen snapshot
  with training_end=2026-01-01T00:00:00Z. Do not load known-outcome reports or raw
  project MEMORY into market role context. Demonstrate DB reopen and stable snapshot.
- [ ] One source-only master brief request over that snapshot and rule policy, no
  candidate or future prices. Capture exact response and role metadata, persist it
  as a proposal; source-only independent review before using it in frozen requests.
- [ ] New local harness uses ResearchJob and evidence_guard envelopes. Fresh
  specialists/critics read only their exact chunks and save actual inner runtime
  returns before emission. No repairs/retries; malformed roles get locked failures.
  Validate receipts independently, store in jobs, grade through real LC adapter.
- [ ] Lock all grades before revealing any new outcomes. Replay two full-cohort code
  books and matched three-book sample through existing replay_isolated_sleeves.
  Enforce fail=reject,unknown=null and invalid agent=null. Report counts,coverage,
  costs,net PnL,risk,missed winners/avoided losses only for valid alternatives.
- [ ] Persist outcomes separately and one post-reveal lesson proposal; verify it
  cannot enter the old frozen snapshot or approved future retrieval without review.
  Do not approve a profitability lesson from four cases. No periodic daemon.
- [ ] Independent quant review verifies locks, candidate accounting and repeat
  result hashes. Broad final code review covers Tasks1-3 and any private-harness
  deviations. Final fresh full tests, git diff check and report commit. No push/PR.

Report distinguishes implemented runtime, real model usage, source reconstruction,
economic findings and unproved master skill. Give exact local CLI commands to
resume/export jobs and memory without sending data to an unapproved endpoint.
If structural pass count or usable agent count is zero, report that as the result,
not an excuse to change the frozen rules or silently sample extra cases.
