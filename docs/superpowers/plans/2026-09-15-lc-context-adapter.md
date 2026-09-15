# LC Context Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a usable, separately versioned LC context decision path and a restartable four-case experiment handoff.

**Architecture:** New context assessment and job adapters consume the reviewed facts module and existing immutable packet, menu, memory and transport primitives. Preserve every v1 file and use a new experiment namespace. A private harness connects actual fixed packets without rebuilding source signals.

**Tech Stack:** Existing Python/pandas/pytest, stdlib JSON/hashlib/SQLite, existing orchestrator transport. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-15-lc-context-discrimination.md`

## Global Constraints

- No production changes or live orders.
- No source rerun or date extension.
- New role namespace: `lc_context_discrimination_v1`.
- No RSI subtype is removed. No intact-parent, daily/4H agreement, same-hour reclaim or minimum ceiling-distance filter is added.
- A reviewed explicit reject is zero exposure; null, missing/invalid transport, incomplete review and material errors are unknown output.
- New artifacts must not overwrite v1 requests, memory snapshots, grades, locks or outcomes.
- At most four fresh specialists/four critics, one per fixed case, no retries or answer repair.
- Lock raw specialist outputs before critics and all grades before any new outcomes.
- Fixed $50,000 notional,12bps round-trip,90seconds processing,zero routing,15minute exclusive expiry,close−2.7ATR stop,actual-entry2R,decision+1440minute deadline.
- Keep the user-selected existing research branch. No push, PR, new dependency, paid fine-tuning or full-source replay.

### Task 1: Context assessment and immutable job adapter

**Files:** Create `scripts/research/lc_context_assessment.py`, `scripts/research/lc_context_jobs.py`, `tests/research/test_lc_context_assessment.py`, `tests/research/test_lc_context_jobs.py`. Do not edit old modules or the reviewed facts module.

**Interfaces:** Consume `describe_lc_context`, existing `compile_menu`, `_validate_snapshot`, hash/JSON/citation primitives and generic storage operations on `ResearchJob`. Produce:

```python
def build_context_request(packet, memory_snapshot, master_brief, settings):
    """Return sealed, context-only request; never mutate original packet."""
def validate_context_request(request):
    """Raise ValueError unless all bindings, context, menu and instructions match."""
def grade_context_choice(request, choice):
    """Return list of deterministic errors; raw strings preserve duplicate-key rejection."""
def build_context_review_request(request, choice):
    """Bind exact raw specialist string and new request."""
def grade_context_review(review_request, response):
    """Return deterministic critic errors, without judging trade profitability."""
def gate_context_choice(request, choice, review):
    """Return status/research_plan/errors; never execution authorization."""
class ContextResearchJob(ResearchJob):
    """New prepare and lock_grade use context validators, not v1 grading."""
```

- [ ] Write literal failing tests first. Reuse old public test `inputs/packet` fixtures, not private market files. An old v1 geometry failure with valid source must allow new `enter` and `wait_5m_high` choices; evidence unknown must not produce a credited rejection. Implement fixture `context_request()` using a sealed empty methodology snapshot for generic API tests and a versioned, source-only context brief. Actual harness separately enforces the five approved records.

```python
def test_context_choice_survives_old_geometry_failure():
    req = context_request(prior_bb=.2)
    answer = choice(req, interpretation='support', plan_id='wait_5m_high')
    assert grade_context_choice(req, answer) == []
    assert set(req['plan_menu']['plans']) == {'enter', 'wait_5m_high', 'reject'}
    assert 'conditions' not in req['source_packet']

def test_unknown_source_is_not_a_profitable_rejection():
    req = context_request(current_validated=False)
    assert grade_context_choice(req, choice(req, interpretation='oppose', plan_id='reject'))
    assert grade_context_choice(req, choice(req, interpretation='uncertain', plan_id=None)) == []
```

Use self-contained fixture/choice builders with the existing LC exact schema. Test actual invalid-source operand examples, not mocks of gates.

- [ ] Run the new focused tests and record expected RED before implementation.

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q tests/research/test_lc_context_assessment.py tests/research/test_lc_context_jobs.py
```

- [ ] Implement the adapter. Preserve original source hash privately/as inert identifier, but market packet must not contain old mandatory conditions, policy-specific curriculum, legacy instructions or an old pass/fail authorization. Use a whitelist projection of needed source fields; retain current/previous features, provenance, native diagnostic, six evidence layers and parent views, plan/limitations/availability. Recompute facts from this context projection and bind them without a self-referential hash. Keep `context` as a separately bound request member if that makes recomputation simpler.

New packet-owned group catalog includes current,previous,native,context,parent4h,parent1d,1m,5m,15m,1h,4h,1d,economics,limitations and approved curriculum IDs. Group citations must actually resolve to their request-owned data. Rebuild the legacy candle evidence catalog solely for compiler compatibility; never grade using the old conditional/LC grader.

Compiler output restricted to enter/wait_5m_high/reject; replace outer and nested instructions and reseal. Fix settings/economics to spec constants; reject silently changed stop/ATR/notional/cost/horizon or menu. If valid plan construction is impossible, prepare must fail explicitly; do not make a fake zero-return job. No alternative wait1m/wait15m remains legal.

Data-readiness checks must not use geometry as permission: reconstructed native long, validated current/previous operands, known complete1m/5m observations, and known provenance-backed parent views (known absence/break is allowed). An unavailable required operand permits only uncertain/null; a known nonnative candidate is invalid input to this LC-only adapter. Historical macro/defaults remain disclosed, not imputed or an automatic veto. Remaining structural interpretation is reviewed judgment.

Retain exact existing LC response fields: case_id,packet_sha256,memory_sha256,interpretation,plan_id,supporting,opposing,unknowns,structural_invalidation. Critic fields: case_id,reviewed_sha256,complete,material_errors,notes. Same text/list/citation limits and support/oppose/uncertain mappings. New prompt requires discussing larger context, contrary observations, horizon, sequence and invalidation without inventing hard gates. No confidence/probability field.

Reject the two full v1-specific record IDs from the spec in snapshot or brief. Validate copied snapshots with existing chronology/hash rules; actual role-run record identity is harness-owned. Require a context-policy/version-bound brief; no arbitrary v1 master brief accepted. Validate outer/nested instruction, context arithmetic, plan and all memory/menu/request bindings on every grade.

- [ ] Implement jobs by subclassing only generic storage as needed. Override prepare and lock_grade so new validation/gating is recomputed from exact captured raw answers. Check namespace on reopen, not just creation, so a ContextResearchJob cannot accept a v1 directory or tampered request. Do not monkeypatch global imports. Retain original stage order, transport validation and byte immutability.

```python
def test_context_job_reopens_and_recomputes_grade(tmp_path):
    job = ContextResearchJob(tmp_path/'case')
    req = context_request()
    job.prepare(req)
    answer = raw_choice(req, plan_id='enter')
    job.capture('specialist', answer, valid_transport('specialist'))
    review = raw_review(build_context_review_request(req, answer))
    job.capture('reviewer', review, valid_transport('reviewer'))
    reopened = ContextResearchJob(tmp_path/'case')
    assert reopened.lock_grade()['research_plan']['action'] == 'enter'
```

Literal fixture helpers must supply real exact raw/schema/transport metadata. Test changed raw response, forged grade, invalid transport, v1 reopen, premature reveal, altered context/menu/settings/source/memory, invalid citations and duplicate JSON keys. A valid reviewed reject returns a reject plan; null/failed critic/invalid transport returns no research plan. Test same real conditional resolver with synthetic bars for immediate and wait90s clocks. No market outcome access.

- [ ] Run focused green and full `tests/research` once; self-review; commit only the four owned files. Report RED/GREEN commands and results plus remaining concerns. Controller dispatches task-scoped independent spec/quality review before Task2 integration.

### Task 2: Fixed-case runnable preparation and actual comparison boundary

**Files:** New private `results/lc_context_discrimination_2026_09_15/experiment.py`, `test_experiment.py`, and `read_chunk.py`; public `docs/knowledge/lc_context_adapter_2026_09_15.md`; update `PROJECT.md` and `docs/knowledge/MEMORY.md`. New preparation under `run_v1/`; do not overwrite `source_smoke.json`.

**Interfaces:** Consume Task1 APIs and existing source packet/selection hashes from spec, `ResearchMemory`, evidence envelopes/actual capture verification and `replay_isolated_sleeves`. Produce a local CLI with `prepare`, `verify`, `capture`, `prepare-review`, `grade`, `reveal`, `status` subcommands. All writes immutable, no command starts network/model/order calls.

- [ ] Freeze controller-authored source-only brief and independently review it against the five existing methodology/code-map records. Build a new SQLite store from those exact reviewed record/review data, leaving old database untouched. Snapshot training_end/as_of Jan1 UTC; no case-bearing outcome lessons. Verify reopen hashes and exclusion of both v1 records. Preparation without an approved brief must fail.
- [ ] Tests first: changed input hash, wrong case roster, old record/brief, changed request, out-of-order critique/reveal and invalid transport fail. Use literal small fixtures and new adapter; no archive dependency in unit tests. Prepare source-only actual4 requests from the pinned old packets, not the old specialist requests. Reject a changed input file instead of resealing it silently. Preserve original18 source rows as full-code cohort; expose only the assigned case to each market role.

```python
def test_reveal_requires_every_frozen_case_graded(tmp_path):
    run = fixture_run(tmp_path)
    run.prepare()
    with pytest.raises(ValueError, match='grade'):
        run.reveal()
```

- [ ] Add separate exposure ledger from existing prior registries and exact case clocks, freeze sources/spec/code/curriculum/requests/role protocol before role access. No new economic outcome reads during preparation. `status` explicitly distinguishes request-ready, captures, reviewed grades and outcomes.
- [ ] Task review before actual roles. Controller runs four fresh packet-only specialists and four independent packet-only critics, exact captured delivery/raw results, no retries/repairs/replacements. Requested model comes from explicit recorded role protocol; do not invent actual snapshot or token measurements.
- [ ] Lock all grades before reveal. Reuse existing same-source outcome scorer and isolated books: full18 A/B separately from matched4 A/B/C. Before loading any new future prices, require immutable source/code/role freezes and every assigned case grade, including locked invalid statuses. Preserve null aggregate if missing output could affect occupancy. Report valid-case contrasts separately. No pooled archetypes, optimized cutoff or holdout claim.
- [ ] Verify saved source/role/result hashes and replay reproducibility, independent final review, fresh tests and actionable report. If an integration check prevents roles/results, deliver the actual runnable state and exact failing check, not fabricated completion. Local checkpoint commits only; no push/PR/live changes.
