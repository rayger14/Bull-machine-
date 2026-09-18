# LC Consolidated Historical Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete one finite historical comparison that measures whether reviewed, multi-timeframe LC selection adds value over immediate entry and generic minute confirmation.

**Architecture:** Reconstruct a fixed monthly source census, freeze a chronological agent cohort, collect every terminal decision before scoring, then replay independent books under all registered execution scenarios. Reuse the reviewed evidence/job/resolver contracts; add only a versioned source wrapper, campaign ledger and accounting/report layer. Preserve every prior experiment unchanged.

**Tech Stack:** Existing Python, pandas, numpy, TA-Lib, pytest, canonical JSON and local immutable artifacts; no new dependencies.

**Spec:** The authoritative consolidated research contract is embedded below. It prospectively replaces the **unexecuted** D1 batch in `docs/superpowers/specs/2026-09-15-lc-economic-validation-protocol.md`; that historical document is retained unchanged. No D1 cases or market calls have run.

## Status and global constraints

**September17 restart override:** the census and two-role execution lifecycle
are implemented; two reviewed cases and the authorized three-case exploratory
reveal are complete. The full campaign is not complete. See the
[current CLI handoff](../../knowledge/CLI_HANDOFF_2026_09_17.md) before following
historical launch instructions below. User authorized checkpoint publication;
the next milestone is a separate single-assessor version, not automatic spending.

**Sampling amendment approved after the completed census:** the user approved
chronological selection of all previously unassessed eligible cases, capped at30,
allowing gaps caused by prior assessments. This supersedes the consecutive-block
requirements below, which remain as the historical specification. See
[sampling amendment](../../knowledge/lc_sampling_amendment_2026_09_15.md).
The source census is complete; campaign market roles and scoring have not run.
The implementation-not-started statement below is the original planning status.

- September 15, 2026: user chose **up to 30 setups / 60 calls**. This is a ceiling, not proof that 30 eligible setups exist or a token/dollar guarantee.
- Planning and read-only data/design review are complete; implementation, expanded source collection, market roles and economic scoring have **not** started.
- Stay on `quant/archetype-evidence-audit` in the existing workspace. No new worktree, production/config/fusion edits, live orders, dependencies, push or PR.
- Preserve old code and hash-bound sources, requests, raw responses, grades and outcomes. New files/new run namespace only, except current handoff documentation.
- No additional four-case pilot, parameter search, context ablation, new hard parent filter, discretionary exit, new archetype or extra master-model call in this campaign.
- LC remains isolated. Minute bars supply nested evidence and execution for hourly LC; this is not the separate minute equal-low strategy or validation of all17 archetypes.

## 1. Research contract

### Question and alternatives

Does the frozen, reviewed agent selection package improve net results relative to both immediate LC entry and a simple automatic confirmation rule? Four cases cannot answer this reliably. Thirty decisions can provide a larger descriptive comparison, but neither thirty nor millions of candles guarantee sufficient independent trading evidence.

Alternatives considered: repeat tiny pilots; immediately optimize gates/CPCV; run one fixed broad baseline plus bounded agent comparison. Choose the third. Optimization before establishing usable judgment and accounting would answer a different question and multiply selection bias.

### Fixed data scope and population

Study ID: `lc_consolidated_v1`. Private root: `results/lc_consolidated_2026_09_15/run_v1/`.

- **Broad code census:** `[2024-01-01 00:00, 2026-08-01 00:00)` UTC, exactly31 monthly units.
- **Agent selection interval:** `[2026-01-01 00:00, 2026-08-01 00:00)` UTC.
- Each source unit uses the existing30-day seed, all17 native archetypes and **every native pre-winner LC long**. Selected-winner metadata, H2, structure, future returns and source-readiness preferences do not filter the population.
- Signal state resets monthly, matching the reviewed collector. This is a monthly-reset reconstructed detector, **not continuous live-state parity**. Position books remain continuous across month boundaries.
- Reuse January–March source artifacts only after verifying their exact original source/code/config/runtime manifests and input hashes. Reconstruct the other28 months in a new collector; do not extend the old Q1 allowlist in place.
- Finish the entire census before roster selection. A failed/missing month cannot silently disappear. Shared source/hash/chronology failure stops the campaign before market roles.
- Freeze the previously assessed-ID union from source-only exposure/assessment registry metadata, including the completed Jan19 pilot and older failed assessments. Record exact registry paths/hashes and matching IDs; a missing registry is an integrity failure, not an empty exclusion list.
- Sort native candidates chronologically. Select the **earliest consecutive block of30** inside the agent interval containing no previously assessed candidate. Do not delete interior candidates and describe the remainder as consecutive.
- If no30-block exists, select the longest contiguous run containing no previously assessed candidate, with earliest start breaking ties. If none exists, conclude `no_eligible_cohort` without market calls. Do not extend dates or replace candidates after observing readiness, answers or returns.
- Freeze roster IDs and N. All subsequent rates use N, not an assumed30. Cases remain in the denominator if the model fails or the strategy does not enter.
- Broad A/B books cover the31-month population. Matched A/B/C books start empty at the selected block and include only that block, with its complete terminal exit tail. Never compare broad A with matched C or extrapolate C to unassessed history.

Authoritative archive:
`data/recovered_binance_minute_2026_09_14/btc_1m_2021_2026.parquet`, SHA256
`5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`.
Root verified2,979,360 rows and90,121,426 bytes. Coverage inventory spans January2021–August2026,68 complete contiguous months. `recovery.json` records Binance Vision USD-M BTCUSDT futures monthly URLs and zip hashes; original authenticated download receipts remain unavailable.

Do not substitute the15,910-event minute-family inventory for a native LC count. Cached derivatives, macro transforms, CME contracts and hourly feature stores are not interchangeable point-in-time inputs. Preserve disclosures for defaulted derivatives, missing macro/model/calibrator, reconstructed availability and cold starts. The study does not test a complete all-feed master trader.

### Compute budget and reproducibility

Prior Q1 collectors took11m49s–13m20s per month, measured from directory/source-file birth times. Extrapolation for28 new units is about5.9 worker-hours, before source checks, implementation and agent work; this is not a promised elapsed runtime. Benchmark the first new unit without inspecting economics.

Maximum two source workers. Ceiling: **18 summed worker-hours**, including incomplete attempts and resumed computations, not18 hours per worker or per restart. Persist start/end/elapsed counters; unknown elapsed after interruption is conservatively charged through controller recovery. Stop with `source_budget_exhausted` if census is incomplete; do not silently increase scope or budget. Identical resumptions are allowed only within remaining budget. Atomic completed artifacts are immutable; retain failed attempt logs.

Record local dependencies: ignored `results/agent_layered_entry_2026_09_11/prepare_sources.py`, sibling `one-strategy/idea_lab/{htf_pivots,structural_range}.py`, frozen parent reference, exact runtime versions and config manifests. GitHub alone is insufficient to reproduce the archive/private artifacts. Fail clearly if any dependency is absent or mismatched.

### Evidence, curriculum and calls

Each candidate uses exactly17,280 minutes in `[decision−12days, decision)`, causal parent ledgers, and completed1m/5m/15m/1h/4h/daily context through `build_lc_packet`, `build_context_request` and `PublishedContextResearchJob`. Known adverse structure is not missing evidence. Integrity failures halt; legitimate unknown context remains visible and can yield uncertainty. No replacement for an inconvenient setup.

Reuse unchanged five-record curriculum in `results/lc_context_discrimination_2026_09_15/curriculum_v1/`. Snapshot ID:
`299193b77af10d1382bc02d537a8f919b1e2cd9a3fe5557543e4bb5184fb727d`.
Its `as_of` and `training_end` are January1,2026; do not backdate them to make2025 candidates pass. Verify snapshot/brief/review hashes from the historical protocol, all record/review identities, current-case exclusions and absence of outcome-derived case memory. No memory update between cases.

At most N fresh specialists and N conditional fresh critics, requested `gpt-6-astra`/high, each history-free and restricted to its detached packet. No retries, replacement cases, answer repair, best-of-N, ablation or reuse of an unused critic allowance for another purpose. Actual model snapshot/tokens/cost remain unknown unless observed. Same-family criticism is correlated review, not institutional certification.

Freeze all specialist packets, role instructions, `{case_id,plan,request}` transport wrappers, scenarios, metric code, critic template/builder and source hashes before the first role. Inner request is the sole answer contract. Freeze each exact critic request **after** its specialist answer is captured, **before** critic invocation. Critics cannot see future prices or old reports. Preserve exact raw strings/chunks and hashes.

Maximum three simultaneously active market roles. Each role has a600-second controller deadline; an invocation consumes its slot before dispatch, including uncertain dispatch outcomes. No second invocation after controller restart. Capture late deliveries separately without changing the deadline terminal. Critic follows only an eligible, captured specialist. A valid answer completed after the15-minute entry expiry may be semantically useful but cannot enter retrospectively in measured-latency scenarios.

### Three arms and two agent estimands

| Arm | Policy | Purpose |
|---|---|---|
| A | Always native immediate entry | Signal benchmark |
| B | Always `wait_5m_high` | Generic confirmation benchmark |
| C_operational | Valid reviewed choice; otherwise controller NO_ORDER | Primary operational-policy simulation under registered timing assumptions |
| C_judgment | Same reviewed choices; unavailable plans remain null | Secondary strict judgment accounting |

`C_operational` and `C_judgment` are two readings of the same calls, not additional model arms. A coherent uncertain choice, invalid/missing response, material critic error or timeout causes a **controller-origin NO_ORDER** in operational simulation. Preserve its exact reason and the original null semantic grade. Do not relabel it a successful reviewed reject. A failure only prevents that candidate's new order; existing positions retain their frozen exits.

Implement controller NO_ORDER using a separately bound reject execution plan for the resolver, with explicit `origin=controller_fallback` in campaign metadata. Never rewrite the job grade. Missing market data, bad source integrity or unknown existing-position state remains unknown and cannot be converted into a convenient flat result. Strict full-cohort judgment PnL/contrasts/drawdown remain null where the original contracts require.

All arms: long-only,$50,000 fixed notional per admitted position, unfunded with starting equity unspecified, original source-close−2.7ATR14 stop, actual-entry2R target, decision+15min exclusive entry expiry and decision+1440min exit deadline. Wait freezes the last aligned completed5m high; requires a fully closed post-arm1m candle strictly above it and fills next open before expiry. Stop cancellation monitors from decision. No new fusion/room/parent gate or confidence cutoff.

Use existing resolver and isolated-sleeve accounting: single-position books; pending intents reserve no capacity; busy entries do not retry; capacity releases at actual known exit. Gap stops use adverse open, ambiguous barrier candles use stop-first, target gaps retain capped-target convention. These OHLC assumptions do not certify fills, funding, impact or order-book queues. Fixed notional is not equal monetary risk.

### All scenarios, no extra model decisions

| ID | Cost, round trip | A/B processing | C processing |
|---|---:|---:|---:|
| S0 primary assumed |12bps|90s|90s|
| S1 friction |24bps|90s|90s|
| S2 delay |12bps|300s|300s|
| S3 joint |24bps|300s|300s|
| S4 measured |12bps|90s|Measured decision path|
| S5 measured + friction |24bps|90s|Measured decision path|

Measured C timer begins immediately before specialist dispatch and ends after critic completion and final controller validation; includes intervening orchestration/queue delays. Record monotonic timing plus wall-clock bounds. A broken timing record invalidates measured scenarios, not an invented90s substitute. Convert elapsed time with integer ceiling seconds; resolver rounds availability to a minute. Critic is explicitly **on the execution path** in S4/S5. Prepacket feature preparation and venue routing latency are not measured; routing remains0, so even these scenarios are not a certified live-latency bound.

Use the same locked menu choice in all scenarios, preserving original stop, wait level, expiry and deadline. Derive scenario plans separately; do not mutate requests/grades. Rerun occupancy when delay changes; fee-only scenarios must preserve fills. Report all six, not just the best. S0 positivity alone is not executable benefit if S4 fails or expires. Models were not asked to reconsider choices under changed costs/timings.

### Outcome boundary, metrics and final decisions

Finish and lock a terminal for **every** roster case before any campaign economic scoring, including broad A/B. Terminal is either a recomputable published grade or a controller record with request/attempt IDs, invoked/captured state, available raw hashes, failure reason and null judgment plan. Keep incomplete jobs unchanged; no fabricated missing-role captures. Shared integrity defects stop the run. Identical saved computation may resume, never a delivered assessment.

Only then read outcome bars through the minute before each deadline plus **deadline open only**. Do not use that bar's future high/low/close. Prefix source reconstruction necessarily traverses historical prices, but does not compute/select using outcome labels; agent roles remain packet-only. This is retrospective/exposed research, not proof of clean model-training holdout.

Final report must include:

1. Exact dates,N,candidate count, exclusions, monthly resets, missing feeds, starting-equity convention and initial monetary risk per arm. Distinguish full-census from matched-block results.
2. For all six scenarios: net dollars,dollars/N,paired C_operational−A and−B, strict C_judgment availability, trades, fees, win/loss sizes, holding times and entry/nonentry reasons. Broad A/B need only the four assumed scenarios; S4/S5 A/B repeat their90s baselines in the matched table.
3. Reliability: all invoked/captured/valid/reviewed/uncertain/material-error/timeout counts, latency distribution and measured expiries. Break fallback versus intentional reject consequences out explicitly. An all-failure flat policy beating a losing benchmark does not validate agent skill.
4. Minute-close dollar MTM drawdown: cumulative gross realized + open unrealized − cumulative full round-trip costs, charging cost once at admission. Include realized observations after deadline-open exits, including the final one; do not double-charge net realized PnL. Flat intervals carry PnL without extra price reads. Missing required marks invalidate MTM. Also report worst closed loss and ambiguous bars.
5. Matched missed winners/avoided losers use **A net contribution** sign and separate each C nonentry reason. This is a descriptive opportunity comparison; capacity displacement prevents interpreting every difference as a causal individual trade effect. B comparisons are separate.
6. Temporal concentration: occupied weeks/months and connected episodes of overlapping `[decision,decision+24h]` intervals. Episodes are not guaranteed independent. Attribute every actual ledger contribution to its candidate's decision month. Leave-one-month-out is subtraction of these fixed contributions **only as a concentration diagnostic**, not reoptimized/rerun books, an independent holdout or a confidence interval.
7. Descriptive winner/loser patterns: only existing predecision hourly subtype,4h/daily lifecycle and minute-sequence facts; show denominators and missingness. No threshold fitting, discovered-rule promotion or new requests from these results.

Conclusions are separate: (a) interface/reliability worked or failed; (b) observed package better/worse/mixed than A and B; (c) assumed versus measured-latency robustness. C−B≤0 gives no observed incremental economic benefit over generic waiting. Zero C entries leaves entry quality untested. Report exact effects rather than a decorative significance result. **Durable edge remains unconfirmed**, regardless of positive sample PnL.

No trained fold-specific policy is fitted here, so chronological tables are not walk-forward training or CPCV. Purged/embargoed walk-forward/CPCV become relevant for a separately specified tuning experiment with adequate independent events; they cannot turn30 correlated historical judgments into proof. Future prospective shadow collection cannot be compressed into this historical batch.

Stop after this fixed campaign and one independent accounting review. Produce a final report even if negative, short or failed. Do not automatically tune, expand until profitable, repeat failed roles or deploy. Recommend the next action from the observed failure class; executing a new hypothesis is outside this campaign.

## 2. Implementation sequence

### Task 1: Generalized immutable monthly source census

**Create:** `scripts/research/lc_campaign_source.py`; `tests/research/test_lc_campaign_source.py`.

**Consumes:** unchanged `collect_native_lc(rows,start,end)`, `run_signal_replay`, `build_parent_ledger`, source guards and the reviewed Q1 construction logic.
**Produces:** `month_bounds(month: str) -> tuple[datetime,datetime,datetime]`; `prepare_source(month: str,out: Path) -> dict`; immutable source plus lean candidate/exposure manifest. Exact31-month allowlist is a module constant. No all-years single parent ledger: each unit stays below2048 hourly inputs.

- [ ] Add literal range/seed/allowlist tests, then run RED:

```python
def test_july_has_seed_and_exclusive_end():
    from scripts.research.lc_campaign_source import month_bounds
    start, end, seed = month_bounds('2026-07')
    assert start.isoformat() == '2026-07-01T00:00:00+00:00'
    assert end.isoformat() == '2026-08-01T00:00:00+00:00'
    assert seed.isoformat() == '2026-06-01T00:00:00+00:00'
```

- [ ] Implement new wrapper using permanent archive path, original full-file hash and exact sliced values. Retain existing TA-Lib initialization, causal-ledger clocks, manifests,17 archetypes and native pre-winner collection. Do not monkeypatch the old allowlist.
- [ ] Add injected tiny replay tests for duplicate candidates, missing minutes, unsupported month, altered source/config, atomic output collision and unchanged old module bytes. Check parity against the three saved Q1 source projections without requiring another full engine run for routine tests; any genuine collector-parity failure blocks source launch.
- [ ] Run focused tests GREEN, independently review source contract, then commit only the two new files.

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q tests/research/test_lc_campaign_source.py
```

### Task 2: Roster, budget and immutable terminal ledger

**Create:** `scripts/research/lc_campaign_contract.py`; `tests/research/test_lc_campaign_contract.py`.

**Consumes:** sorted source/exposure manifests and published jobs.
**Produces:** `select_block(candidate_ids: list[str],excluded: set[str],cap: int=30) -> list[str]` (input order already validated chronological); `CampaignLedger(directory: Path)` with `freeze(manifest)`, `start_attempt(case_id,role)`, `finish_attempt(case_id,role,result)`, `lock_terminals(terminals)` and `assert_reveal_allowed()`.

- [ ] Write RED tests for earliest full block and longest-run fallback:

```python
def test_exclusion_splits_run_instead_of_deleting_interior():
    from scripts.research.lc_campaign_contract import select_block
    assert select_block(['a','b','c','d','e'], {'c'}, 3) == ['a','b']
    assert select_block(['a','b','c','d','e'], {'a'}, 3) == ['b','c','d']
```

- [ ] Implement pure selection and atomic append-only attempt accounting. Reject duplicate invocation even after restart; consume slot before dispatch. Persist explicit not-invoked roles, timing fields, raw hashes, deadline terminals and immutable late-delivery side records.
- [ ] Add literal tests for30specialist/30critic caps, invalid-specialist critic prohibition,10min timeout, altered request/hash on restart, freeze-before-attempt, all-N-before-reveal and budget persistence. Do not fabricate grades to satisfy missing-job states.
- [ ] Run focused tests GREEN, review state/restart boundaries, commit the two new files.

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q tests/research/test_lc_campaign_contract.py
```

### Task 3: Operational fallback, scenario books and MTM report

**Create:** `scripts/research/lc_campaign_accounting.py`; `tests/research/test_lc_campaign_accounting.py`.

**Consumes:** frozen source menus, original published terminal grades, campaign failure/timing records, outcome bars and unchanged `replay_isolated_sleeves`.
**Produces:** `operational_plan(reviewed_plan: dict|None,reject_plan: dict,reason: str|None) -> dict` returning `{plan,origin,reason}`; `scenario_plan(plan: dict,cost_bps: int,processing_seconds: int) -> dict`; `mtm_curve(bars,book: dict) -> dict`; `score_campaign(bars,manifest,terminals) -> dict` with separately named broad/matched/operational/judgment results.

- [ ] Start with a literal fallback nonmutation test:

```python
def test_controller_fallback_does_not_become_agent_reject():
    from scripts.research.lc_campaign_accounting import operational_plan
    reject = {'action': 'reject'}
    result = operational_plan(None, reject, 'specialist_timeout')
    assert result == {'plan': reject, 'origin': 'controller_fallback',
                      'reason': 'specialist_timeout'}
    result['plan']['action'] = 'enter'
    assert reject == {'action': 'reject'}
```

- [ ] Implement deep-copy plan transformations; validate complete plans at book boundary. Feed old resolver/sleeves their exact schema; keep origin metadata outside their strict keys. Unknown source or price never invokes fallback substitution. Strict judgment continues to use null plans.
- [ ] Add synthetic tests using literal100entry/95stop,$50k notional: $60one-time12bps fee; adverse94gap produces−$3,060; fee-only24bps changes−$60without changing fills. Test post-arm wait, exclusive expiry, pending stop, same-open capacity release, busy no-retry, cross-month open position, stop-first ambiguity and deadline-open gap in final MTM.
- [ ] Assert measured901seconds cannot enter before15minexpiry; invalid timing nulls measured scenarios; hypothetical scenarios do not mutate original grades. Test all-failure flat operational result versus null judgment, missing marks, paired denominators, decision-month attribution and no fictitious100%-win denominator.
- [ ] Run focused tests GREEN and independently reproduce literal arithmetic; commit only the two new files.

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q tests/research/test_lc_campaign_accounting.py
```

### Task 4: One restartable controller and full source-only freeze

**Create:** `scripts/research/lc_campaign.py`; `tests/research/test_lc_campaign.py`.

**Consumes:** Tasks1–3, old packet/context builders, published jobs and existing exact-byte transport.
**Produces:** CLI phases `inventory`, `source`, `prepare`, `status`, `score`, `report`, all with `--run-dir`. Market role invocation stays in the lead controller through available collaboration tools; no invented background API, embedded credentials or hidden extra model loop.

- [ ] Write RED subprocess/state tests: score before full terminal lock fails; status never starts work; source resume verifies completed hashes; prepare selects only after all31months; report separates planned/running/completed states. Use injected tiny sources/jobs, not historical outcomes.
- [ ] Implement phases wired to previous interfaces. The exact sequence is:

```text
inventory -> source census -> roster + all request freeze -> role attempts
          -> all terminal lock -> all economic scenarios -> audit -> final report
```

- [ ] Make inventory persist dependency/exposure manifest, exact31-unit queue, old frozen file-state guards and worker-time ledger. Source phase enforces max2workers/18summed hours. Prepare freezes unchanged curriculum and all source packets before roles. Score refuses missing/mismatched shared manifests and never invokes a model.
- [ ] Reuse tested detached transport structure from the Jan19 pilot without editing/importing its campaign-specific mutable runner. Freeze new chunk reader/instructions; capture actual tool-return bytes. Add synthetic full-cycle/restart tests, including missing delivery and controller crash after dispatch.
- [ ] Run focused and complete research suite; independently review implementation and source-only freeze before any market call. A software defect is fixed and reviewed **before** first call, not patched into an active frozen market experiment.

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q tests/research/test_lc_campaign.py
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q tests/research
git diff --check
```

- [ ] Commit controller/tests and record verified counts, elapsed times and local dependency requirements. No claim that synthetic tests demonstrate trading benefit.

### Task 5: Execute once, independently audit, report and stop

**Create:** private run artifacts under the fixed namespace; public `docs/knowledge/lc_consolidated_results_2026_09_15.md`.
**Update:** `PROJECT.md`, `docs/knowledge/MEMORY.md` and checked tasks in this plan.

- [ ] Run inventory and full source census, then prepare/freeze. Report actual candidate counts,N,selected dates, exclusions and source limitations before roles. If a terminal preflight condition prevents completion, publish the finite failure report instead of picking substitute dates.
- [ ] Dispatch bounded fresh specialists/conditional critics, recording every attempt and controller timing. Follow unchanged packet-only instructions; no project-memory or outcome access for market roles. Continue fixed remaining cases after individual failures; halt on shared integrity failure.
- [ ] Lock all case terminals, authorize campaign reveal and score all registered books/scenarios once. Reopening identical saved scoring is permitted; changing decisions is not.
- [ ] Independent quant review reopens raw ledgers, reproduces fills/fees/MTM/contrasts and confirms cohort/budget/outcome boundaries. Review is research auditing, not an extra market assessment or retry. Root independently reruns deterministic report generation and compares hashes.
- [ ] Publish one plain-language answer: what agent judgment changed, its dollars and risk relative to both baselines, whether latency erased it, how often roles failed, what patterns are merely exploratory, and the single recommended next action. Include actual scope and every deviation. Do not call positive results a golden strategy or durable edge.
- [ ] Update handoff with completed work, verification, no active work if finished, next action and private artifacts needed by another CLI. Commit only public implementation/tests/docs; do not force-add data or push.

## Planning review record

Read-only source audit verified archive identity/coverage, native-source gaps, local dependencies and measured Q1 runtime. Independent quant design review approved the finite census/cohort, operational-versus-judgment distinction, latency scenarios and stopping rules; required preserving January2026 curriculum chronology rather than assigning an earlier date to undated teachings. The revised agent interval does that. Final written-plan review found no blocking omission or contradiction; its wording correction removes a deployability implication because preparation/routing time is not measured. Root verified archive identity, existing curriculum cutoffs and unchanged Jan19 locked status without revealing outcomes. Implementation and final economic results still require their own reviews.
