# Bull Machine — start here

Updated September17,2026. This is the current cross-chat/CLI handoff, not a trading
model's training context. Check git status/log because active work may be newer.

**Start with the [September17 CLI handoff](docs/knowledge/CLI_HANDOFF_2026_09_17.md).**
It supersedes historical next-step and authorization statements below. User
authorized commit/push/PR publication; no live deployment or new paid campaign.
Current next implementation: a separately versioned cheaper single-assessor
contract, followed by a frozen independent-book comparison, not another census.

## User's goal

Build an evidence-grounded master research trader that understands the trader
teachings, Wyckoff backbone, nested timeframe structure,17archetypes and fusion
intent; briefs specialists and tests whether its judgments improve entries.
The immediate approved priority is **LC-first**, using minute data as execution
context within larger structure. The separate minute equal-low research family
is not a substitute. Aim for dependable evidence of an edge; profitability is not
guaranteed or presently certified. Archetype/arm books must remain independent.

## Current approved work

- Branch: `quant/archetype-evidence-audit`; user explicitly chose continued work
  here rather than a new worktree.
- Latest completed [integration spec](docs/superpowers/specs/2026-09-15-lc-citation-integration-design.md)
  and [plan](docs/superpowers/plans/2026-09-15-lc-citation-integration.md).
  The [context comparison](docs/superpowers/specs/2026-09-15-lc-context-discrimination.md)
  and its plan commit52cd5cc remain preserved.
- Historical [v1 spec](docs/superpowers/specs/2026-09-14-lc-persistent-master.md)
  and [plan](docs/superpowers/plans/2026-09-14-lc-persistent-master.md)
  remain preserved, not silently relaxed.
- User explicitly requested planning followed by completion, with quant review.
- No live deployment, production/config/fusion edits or fine-tuning. User
  authorized checkpoint commit/push/PR on September17.
- Latest separately approved research direction:
  [LC context discrimination](docs/superpowers/specs/2026-09-15-lc-context-discrimination.md).
  Facts, request/grader/jobs, five-record memory and the fixed comparison ran.
  Independent accounting audit passed; all4agent answers failed a dual-citation
  interface, so validated agent PnL is null. A bounded prospective
  [single-catalog publication plan](docs/superpowers/plans/2026-09-15-lc-citation-publication.md)
  is implemented and independently approved, without changing or retrying the
  frozen run. The new offline decision/critic/job integration is now completed
  and independently approved. The January19 real-role validity pilot is now
  complete: one wait choice passed an independent critic; no outcomes scored.

## Historical v1 implementation state

| Item | Verified state |
|---|---|
| Earlier April code/agent comparison | Completed, independently reproduced; report below |
| LC-first source/design preflight | Completed; source-only master recommendation and independent quant review |
| New spec and four-task implementation plan | Written and committed; no new profitability claim |
| Task1 native LC source collector | Committed40f86de;702tests independently pass; independent review approved |
| Task2 persistent reviewed memory | Committed eb54a64/2cfa453; independent review approved after clock-order fix;746tests pass |
| Task3 LC judgments/restartable jobs | Committed bad3555/fb7d434; independent review approved;87focused/833research tests pass |
| Task4 actual Q1 comparison | PhaseA harness/source preflight built; root849tests pass; independent harness review and actual roles/outcomes pending |

## Latest deliverables

**September16 — subtype/evidence preflight implemented:** new outcome-free `lc_setup_preflight.py` and tests;77 combined tests pass.20 frozen original packet hashes match,20 evidence-ready;13 downside/7 upside. Remaining18 comprise11 downside/7 upside; the two already revealed are not reassessed. No model calls or live changes. [Status of all four requirements and next single-assessor milestone](docs/knowledge/lc_context_preflight_2026_09_16.md). Cheaper single-role capture, new economic comparison and live shadow service remain pending; do not claim them complete.

**September16 — live September LC comparison completed:** recovered actual server feature rows for Sep10 winner and Sep11/15 losers; one independent agent reviewed. Sep15 had bearish4H/daily context despite hourly accumulation; Sep11 was high-RSI/hourly distribution near its20h range top. Broader check21 matched groups shows distribution/high-RSI includes major winners (six groups,+$5,270.73), so no blanket veto. Dual-bearish oversold subgroup onlytwo losses: insufficient. Minute source/availability and sizing attribution remain unresolved. [Findings and proposed shadow-only context integration](docs/knowledge/lc_september_distinguishing_factors_2026_09_16.md). No production changes; candidate implementation design awaits approval.

**September16 — user-authorized three-case economic reveal completed:** no new agent calls. Fixed90s/12bps exploratory totals: immediate -$1,166.46; mechanical wait -$598.80; agent -$103.29. Jan19 accepted wait lost; Jan20 reject matched an expired mechanical trigger; Jan25 reject avoided both baseline losses. Reject-all would beat all three here, so no profitable edge established. Jan20/25 are now outcome-exposed; full20-case reveal gate remains closed and18 decisions pending. [Economic results, sensitivities and next step](docs/knowledge/lc_three_case_economics_2026_09_16.md). User favors cheaper single-assessor calls; prospective single-role mode is not yet implemented.

**September16 — TWO actual LC assessments completed:** first frozen cases January20 and January25 both chose reject; two fresh critics found no material errors; both grades are research_ready. Four role calls total, no retries or outcomes revealed. Measured full paths were 7.68 and 8.58 minutes, not the assumed90 seconds. Actual credits unavailable; no hard credit-cap claim. Remaining18 cases pending. See [results and next decision](docs/knowledge/lc_two_case_checkpoint_results_2026_09_16.md). This supersedes the older zero-paid-attempt status below. No automatic further spending.

**September16 — FULL assessment lifecycle complete; paid campaign not run:**
`lc_judgment_execution.py` now handles exact captures, conditional critic,
immutable grades/failures, restart-safe no-retry bookkeeping and all-terminal
reveal gating. Three end-to-end tests passed;14 root preparation/foundation
tests passed; independent review findings closed. Real runtime_v2 verified with
20 jobs and zero attempts. Latest observed797.76123credits <1900 start floor.
NEXT: fresh sufficient budget (or explicit revision), then run the frozen20;
no further runner feature phase. [Checkpoint](docs/knowledge/lc_runner_checkpoint_2026_09_16.md)
and [operating runbook](docs/knowledge/lc_assessment_runbook_2026_09_16.md).

**September16 — runner foundation prepared, full lifecycle still blocked:**
New `lc_judgment_runner.py` prepared20 runtime jobs and reopens with zero
specialist/reviewer attempts. It adds persistent in-process ledger ownership,
fresh start-budget checks and synthetic no-retry recovery. Real dispatch remains
disabled. Capture/critic/terminal lifecycle and spend tracking are NOT implemented.
Latest local balance record1437.24982 is below1900 launch floor; not a live
billing reading. [Exact checkpoint and next work](docs/knowledge/lc_runner_checkpoint_2026_09_16.md).

**Latest execution checkpoint —20-case evidence frozen, roles blocked:**
Versioned source preparation and20 exact causal packets/specialist requests are
saved locally under `results/lc_consolidated_2026_09_15/judgment_v1/`, cross-bound
by `preparation_binding.json`. Independent review accepted preparation integrity;
root7 new sampling/preparation tests and54 related contract tests passed (3 overlap).
No trader assessments/outcomes. NEXT: versioned20-case capture runner, persistent
in-process timing identity and interruption handling; see
[checkpoint and blockers](docs/knowledge/lc_sampling_amendment_2026_09_15.md).
Observed balance1589.73461 is below existing1900 market-start gate; recheck or
obtain explicit budget revision before paid roles. No live engine changes.

**Latest sampling update — census complete, assessments pending:** all31 source
months yielded142 LC candidates;36 lie in January–July2026 and16 are previously
assessed, leaving20. User approved chronological unassessed selection with gaps
instead of the longest consecutive block of5. See
[approved amendment](docs/knowledge/lc_sampling_amendment_2026_09_15.md).
The new selector is separate from the frozen source controller; versioned
preparation integration, packet/request freeze and paid-role readiness remain
next. No campaign market roles or outcome scoring have run. The older statuses
below are historical and are superseded by this entry.

**Newest — consolidated LC campaign implementation complete; no campaign run yet:**
[Authoritative design and implementation plan](docs/superpowers/plans/2026-09-15-lc-consolidated-campaign.md).
User capped market assessments at30setups/60calls. Fixed31-month code census:
January2024–July2026; matched agent cohort is earliest contiguous30 previously
unassessed native LC opportunities in January–July2026, or the longest eligible
run if shorter. The unchanged curriculum has a January2026 cutoff; do not
backdate it to admit2025 agent cases. No outcome-based selection or replacements.
Independent source and quant design reviews informed the plan. Root verified
the permanent2,979,360-row minute archive hash; onlyQ1 native sources currently
exist. Source work is estimated in hours, capped at18summed worker-hours/max2workers.
Six cost/latency scenarios distinguish operational NO_ORDER fallback from valid
judgment, and include measured specialist+on-path critic time. No live changes,
automatic tuning, extra pilots or claim that30cases prove an edge.
Tasks1–4 are implemented and independently reviewed through commits62a56ba..
2488787. The source census, immutable role ledger/decision timing, isolated
hourly/minute accounting and restartable controller are present. Root final
research regression:1081passed in144.26s with the existing LibreSSL warning.
No expanded source run, campaign market role or new economic outcome has begun.
NEXT: run Task5 inventory/source-only census, inspect actual eligible N and
freeze all requests, then perform the single readiness check before any paid
market role. Current execution ledger:
`.superpowers/sdd/2026-09-15-lc-consolidated-campaign/progress.md`. Do not
redispatch completed implementation or duplicate paid roles.

**Superseded prospective schedule — four-case economic protocol never executed:**
[Protocol](docs/superpowers/specs/2026-09-15-lc-economic-validation-protocol.md)
and [checkpoint](docs/knowledge/lc_validation_protocol_2026_09_15.md).
Independent quant review approved the fixed Jan20/25/29/31 development batch:
three isolated books, four cost/delay scenarios, at most4specialists/4critics,
all case terminals before outcomes, explicit null/failure treatment and dollar
MTM drawdown. Root source-only assembly verifies all4predecision prefixes and
published requests. No D1 market roles/outcomes have run. Current18Q1 candidates
are exposed and insufficient evidence for institutional edge claims. Broader
walk-forward/CPCV and prospective paper phases remain separately gated.
The consolidated plan above replaces this pending D1 implementation/run schedule;
preserve the old protocol, but do not launch its four-case batch. Do not extend
the frozen Jan19 runner in place or repeat completed roles.

**Completed — one real LC judgment survived independent review:**
[January19 validity pilot](docs/knowledge/lc_jan19_validity_2026_09_15.md).
Case LCV1: one fresh requested Astra/high specialist chose `wait_5m_high`; one
separate fresh requested Astra/high critic completed with zero material errors.
All42 captured chunks validate; terminal `research_ready` grade is locked and
recomputes on restart. This is research usability, NOT live readiness or agent
profitability. No post-decision prices or PnL were read/scored. Root114focused
tests pass (104existing +10private); independent preflight review approved.
Private runner/data/captures remain under `results/lc_jan19_validity_2026_09_15/`.
The next economic protocol has now been written/reviewed as described above;
implementation and execution remain next. Freeze cases/budget/metrics before
calls and all case terminals before outcomes. Do not
repeat this pilot or the earlier four roles. No trading threshold is validated.

**Completed — shared-catalog offline integration:**
[single-catalog integration plan](docs/superpowers/plans/2026-09-15-lc-citation-integration.md)
and [design](docs/superpowers/specs/2026-09-15-lc-citation-integration-design.md),
commits 269c2bb/54e05b3. Implementation d2f1b0e and recovery fix cfc5214 have passed
independent task and final integration review. The new role request, shared
specialist/critic catalog and immutable jobs include explicit critic-not-invoked
events. Final root verification: **1,031 distinct tests** (1,015 +16), existing
LibreSSL warning. All 798 source IDs and 16 synthetic jobs passed both initial
and post-fix checks. All 368 old frozen files still verify. No work remains
running for this integration; do not re-dispatch its implementation.
Check `.superpowers/sdd/2026-09-15-lc-citation-integration/progress.md` and live
processes before restarting. See the [integration report](docs/knowledge/lc_published_integration_2026_09_15.md).
The separately preregistered January19 01:00 UTC specialist/critic pilot above
has now completed. It is exposed-Q1 development, not a holdout. Older results
below remain unchanged; validated agent PnL is still unknown.

[September15 adapter report](docs/knowledge/lc_context_adapter_2026_09_15.md):
context request/grader/jobs committed c979261, separate reviewed5record memory,
restartable runner and actual fixed run complete. Final root verification passed
966 distinct tests: 950 public/new-private plus 16 original-private. Four requestedAstra/high
specialists read71byte-validchunks; rawchoices3wait/1reject. All4cite visible
compilerE####IDs thatthefrozengraderdoesnotaccept. Zero criticsinvoked; all4grades
locked beforeoutcomes; noanswerrepair. Finalgradeinvalid_transport reflectsthe
transparentabsent-critic sentinel, NOT failedspecialistdelivery. AgentPnL=null.

Independentarchive replay reproduced full18native+$10,157.47 (18entries) versus
alwayswait+$9,846.49 (12entries/6expiries); matched4native+$2,789.80 versus
wait+$1,392.35. All4pilotnativeoutcomespositive; noagentadvantage/edge/WFO/CPCV
claim. Postwinnerselectionbugfixedbeforefreeze tokeepall18prewinnerLCsignals.
The prospective citation publisher is complete and independently approved:
16 new tests; all 798 advertised IDs resolve and validate across four source
requests. This validates citation membership, not claim truth or trading quality.
That publication is now connected through the separately versioned offline path
described above. Its single real-role pilot has passed; next is a separately
registered economic comparison. Do not repeat sources/memory or regrade/retry
the revealed four cases.

[Earlier facts checkpoint](docs/knowledge/lc_context_checkpoint_2026_09_15.md)
remains valid. Separate three-arm contract: native immediate, generic5m-high wait,
same-menu context-judging agent. No new cutoff or old-policy mutation.

Previous deliverable: [LC setup comparison](docs/knowledge/lc_pattern_table_2026_09_14.md)
and private `results/lc_pattern_table_2026_09_14/table.json`:11 already-revealed
reconstructed LC cases across three separate studies plus37 recorded live groups.
Descriptive extraction complete; independent assembled-table review approved for
descriptive research, not trading-rule or statistical validation.
No pooled performance, new backtest, threshold optimization or newly revealed Q1
outcome. Upside expansions win in the small layered sample but lose in April;
last5m recovery also appears before a losing upper rejection. This informed the
new separate contract above. Do not treat this table as validated tuning.

Prior user-directed diagnostic: [LC pattern audit](docs/knowledge/lc_pattern_diagnostic_2026_09_14.md).
All18 fail the new same-hour child-rejection reference:13 downside expansions,
5 upside expansions, zero sweep/reclaims. The13 evidence failures mean10 broken
parents plus3 forming/absent active parents, not13 missing-data cases. Independent
read-only gate review reproduced this. Existing live LC37recorded-exit groups
(20positive/17negative) support descriptive score/outcome contrasts but lack a
verified full nested-feature/closed-position join. Proposed next: a separately
versioned subtype/lifecycle/minute-confirmation comparison, not a threshold tweak.
The new facts slice and separate executable adapter are implemented. Preserve
the original frozen hypothesis; do not describe
the all-reject role protocol as an agent entry-discrimination experiment.

### Historical v1 details — preserved, not the current resume task

Task1 implementation by `lc_source_implementer` is committed; independent review by
`lc_source_task_review` approved it. January/February source reconstructions are
complete (9/4native candidates); March completed5. All18unique IDs, complete source
minute coverage, unchanged manifests and rebuilt hourly-input hashes verified.
Task2 passed independent review
after fixing review supersession under clock rollback. Task3 belongs to
`lc_assessment_jobs_implementer` completed Task3, then independent review approved
the copied-memory chronology correction. Task4 PhaseA private harness was completed
by `lc_experiment_implementer`; `preflight_phase_a_final` has four request-only
jobs. Whole-harness review remains pending; no permanent run/outcome freeze.
One source-only `lc_curriculum_master_brief`
completed with exact captured input and valid source citations, then independent
source review approved it. Final7record snapshot is
`ab473778d356b3d16ebe0e06ed24ee20b31c18dcd3be49146d2a8a4d362d4ffd`.
Database reopen recovered it exactly; the old6record snapshot remains unchanged.
No outcome-derived lesson has been added.
Agent names are not portable across CLIs. Check current processes, git changes
and local ledger before taking over; do not run two writers or duplicate a model
experiment. The controller must update this table when verification changes it.

Continuity check: a fresh agent without this chat recovered the goal, experiment,
first incomplete task and private-data boundary from these startup documents. Its
identified stale MEMORY planning sentence was corrected. This checks handoff
usability, not automatic loading by every CLI or correctness of future agent work.

## Immediate resume instructions

1. Read this handoff, newest `docs/knowledge/MEMORY.md` overrides, then active spec
   and plan. Do not restart the entire trader audit.
2. Inspect `git status --short` and `git log -8 --oneline`.
3. Check local `.superpowers/sdd/2026-09-15-lc-citation-integration/progress.md` and
   task reports if present. These are ignored local execution detail, not portable
   truth; reconcile with commits and tests if absent.
4. Read the consolidated campaign plan above. First incomplete task is Task1:
   new immutable monthly source wrapper, not another D1 plan/pilot. Complete
   implementation/readiness review before bounded source/call execution. No
   campaign source expansion or roles have run. Preserve Jan19 and all earlier
   runs; no retries, threshold tuning or live changes. The broader source census
   is prospectively specified in the new plan, not permission to change old
   sources. Historical research remains retrospective/exposed; no WFO/CPCV or
   prospective profitability claim.
5. Update this handoff after each accepted task so another session can continue
   without relying on uncommitted plans or an agent's conversation memory.

Root fresh combined verification after PhaseA:849passes (833public plus16private),
18.93s, one existing urllib3/LibreSSL warning. This is software verification, not
strategy validation. Public-suite command below; add the private
`results/lc_persistent_master_2026_09_14/test_experiment.py` for combined scope.

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q tests/research
```

## Critical decisions not to rediscover or silently change

- Historical Q1 candidate window Jan1–Apr1 exclusive; outcome coverage ends Apr2.
  The new consolidated campaign has its own exact dates in the active plan. Monthly
  30day warm-ups reset signal source only; Q1 position books remain continuous.
- Collect native LC pre-winner longs BEFORE H2/structure filtering. The old
  `hourly_eligible` helper already filters H2 and is not the new native baseline.
- Active three books: native immediate,always wait_5m_high,agent choosing from
  the same menu. The historical v1 nested-child geometry remains a separate
  project hypothesis, not universal LC identity or a new hidden entry gate.
- Four first previously unassessed LC cases maximum; no outcome-based selection,
  retries or answer repair. Full code cohort and agent subset are different reports.
- Source,policy,memory,requests precede roles; responses precede critiques; all
  grades precede prices. Invalid/unknown is not a profitable rejection.
- Project handoff memory can mention outcomes. Trading-agent memory must exclude
  its current/future test labels. Do not load this file into market assessment.
- Persistent master means durable reviewed records and restartable research jobs
  in version1, not trained model weights or an autonomous live daemon.

## Existing results and private data

[Latest completed economic report](docs/knowledge/lc_context_adapter_2026_09_15.md)
contains the Q1 code controls and invalid-agent boundary above. Earlier
[April comparison](docs/knowledge/isolated_entry_comparison_2026_09_14.md),
commit `1acdb81`: April3hourly/242minute candidates; code references net negative;
all4 agent factual/schema checks passed but semantic reviews failed, so validated
agent PnL remains null. Raw diagnostic choices are not validated performance.
This does not test all17 or establish WFO/CPCV or live expectancy.

Permanent local Binance archive:
`data/recovered_binance_minute_2026_09_14/btc_1m_2021_2026.parquet`.
SHA256 `5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`.
Earlier `results/` artifacts and helper modules, archive, external parent-source
files under `/Users/rayghandchi/Bull Machine/one-strategy/idea_lab`, and local
execution ledgers are NOT all included in a GitHub clone. Read precise paths in
source manifests. Do not invent substitute data or claim GitHub is complete.

Unrelated untracked graph directories and the prior
`scripts/research/rebuild_entry_population.py` are to be preserved. That capture
driver has a documented persisted-JSON finalizer boundary; do not mistake it for
a standalone end-to-end clean-clone workflow.

## What to tell a new CLI

> Read AGENTS.md and PROJECT.md, then the active spec/plan and newest
> MEMORY overrides. Verify actual git/test state, report the current goal and
> first incomplete task, and continue without changing frozen/live behavior.

Same-machine sessions can read the same files. Another machine needs the correct
branch/commits plus separately transferred private artifacts. No push has been
performed for the current plan; local commits are not automatically on GitHub.
