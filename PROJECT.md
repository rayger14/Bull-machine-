# Bull Machine — start here

Updated September15,2026. This is the current cross-chat/CLI handoff, not a trading
model's training context. Check git status/log because active work may be newer.

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
- [Spec](docs/superpowers/specs/2026-09-14-lc-persistent-master.md).
- [Implementation plan](docs/superpowers/plans/2026-09-14-lc-persistent-master.md).
- Plan commits: `28d67c0`, quant clarifications `e31034e`.
- User explicitly requested planning followed by completion, with quant review.
- No live deployment, production/config/fusion edits, fine-tuning, push or PR.
- Latest separately approved research direction:
  [LC context discrimination](docs/superpowers/specs/2026-09-15-lc-context-discrimination.md).
  Only its facts-only slice is implemented; the original spec/plan below remains
  historical unfinished work, not permission to silently relax its policy.

## What is done versus not done

| Item | Verified state |
|---|---|
| Earlier April code/agent comparison | Completed, independently reproduced; report below |
| LC-first source/design preflight | Completed; source-only master recommendation and independent quant review |
| New spec and four-task implementation plan | Written and committed; no new profitability claim |
| Task1 native LC source collector | Committed40f86de;702tests independently pass; independent review approved |
| Task2 persistent reviewed memory | Committed eb54a64/2cfa453; independent review approved after clock-order fix;746tests pass |
| Task3 LC judgments/restartable jobs | Committed bad3555/fb7d434; independent review approved;87focused/833research tests pass |
| Task4 actual Q1 comparison | PhaseA harness/source preflight built; root849tests pass; independent harness review and actual roles/outcomes pending |

Latest deliverable: [September15 context checkpoint](docs/knowledge/lc_context_checkpoint_2026_09_15.md).
New `lc_context_facts.py` and34literal tests independently reviewed; root combined
883tests pass,19.61s, existingwarning. Four original source packets describe
3upside/1downside closes;4H states intact/broken-up/broken-up/broken-down. Last
case daily absent-before becomes active by decision, without backdating binding.
Separate three-arm contract specified/reviewed: native immediate, generic
5m-high wait, same-menu context-judging agent. No new adapter/jobs/role run or
outcomes yet. Next: build/test the separate request/grader/job adapter, remove
old mandatory hypothesis from new memory only, review, then freeze four roles
and critics before any new economic reveal. See exact contract; no new cutoff.

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
The new facts slice is implemented; the new executable adapter is not. Preserve
the original frozen hypothesis; do not describe
the all-reject role protocol as an agent entry-discrimination experiment.

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
3. Check local `.superpowers/sdd/2026-09-14-lc-persistent-master/progress.md` and
   task reports if present. These are ignored local execution detail, not portable
   truth; reconcile with commits and tests if absent.
4. Read the September15 context checkpoint and separate comparison contract.
   Continue with the new adapter, not a rerun or silent relaxation of old Task4.
   Preserve original sources/curriculum/freezes; new memory excludes only the
   two exactly identified v1-specific records. Do not repeat source jobs or the
   old bounded master brief. Integration review still precedes permanent
   market-role freeze, and all grades precede outcomes.
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

- Q1 candidate window Jan1–Apr1 exclusive; outcome coverage ends Apr2. Monthly
  30day warm-ups reset signal source only; Q1 position books remain continuous.
- Collect native LC pre-winner longs BEFORE H2/structure filtering. The old
  `hourly_eligible` helper already filters H2 and is not the new native baseline.
- Three books: native,explicit nested-child structure,same structure plus agent.
  New prior-hour compressed-child geometry is a project hypothesis, not universal
  LC identity or an exact trader rule. Parent-floor reclaim is only one subtype.
- Four first previously unassessed LC cases maximum; no outcome-based selection,
  retries or answer repair. Full code cohort and agent subset are different reports.
- Source,policy,memory,requests precede roles; responses precede critiques; all
  grades precede prices. Invalid/unknown is not a profitable rejection.
- Project handoff memory can mention outcomes. Trading-agent memory must exclude
  its current/future test labels. Do not load this file into market assessment.
- Persistent master means durable reviewed records and restartable research jobs
  in version1, not trained model weights or an autonomous live daemon.

## Existing results and private data

[Latest completed economic report](docs/knowledge/isolated_entry_comparison_2026_09_14.md),
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
