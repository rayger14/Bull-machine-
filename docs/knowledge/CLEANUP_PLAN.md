# Bull Machine — Cleanup & Separation Plan

**Author**: quant-analyst agent (read-only analysis)
**Date**: 2026-08-11
**Branch**: `docs/cleanup-plan` (off `main` @ `73f36a5`)
**Status**: PLAN ONLY. Nothing executed, nothing merged, nothing deployed. Every mutating step below is a checklist item gated on explicit user approval.

---

## Why this document exists

The engine currently blends three different things into one undifferentiated blob, and that is why the system feels illegible:

1. **A validated strategy stack** — the champion pair `wick_trap + exhaustion_reversal` plus the 6 validated sizing boosts. This is the part that is *supposed* to make money.
2. **A data-collection instrument** ("junk book") — the other ~13 archetypes, running at full size with `bypass_threshold: true`, deliberately kept live to harvest maximum forward signal. This part is *expected to bleed* — its output is data, not PnL (user decision 2026-07-23: `project_junk_book_full_size_decision`).
3. **A research pile** — ~95 git branches, 50+ experiment JSONs, 3 near-duplicate archetype config dirs, and a 57-file knowledge folder, most of it spent-fold studies that already have a recorded verdict.

Everything reports into a single combined PnL line, so the owner cannot tell at a glance whether the *validated* book is working or whether the *instrument* is just noisy. The fix is **separation of concerns in reporting and structure** — NOT sizing, NOT disabling, NOT gating.

### Standing constraints honored throughout (do not violate)

- **Junk book stays FULL SIZE.** This plan changes REPORTING and STRUCTURE only. No sizing cuts, no disables, no filters. (`project_junk_book_full_size_decision`, 2026-07-23.)
- **`bypass_threshold` stays `true`** — deliberate data-collection mode.
- **No fusion-based filters** — Lesson #54 (fusion has negative predictive power). This plan proposes zero new gates.
- **Boosts-over-filters doctrine** — nothing here adds a filter.
- **No deploy / no merge without explicit user approval.** Approval gates are listed per phase; prior approvals never carry forward (`feedback_always_ask_before_deploy`).

---

## Phase map & effort estimate

| Phase | Theme | Executable? | Approval gate | Est. effort |
|---|---|---|---|---|
| 0 | Snapshot & safety net | yes | none (read-only tags) | 15 min |
| 1 | Two-book separation (reporting) | design + wiring later | **GATE 1** before any code change | 0.5 day design, 1 day wiring |
| 2 | Branch hygiene (archive/delete) | yes | **GATE 2** before any delete | 2–3 hrs |
| 3 | Config hygiene (archive sprawl) | yes | **GATE 3** before any move | 2–3 hrs |
| 4 | Pending-fix integration (2 PRs → deploy) | yes | **GATE 4a/4b** per merge, **GATE 4c** deploy | 1 day incl. post-deploy watch |
| 5 | Knowledge consolidation (DOCTRINE + index) | yes | **GATE 5** before restructure | 0.5 day |
| 6 | Research discipline (rule-set) | adopt | none (a rule the user agrees to) | ongoing |

**Total one-time effort: ~3–4 focused days**, spread across the gates. Phases 2, 3, 5, 6 are independent of Phase 1 and 4 and can proceed in parallel.

---

## Phase 0 — Snapshot & safety net (do this first)

**What**: Before touching anything, create restore points so every later step is trivially reversible.

**Why**: The working directory is shared by multiple concurrent agent worktrees (observed at plan time: `docs/state-of-engine`, `study/risk-neutral-deploy`, `study/wyckoff-campaign-v2`). Deletes and archives must be recoverable.

**Commands** (run from repo root, when approved):
```bash
# 0.1 Tag the current main so any archived branch is always recoverable by SHA
git tag snapshot/pre-cleanup-2026-08-11 main
git push origin snapshot/pre-cleanup-2026-08-11

# 0.2 Record the full branch + worktree inventory to a file (evidence trail)
git branch -a > /tmp/branches_pre_cleanup.txt
git worktree list > /tmp/worktrees_pre_cleanup.txt
git for-each-ref --format='%(refname:short) %(objectname:short) %(committerdate:short)' refs/heads/ > /tmp/branch_shas_pre_cleanup.txt

# 0.3 Prune stale worktree pointers (safe — only removes dead metadata, not branches)
git worktree prune -v
```

**Risk**: none — tags and inventory files only.
**Revert**: `git tag -d snapshot/pre-cleanup-2026-08-11` (local) / `git push origin :snapshot/pre-cleanup-2026-08-11` (remote).

---

## Phase 1 — TWO-BOOK SEPARATION (the core fix)

The single most important change: make "did the validated book make money?" and "how much did the instrument cost us for data?" answerable **at a glance**, without changing a single trade.

### 1.0 Book definitions (structural, not sizing)

| Book | Members | Meaning |
|---|---|---|
| **Book A — VALIDATED** | `wick_trap`, `exhaustion_reversal` | Champion regime-complementary pair (`champion_strategy_pair_2026_06_10`). All 6 validated boosts attach to `wick_trap`: `seller_flow_boost`, `bojan_wick_boost`, `breadth_boost`, exodus-refusal, `wyckoff_phase_boost`, `rotation_calm_boost`. Expected to make money. |
| **Book A — pending** | trend-continuation door | The add.48 breakout-retest door — cross-asset validated on edge (4/4 assets) but too rare to prove on history; forward-collection candidate. **Joins Book A only on explicit user approval** (not deployed today). |
| **Book B — INSTRUMENT** | the other 15 live archetypes: `confluence_breakout`, `failed_continuation`, `funding_divergence`, `fvg_continuation`, `liquidity_sweep`, `liquidity_vacuum`, `long_squeeze`, `oi_divergence`, `retest_cluster`, `spring`, `trap_within_trend`, `volume_fade_chop`, `whipsaw`, plus the two probation candidates below | Full-size, `bypass_threshold: true`, expected to bleed. Its output is **data**. |
| **Book B — probation queue** | `liquidity_compression` (holdout PF 1.14, `lc_battery_2026_07_14`), `order_block_retest` (V15 holdout PF 2.08, `v15_structure_verdicts_2026_07_17`) | The two strongest *validated-but-thin* second/third edges. They sit in Book B until the user decides to promote. **Report them as a distinct sub-tag (`B-probation`)** so their live behavior is watched separately for a promotion decision. |

> Note: membership is a *reporting label*, not a behavior switch. Every archetype keeps trading exactly as it does now, at full size.

### 1.1 What changes (three surgical touchpoints — all downstream of the engine)

The critical enabling fact from the code trace: **every trade and position already carries its originating archetype** (`TrackedPosition.archetype` at `bin/live/v11_shadow_runner.py:50`; `CompletedTrade.archetype` at `:115`; source of truth `signal.archetype_id`). So the book label is a **pure derivation** `book_of(archetype) -> 'A' | 'B' | 'B-probation'`. No new state, no engine change, no PnL-math change. `pnl_tracker.py` / `pnl_tracker_v2.py` are **legacy and unused by the live path** — ignore them.

**(a) A single source-of-truth mapping** — the book map should live in config, not code, so it is auditable and versioned:
- Add a `"books"` block to `configs/champion_paper.json` (and mirror in `configs/bull_machine_isolated_v11_fixed.json` for backtest parity): `{"A": ["wick_trap", "exhaustion_reversal"], "B_probation": ["liquidity_compression", "order_block_retest"], "B": [<the rest>]}`.
- One shared helper `book_of(archetype: str) -> str` reads this block. Default any unmapped archetype to `"B"` (fail-safe: new archetypes are instrument until validated).

**(b) Per-trade book tag in the written record** — the single writer of `trades.json`:
- **`CoinbasePaperRunner._save_trades()`** — `bin/live/coinbase_runner.py:1614-1648`. Add `"book": book_of(t.archetype)` to the `trade_dict` built at lines 1620-1647. This is behavior-preserving (adds a field).
- Optional parallel CSV tag for offline analysis: `V11ShadowRunner._append_outcome_row()` (`bin/live/v11_shadow_runner.py:431-469`) + header `_init_outcome_log` (`:344`) for `trade_outcomes.csv`.

**(c) Separate PnL lines in the aggregated summary** — the single writer of `performance_summary.json`:
- **`CoinbasePaperRunner._save_performance_summary()`** — `bin/live/coinbase_runner.py:1691-1748`. It already iterates `valid_trades` (which carry `.archetype`) to compute wins/losses/PnL at lines 1702-1709. Add a `"books": {"A": {...}, "B": {...}, "B_probation": {...}}` block to the `summary` dict (lines 1721-1742), each with `{pnl, pf, trades, win_rate}`, and keep the existing combined totals untouched. Additive only.
- Optional live-heartbeat per-book open-position PnL: `_write_heartbeat()` (`coinbase_runner.py:2781+`) using `_serialize_position()` (`:531-579`, already emits `archetype` at `:564`).

### 1.2 Dashboard: three PnL lines instead of one

The dashboard is Flask (`bin/live/dashboard.py`) + React (`dashboard/src`). Reading path: `/api/status` → `performance_summary.json` → `useStatus.ts`; `/api/trades` → `trades.json` → `useTrades.ts`.

- **Backend**: no new endpoint needed — `/api/status` (`dashboard.py:104-114`) and `/api/trades` (`:135-140`) already pass through the JSON verbatim, so the new `book` field / `books` block flow through for free once 1.1 lands.
- **Frontend, minimal**: the component `dashboard/src/components/trades/TradesByArchetype.tsx` (lines 12-32) **already groups trades by `t.archetype`** — this is the natural home for a "book" column and a book-level rollup. Add:
  1. A **book summary row/tiles** on `dashboard/src/pages/TradesPage.tsx` (renders `TradeStats` at line 16, `TradesByArchetype` at 18): three tiles — "Book A (validated) PnL", "Book B (instrument) PnL", "Combined" — read from the new `performance.books` field (`Performance` type in `dashboard/src/api/types.ts:461+`; extend it).
  2. A **book column** in `TradesByArchetype.tsx` (derive per-row via a mirror `bookOf()` in the frontend, or read the new `trade.book` field on the `Trade` type at `types.ts:549-578` once 1.1(b) is wired).
  3. Optionally split `EquityCurve.tsx` into an A-line and a B-line if the user wants time-series separation (requires per-book equity in `equity_history.csv` — a heavier lift; defer unless requested).

**The at-a-glance outcome**: the dashboard header reads e.g. *"Book A (validated): +$X, PF 1.4 · Book B (instrument): −$Y for data · Combined: net"*. That single line is the whole point of the exercise.

### 1.3 Verification before it counts as done
- Backtest parity check: run `bin/backtest_v11_standalone.py --start-date 2020-01-01 --commission-rate 0.0002 --slippage-bps 3` before and after wiring — combined PnL/PF/trade-count must be **bit-identical** (the change is reporting-only; any delta means a bug).
- Reconcile: `sum(book_A.pnl, book_B.pnl, book_B_probation.pnl) == combined.pnl` exactly.
- Confirm `book_of()` covers all 17 archetypes in `configs/champion/archetypes_v14rq/` with no `"unknown"` fallthrough in the live log.

**Risk**: low — additive, reporting-only, no engine path touched. The one real risk is a mapping typo silently bucketing an archetype into the wrong book; the reconciliation check above catches it.
**Revert**: `git revert` the wiring commit; the `"books"` config block is inert if unread.

> ⚠️ **GATE 1** — user approves the book definitions (esp. that LC + OBR sit in `B-probation`, not Book A yet) and the touchpoints BEFORE any code is written. This phase writes production code, so it stops at design until approved.

---

## Phase 2 — BRANCH HYGIENE

**State at plan time**: 95 local branches (incl. 21 ephemeral `worktree-agent-*`), ~45 remote. This is the single biggest legibility drain. Classification is by (a) merged-into-main status, (b) whether the branch holds an un-merged *deliverable* referenced by the audit ledger, and (c) the recorded verdict.

**Golden rule**: **archive before delete.** Rejected studies are tagged `archive/<name>` (so the SHA and its history survive forever, findable by `git tag -l 'archive/*'`) and only then is the branch pointer deleted. True dead-ends (merged, no unique content) are deleted directly. **Never delete a branch that is checked out in a worktree** (git refuses anyway): at plan time that protects `docs/state-of-engine`, `study/risk-neutral-deploy`, `study/wyckoff-campaign-v2`, `docs/cleanup-plan`.

### 2.1 KEEP (do nothing)

| Branch | Reason |
|---|---|
| `main` | trunk |
| `fix/boms-direction` | **pending review** — wyckoff_audit addenda 51-53; revives HTF BOMS direction; feeds Phase 4 |
| `fix/live-data-integrity` | **pending review** — `repair_package_2026_08_11`; config-honesty + wick_trap gate removal; feeds Phase 4 |
| `docs/state-of-engine` | **active** — sibling agent writing `STATE_OF_THE_ENGINE.md` |
| `docs/cleanup-plan` | this deliverable |
| `docs/quant-findings-consolidation` | holds `quant_study_master_findings` — verify merged into `docs/knowledge/`; if fully captured, downgrade to DELETE, else KEEP |
| `study/trend-continuation` | **active research asset** — add.48 best surviving edge candidate, forward-collection queued |
| `study/htf-ltf-expansion` | **active** — add.52 "promising pulse" (HTF-BOMS gates LTF entry), the first result worth escalating in a while; needs the boms fix + store rebuild to validate |
| `feat/enforce-gates-under-bypass` / `feat/eye-gate-2b` `enforce_gates_under_bypass` item | **open user decision** (repair package §5: CB `enforce_gates_under_bypass:false`) — keep until the user rules on it |

### 2.2 ARCHIVE (tag `archive/<name>`, then delete branch) — all have recorded verdicts

Command template (per branch, after **GATE 2**):
```bash
git tag archive/<name> <branch> && git push origin archive/<name> && git branch -D <branch> && git push origin :<branch>   # (remote push only if it exists on origin)
```

| Branch | Ledger ref | Verdict |
|---|---|---|
| `feat/po3-archetype`, `feat/po3-mancini-exit`, `study/po3-confirm`, `study/po3-cpcv`, `study/po3-mancini-rt`, `study/po3-orch-v2`, `study/po3-orchestration`, `study/po3-regime`, `quant/mancini-level-ladder`, `study/mancini-level-ladder` | `project_po3_fully_rejected` | PO3 killed on all 7 angles; definitively closed |
| `study/wyckoff-campaign` | wyckoff_audit add.5 | standalone REJECTED (3rd full-story failure) |
| `study/wyckoff-campaign-v2` *(currently a worktree — detach first)* | add.9 | v2 gate CLOSED — WI geometry does not rescue campaign entries |
| `feat/all-seeing-eye`, `feat/eye-gate-2b` | add.12,13 | eye tiers REJECTED on untouched fold; eye sizing-dial PARKED |
| `feat/bojan-unf-registry`, `study/bojan-real` | add.10, `resurrection_verdicts_2026_07_18` | Bojan/UNF Stage-1 gate REJECTED; trap-reset negative fwd returns |
| `study/dominance-bias` | add.14 | Dominance-HTF REJECTED |
| `study/rot-calm-boost`, `study/rot-calm-surgical` | add.18,20 | Boost 7 book-wide + 7b surgical REJECTED (rotation_calm later validated *unconstrained* and shipped via merged `feat/live-boost7-wallet` — these intermediate branches are superseded) |
| `study/poc-orthogonality`, `study/poc-structural`, `feat/live-poc-boost8` | add.21,23 | POC orthogonality finding KEPT (in knowledge); Boost 8 action REJECTED / deploy HALTED |
| `study/unified-m2`, `study/unified-archetype`, `study/unified-forward` | add.45, `unified_strategy_verdict_2026_07_13` | unified = wick_trap by proof; strict M2 architecturally inert |
| `study/xasset-spx` | add.47 | SPX spring failure INTRINSIC; superseded by KEEP `study/trend-continuation` |
| `study/short-mirror`, `feat/short-archetypes`, `chore/shorts-investigation` | champion pair §1, add.52 caveat | no functioning short side; short mirror rejected |
| `study/structural-range`, `study/htf-structural-range`, `study/range-poc` | add.15-17 | range/POC findings kept; range-POC battery-gated, nothing shipped |
| `study/t-windows`, `study/temporal-ablation` | add.19 | T-window study KNOWLEDGE-ONLY, nothing passes 3/3 |
| `study/unconstrained-boosts` | add.22 | rescue finding captured; Boost 7 shipped elsewhere, Boost 8 halted |
| `study/price-time-confluence`, `study/meta-label-prototype`, `study/idea-lab` | `resurrection_verdicts`, add.50, ML AUC 0.585 | meta-label/ML disabled; idea-lab overlay honest null |
| `quant/spring-gate-ablation`, `quant/retest-cluster-gate-ablation`, `quant/order-block-retest-gate-ablation`, `quant/failed-continuation-gate-ablation`, `quant/losers-as-anti-signals`, `quant/loser-features-plus-derivatives-heat`, `quant/composite-n-of-m-research`, `quant/concurrent-winners-conviction`, `quant/dedup-fairness-investigation`, `quant/oi-div-sizing-boost-wfo`, `quant/oi-divergence-retest-with-real-oi`, `quant/tp-strategy-research` | Lessons #54-62; `four_fixes_validation`, `gate_enforcement_audit` | gate/filter/CMI ablations — all rejected per boosts>filters doctrine; verdicts in MEMORY |
| `validate/p3-p4-bear-period` | bear-period validation | run complete; verdict captured |
| `chore/lc-winner-analysis`, `chore/recover-knowledge-files`, `chore/quant-analyst-subagent` | — | one-off chores; verify their output is on `main` first, else the output is the deliverable — capture then archive |
| `feat/dashboard-sizing-boost-badge`, `feat/dist-exhaustion-2of3-prereq`, `feat/event-calendar`, `feat/feature-store-4h-wyckoff-rebuild`, `feat/trade-outcomes-schema-fix` | mixed | superseded by later merged work; confirm no unique un-merged code before archiving |

### 2.3 DELETE (merged into main, no unique deliverable)

These are safe `git branch -d` (git only deletes if truly merged). Confirmed merged into `main` at plan time:

```bash
# Ephemeral agent worktree branches — all 21 merged, pure noise
git branch -d worktree-agent-a0a708a989725da94 worktree-agent-a15dc4d06d77a4abe \
  worktree-agent-a1b7352a8eba4b88b worktree-agent-a2b14a82e521dae16 \
  worktree-agent-a2dbe13fab00b21ac worktree-agent-a30d0b479e55f5ebd \
  worktree-agent-a33d9c4a7132fba8a worktree-agent-a57105cb7ba99178b \
  worktree-agent-a59a57043878f1c64 worktree-agent-a624102143669ea51 \
  worktree-agent-a9049a7ea72227e31 worktree-agent-a90862e9af5b33a18 \
  worktree-agent-aa5eaef71a131fa32 worktree-agent-ab307083bd7dad3a5 \
  worktree-agent-ac8771d5fc72c384c worktree-agent-acefd8e544354bb4b \
  worktree-agent-ad767be3c6c1f78f9 worktree-agent-adb22984d5837ab30 \
  worktree-agent-adbcbd014567aeb56 worktree-agent-ae5d2462eba760c15 \
  worktree-agent-af53c4bc51fffa901
git push origin :worktree-agent-a30d0b479e55f5ebd   # the one with a remote counterpart

# Merged feature/fix branches whose work is fully on main (no doc-only value)
git branch -d feat/distribution-exhaustion-wiring feat/utad-detection-improvement \
  fix/wyckoff-regime-score fix/cb-hard-gate-mode fix/dashboard-ui-labels \
  feat/tp-tier1-defaults feat/tp-tier1-defaults-validated feat/champion-battery \
  feat/cmi-crisis-prob-rebuild feat/optuna-per-archetype-objective feat/live-boost7-wallet \
  feat/live-evidence-engine feat/ml-trade-quality-pipeline feat/dist-exhaustion-3of3-boost \
  chore/quant-lessons-v2-codify quant/loser-features-as-gates quant/three-fix-followups
```

**Caveat before running 2.3**: `git branch -d` (lowercase, safe) will refuse any branch that is *not* actually merged — if it refuses, that branch belongs in ARCHIVE (2.2), not DELETE. Do not force with `-D` in this phase.

### 2.4 Remote-only stale branches (origin)
Old release/experiment branches on origin with no local counterpart (`origin/v1.2.1`, `origin/v1.4-*`, `origin/v1.5.1-*`, `origin/v1.6.*`, `origin/feature/v1.7`, `origin/feature/v1.8-hybrid`, `origin/whale-footprint-v2`, `origin/merge-v1.1.2-to-main`, `origin/sync/backup-snapshot`, `origin/codex/*`, etc.). **Recommendation**: tag `archive/legacy-<name>` for any that predate the v11 architecture, then `git push origin :<name>`. Lower priority than local hygiene; batch after 2.2/2.3.

**Risk**: losing a branch that secretly held un-merged work. Fully mitigated by Phase 0 tags + the archive-before-delete rule + `-d` (not `-D`) safety in 2.3.
**Revert**: `git checkout -b <name> archive/<name>` (archived) or `git checkout -b <name> <sha>` from `/tmp/branch_shas_pre_cleanup.txt` (deleted-but-was-merged, SHA still reachable from main history).

> ⚠️ **GATE 2** — user approves the KEEP/ARCHIVE/DELETE partition (especially the ARCHIVE list) before any `git branch -D` or `git push origin :`.

---

## Phase 3 — CONFIG HYGIENE

**State at plan time**: `configs/` holds `bull_machine_isolated_v11_fixed.json` (canonical backtest) + `champion_paper.json` (live) + **50 `configs/champion/*.json` experiment configs** + **3 near-identical archetype dirs** (`archetypes_v14rq`, `archetypes_sweep`, `archetypes_unified`) + the original `configs/archetypes/` + **10 versioned dirs** (`v141`…`v186`) + `adaptive/`, `live/`, `optimized/`, `stock/`. Several `configs/champion/*.json` are **untracked** (not even committed): `eww_wt.json`, `lc_battery.json`, `lc_eww.json`, `sweep*_wt.json`, `tc_*.json`, `unified_wt.json`, plus `configs/champion/archetypes_sweep/`.

### 3.1 The two files that are load-bearing (mark and protect)

| Role | File | Consumed by |
|---|---|---|
| **LIVE (single source of truth)** | `configs/champion_paper.json` + `configs/champion/archetypes_v14rq/` | `deploy/coinbase-paper.service` → `bin/live/coinbase_runner.py --config configs/champion_paper.json` |
| **CANONICAL BACKTEST** | `configs/bull_machine_isolated_v11_fixed.json` | `bin/backtest_v11_standalone.py` |

**Action**: add a top-of-file `"_ROLE"` marker string to each (`"LIVE — DO NOT ARCHIVE"` / `"CANONICAL BACKTEST — DO NOT ARCHIVE"`), and a `configs/README.md` stating the one-live-one-backtest rule. Everything else is an experiment and must be archived-with-date.

### 3.2 Archive the experiment sprawl

**What**: Move dead experiment configs into a dated archive tree, preserving them but out of the working namespace.
```bash
mkdir -p configs/_archive/2026-08-11/champion configs/_archive/2026-08-11/versioned
# 3.2a — the 50 champion/*.json experiment configs EXCEPT anything the live/backtest configs reference
git mv configs/champion/be_*.json configs/champion/rebase_*.json configs/champion/replay_*.json \
       configs/champion/risk_*.json configs/champion/stack_*.json configs/champion/downtrend_full.json \
       configs/_archive/2026-08-11/champion/
# 3.2b — the two SUPERSEDED archetype dirs (keep archetypes_v14rq = live)
git mv configs/champion/archetypes_sweep configs/champion/archetypes_unified configs/_archive/2026-08-11/champion/
# 3.2c — legacy versioned dirs (pre-v11 architecture)
git mv configs/v141 configs/v142 configs/v150 configs/v160 configs/v170 configs/v171 \
       configs/v185 configs/v186 configs/adaptive configs/stock configs/_archive/2026-08-11/versioned/
```
> **Before moving any `champion/*.json`**: grep the two load-bearing configs and the runner/backtester for the filename to confirm it is not referenced (`grep -rn "champion_wick_trap.json" bin/ configs/champion_paper.json configs/bull_machine_isolated_v11_fixed.json`). The `champion_*.json` per-archetype baselines are the likeliest to be referenced by battery scripts — verify each before archiving; when in doubt, keep.

### 3.3 The untracked experiment configs (git status noise)

The untracked `configs/champion/{eww_wt,lc_battery,lc_eww,sweep_wt,sweep2_wt,tc_liquidity_compression,tc_wick_trap,unified_wt}.json` and `configs/champion/archetypes_sweep/` are experiment outputs that were never committed. **Recommendation**: move them straight into `configs/_archive/2026-08-11/champion/` (or delete if they are reproducible sweep outputs). Do NOT `git add` them to `main`. This clears the persistent dirty-tree state that makes every `git status` unreadable.

### 3.4 The rule going forward (write into `configs/README.md`)
> **One live config, one canonical backtest config. Every other config is an experiment and must live under `configs/_archive/<date>/` with a one-line note of what it tested and its verdict.** New experiment configs are created *inside* the dated archive folder from day one, never in `configs/` root or `configs/champion/`.

**Risk**: archiving a config that a battery/sweep script imports by path → that script breaks. Mitigated by the grep-before-move check in 3.2 and by using `git mv` (history-preserving, trivially revertible).
**Revert**: `git mv configs/_archive/2026-08-11/... configs/...` or `git checkout main -- configs/`.

> ⚠️ **GATE 3** — user approves the archive move-list (particularly which `champion_*.json` baselines are safe) before any `git mv`.

---

## Phase 4 — PENDING-FIX INTEGRATION

Two review-ready branches must land in the right order, each behind its own approval gate, with a defined post-deploy verification. Both are honest-scoped and small.

**Dependency / ordering rationale**: `fix/boms-direction` is a pure *source-detector* fix (`engine/structure/boms_detector.py`, +12 lines) with **no live consumer today** (only `knowledge_hooks` reads BOMS, and it gates on `boms_detected`, still `False`) — so it is behavior-safe to merge first and cannot regress live trading. `fix/live-data-integrity` is *config-only* (4 files, 0 Python) but **does** change live behavior in one place (A2: un-blocks ~14% of wick_trap bars). Merge the inert one first, then the behavioral one, so any post-deploy delta is unambiguously attributable to A2.

### 4.1 Sequence

1. **Merge `fix/boms-direction` first** (inert live; enables the add.52 HTF-state research on `study/htf-ltf-expansion` after a store rebuild).
   - Pre-merge verify: `git diff main...fix/boms-direction --stat` shows only `engine/structure/boms_detector.py` (+12) and `docs/knowledge/wyckoff_audit.md` (addenda 51-53). Run the backtest — combined PnL/PF **must be bit-identical** (no archetype reads `tf*_boms_direction`).
   - **GATE 4a** → user approves merge. PR to `main`. **Do not deploy yet.**

2. **Merge `fix/live-data-integrity` second** (config honesty A1 + wick_trap gate removal A2).
   - Pre-merge verify: `git diff main...fix/live-data-integrity --name-only` = exactly `configs/champion_paper.json`, `configs/bull_machine_isolated_v11_fixed.json`, `configs/champion/archetypes_v14rq/wick_trap.yaml`, `configs/archetypes/wick_trap.yaml`. A1 is notes-only (zero runtime delta); A2 removes the never-backtested `instability_score max 0.45` live-only gate on wick_trap (restores backtest-validated behavior; boosts-over-filters compliant).
   - Backtest check: combined PnL/PF for A1 must be **identical**; A2 may change wick_trap trade count in-store only if the store carries `instability_score` (it does not — gate was inert in backtest), so backtest should also be ~identical. The real effect is live-only.
   - **GATE 4b** → user approves merge. PR to `main`. **Do not deploy yet.**

3. **Deploy** (only after both merged and the user gives an explicit, current go — `feedback_always_ask_before_deploy`).
   - Reminder: `deploy/deploy.sh` **excludes `deploy/`**, so if any `.service` file changed it must be installed **manually** (`scp` + `systemctl daemon-reload` + `restart`). Neither of these two branches changes the service file, so a normal `deploy.sh` suffices — but confirm.
   - **GATE 4c** → explicit deploy approval in the current exchange.

### 4.2 Post-deploy verification (the whole point of the boms fix + A2)

After deploy, watch `sudo journalctl -u coinbase-paper -f` and the next live feature-log sample (`results/coinbase_paper/live_features/*.jsonl`):

| Check | Expected | Why it matters |
|---|---|---|
| **BOS/CHoCH firing non-zero** | `tf1h_bos_bullish/bearish` and derived `any_bos_*` go from 0% to ~1–3% of live bars | Confirms the emitter fix propagated live (was 0% for all 533 prior bars — engine_integrity_audit #2). Note: audit #2's lower-TF BOS/CHoCH parity bug is *separate* from the add.51 BOMS-direction fix; `fix/boms-direction` addresses HTF `tf*_boms_direction`, not the 1H BOS emitter. Verify both if the store/emitter rebuild also shipped; if only `fix/boms-direction` deployed, expect `tf*_boms_direction` ~2% and 1H BOS possibly still 0 (flag if so). |
| **`tf4h_boms_direction` / `tf1d_boms_direction` non-zero** | ~2% of bars, direction-aligned to structure breaks | Directly confirms add.51 landed live |
| **wick_trap signal-rate change** | wick_trap becomes eligible on ~14% more candidate bars (the ones A2 stopped silently dropping) | Confirms A2 removed the `instability_score` gate. If wick_trap live share *doesn't* rise, A2 didn't take effect |
| **wick_trap live PnL/PF, per-archetype** | watch Book A separately (Phase 1 makes this trivial) — no regression vs pre-deploy | A2 restores backtest-validated behavior; it should not *loosen* beyond what was tested |
| **No new ERROR/CRITICAL** in journal | clean startup, all 17 archetypes load | Repair-package dry-run already confirmed 17 load, no crash; re-confirm live |

Set a **72-hour watch window** post-deploy; if BOS/BOMS stay at 0% live, the fix did not propagate (store-vs-live path divergence — the exact class of bug the integrity audit exists to catch) → roll back and investigate before proceeding.

**Risk**: A2 changes live behavior. Mitigated: it *restores* backtest-validated behavior (removes a never-validated filter), it is config-only, and Phase 1's book split makes the wick_trap effect observable in isolation.
**Revert**: `git revert` the merge, or per-file `git checkout <prev-sha> -- <file>`, then redeploy. The repair package documents exact revert lines per item.

---

## Phase 5 — KNOWLEDGE CONSOLIDATION

**State**: `docs/knowledge/` = 57 files, 42 KB `MEMORY.md`, 66 KB append-only `wyckoff_audit.md`. It is comprehensive but has no clear entry point, and the meta-lessons are scattered across 62 numbered lessons + 52 addenda.

### 5.1 Target structure

| File | Role | Action |
|---|---|---|
| `STATE_OF_THE_ENGINE.md` | current wiring / what actually fires | **being written by a sibling agent (`docs/state-of-engine`) — reference, do NOT write here** |
| `wyckoff_audit.md` | append-only research ledger (52 addenda) | **unchanged** — remains the chronological ledger of every study + verdict |
| `MEMORY.md` | lessons index + critical-lessons list | keep as the index; add a pointer to `DOCTRINE.md` at the top |
| **`DOCTRINE.md`** *(new — draft below)* | one-page meta-lessons: the durable *how-we-decide* rules | **create** (content drafted in 5.2) |
| `INDEX.md` *(new)* | a categorized table of contents for the 57 files (Validated / Rejected / Infra / Live-forensics / Doctrine) | **create** |

**Rule**: `DOCTRINE.md` holds *timeless decision rules* (rarely changes). `wyckoff_audit.md` holds *dated experiments* (append-only). `MEMORY.md` indexes both. This stops the "where do I write this?" ambiguity that grew the pile.

### 5.2 DOCTRINE.md — full draft (create verbatim, then user edits)

```markdown
# Bull Machine — Doctrine (meta-lessons)

The durable decision rules distilled from 60+ lessons and 52 research addenda.
Timeless "how we decide" principles. For dated experiments see wyckoff_audit.md;
for the lessons index see MEMORY.md. This page changes rarely.

## 1. Boosts beat filters (6/6 vs 0/10)
Sizing/exit BOOSTS that ADD size at confluence generalize; hard/soft GATES that
SUBTRACT trades do not. Empirical this era: boosts 6/6 accepted (seller_flow,
bojan_wick, breadth, exodus-refusal, wyckoff_phase, rotation_calm); filters/gates/
mode-flips/CMI-tuning 0/10. Filters fail via dedup-reshuffling (blocked bars re-
route, system PnL rises but the target archetype loses — a FALSE win), selection
asymmetry (regime gates filter the WINS), or gate-immune architecture (soft+bypass
makes hard gates inert). Default any new-gate idea to its boost equivalent.

## 2. Conjunctive beats additive
Edges live at the CONFLUENCE of several structural conditions holding together
(3-of-3 boosts), not at a weighted SUM of scores. Fusion score — the additive
blend — has NEGATIVE predictive power (r=-0.082). Structure + hard gates carry the
edge; the fusion blend does not. Never gate on a *_score.

## 3. Buy strength, not weakness
Every "buy the dip / spring at discount" thesis died cross-asset (PO3, unified M1/M2,
short mirror, spring dip-buyer — falling-knife DNA on BTC, SPX, NDX, Gold). The edge
that keeps surfacing is its OPPOSITE: buy the pullback AFTER a confirmed up-break
(trend continuation, above EMA200, self-regime-filtering — it cannot fire deep in a
bear). Resilience comes from a door that DOESN'T FIRE in bears, not from shorting them.

## 4. Rare is the nature of a real edge
The surviving edge candidates fire ~1-4x/yr/asset. Real structural edges are RARE;
frequency is the binding constraint, not signal quality. Do not inflate trade count
by loosening — a high-n "edge" is usually a regime proxy that dies OOS.

## 5. History is mined out — forward-proof only
The BTC folds are spent: strong train AUCs evaporate OOS; every lever on wick_trap
has been tested (it is a local maximum in all directions). New information comes ONLY from:
(a) live/forward data, (b) a genuinely NEW hypothesis class, or (c) a store rebuild
that fixes a detection bug (BOS/BOMS). Do NOT run new studies on spent folds to
"confirm" a prior — that is overfitting the past.

## 6. Train AND OOS must move together
Train-regression + OOS-gain is the overfit signature (the OOS window's regime
tailwind aligned with the noise the variant picked up). Reject any change where train
and OOS disagree in sign. Report per-archetype, not just system-level.

## 7. Watch-item (WI) cadence
When a signal is promising but n<30 or single-regime, it becomes a WATCH-ITEM, not a
deploy. Log it, accumulate live fires, re-adjudicate at n>=30. The live evidence
engine runs weekly; watch-items are its ledger (e.g. LC time-cut, oi_div anti-select,
funding_z live-inversion, Gold-style bear-rally breakout leak).

## 8. Validated vs Instrument (two books)
The live system runs a VALIDATED book (champion pair + boosts, meant to earn) and an
INSTRUMENT book (junk book, full size, bypass on, meant to bleed for DATA). Never
confuse the two: report their PnL separately, judge them by different yardsticks
(Book A by expectancy, Book B by data yield). Junk book stays full size by user
decision — separation is about reporting, not sizing.

## 9. Detection quality is the real bottleneck
Repeatedly, an archetype's "failure" traced to a dead/mis-scaled feature, not a bad
concept (BOMS direction discarded, BOS emitter 0% live, impossible wick gates, ST
self-loop, spring_b look-ahead). Audit whether the input FIRES before concluding the
idea is dead. Fine Wyckoff-phase labels are un-buyable, un-trainable, and absent at
crypto tops — the tractable regime tool is a COARSE CAUSAL rules-based gate, used as a
RISK filter, never a return-timer.

## 10. Config honesty is an integrity requirement
A config whose notes contradict its live values is a hazard (bypass=true while the
note says false). The engine must be legible: one live config, one backtest config,
dead keys removed, stale notes corrected. Store-vs-live feature parity is checked, not
assumed.
```

### 5.3 INDEX.md — categorized TOC (create)
A single table grouping the 57 knowledge files under: **Doctrine & State** (DOCTRINE, STATE_OF_THE_ENGINE, MEMORY, wyckoff_audit) · **Validated edges** (champion_strategy_pair, lc_battery, v15_structure_verdicts, composite_boost_wfo, distribution_exhaustion_3of3) · **Rejected studies** (resurrection_verdicts, path_conditional, risk_overlay, unified_strategy, trailing_sweep, breakeven_study, po3 refs) · **Live forensics** (live_50_forensic, live_emergent_mining, live_trade_forensic, live_evidence_engine) · **Infra/audits** (engine_integrity_audit, gate_enforcement_audit, repair_package, wyckoff_regime_bug). One line per file with its verdict.

**Risk**: none — additive docs; no code touched.
**Revert**: delete the two new files.

> ⚠️ **GATE 5** — user approves the DOCTRINE.md draft wording (it becomes the canonical decision doctrine) before it is committed as canon.

---

## Phase 6 — RESEARCH DISCIPLINE GOING FORWARD

A short rule-set to stop the sprawl from regrowing. Adopt as a standing process (add to `CLAUDE.md` under a "Research Discipline" heading).

1. **One study = one branch = one verdict.** Every study is PRE-REGISTERED (hypothesis + pass rule written to `wyckoff_audit.md` *before* measurement), runs on exactly one branch, and ends with a verdict appended to the ledger. No study without a pre-registered pass bar.
2. **Archive on completion.** The moment a study has a verdict, its branch is tagged `archive/<name>` and deleted. Deliverables (findings) live in `docs/knowledge/`, not in a lingering branch. Target: <15 live branches at any time.
3. **No new historical studies on spent BTC folds** unless it is a genuinely NEW hypothesis *class* (Doctrine #5). "Re-confirming" a prior on the same folds is banned — it is overfitting.
4. **Forward data is the default arbiter.** New edges are proven by forward paper-collection (Book B is the instrument that collects it), not by another in-sample backtest. Watch-item cadence (Doctrine #7) governs promotion.
5. **Experiment configs are born archived.** Per Phase 3.4 — created inside `configs/_archive/<date>/`, never in the working config namespace.
6. **Boost-first for any new signal.** Per Doctrine #1 — propose the boost form first; a filter must justify why the boost equivalent won't work AND report per-archetype train+OOS co-movement.

---

## Approval gates (summary)

| Gate | Blocks | What the user is approving |
|---|---|---|
| **GATE 1** | any Phase-1 code | book definitions (LC/OBR in B-probation) + the 3 touchpoints |
| **GATE 2** | any branch delete/archive | the KEEP/ARCHIVE/DELETE partition |
| **GATE 3** | any config `git mv` | the archive move-list |
| **GATE 4a** | merge `fix/boms-direction` | first merge (inert live) |
| **GATE 4b** | merge `fix/live-data-integrity` | second merge (A2 changes live) |
| **GATE 4c** | deploy | explicit, current-exchange deploy go |
| **GATE 5** | commit DOCTRINE.md as canon | doctrine wording |

Nothing in this plan is executed until its gate is cleared. Prior approvals do not carry forward.

---

## What this plan deliberately does NOT do
- Does not change any sizing (junk book stays full size).
- Does not disable any archetype or touch `bypass_threshold`.
- Does not add a single filter or fusion gate.
- Does not resolve the open `enforce_gates_under_bypass:false` CB decision (left for the user).
- Does not root-cause `liquidity_vacuum` residual signal-deadness (flagged in the repair package as a separate item).
- Does not rebuild the store to propagate the BOMS fix into offline study data (a separate, larger task that `study/htf-ltf-expansion` needs).

## Files this plan would touch (when gates clear — NOT touched now)
- Phase 1: `configs/champion_paper.json`, `configs/bull_machine_isolated_v11_fixed.json` (books block); `bin/live/coinbase_runner.py` (`_save_trades`, `_save_performance_summary`); optionally `bin/live/v11_shadow_runner.py`; `dashboard/src/pages/TradesPage.tsx`, `dashboard/src/components/trades/TradesByArchetype.tsx`, `dashboard/src/api/types.ts`.
- Phase 3: `configs/README.md` (new), `configs/_archive/2026-08-11/**` (moves).
- Phase 5: `docs/knowledge/DOCTRINE.md` (new), `docs/knowledge/INDEX.md` (new).
- Phase 6: `CLAUDE.md` (research-discipline section).

**This document itself is the only file created by this task, on branch `docs/cleanup-plan`. Production code, live config, and `main` are untouched.**
