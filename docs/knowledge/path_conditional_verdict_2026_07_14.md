# Path-Conditional TIME-CUT Verdict — LC x0.25_h24: REAL MECHANISM, NOT DEPLOYABLE (WATCH-ITEM)

**Date**: 2026-07-14. **Scope**: adversarial adjudication of the pre-registered early-weakness
time-cut grid (exit at market if MFE < X R within H hours; X∈{0.25,0.5} × H∈{12,24,48}) on
liquidity_compression and wick_trap, V14L store, thresholds enforced, real backtests.
**Artifacts**: `results/timecut_grid/` (grid), `results/timecut_grid/_adjudication/` (reruns with
trade logs + analysis scripts). **Method companions**: `scripts/champion/path_conditional_study.py`
(curves), `scripts/champion/timecut_grid.py` (grid runner).

## Executive Summary

- **Accounting is CLEAN and the reshuffle confound is ABSENT** — a first for an exit study here.
  Stats-file PFs reproduce exactly from trade-log gross P/L; all cut exits fill at
  `close × (1 − slippage)` with commission; baseline and x0.25_h24 trade populations are
  **100% identical** in all three windows; the entire PnL delta decomposes to cut-event savings.
- **But the holdout pass rests on exactly 2 cut events** (mid pass rests on **0** — the variant is
  bit-identical to baseline in 2023-24). Total policy activity: **7 cuts in 8.5 years**, all 7
  baseline losers, saving $604/event on average.
- **The curve evidence overstated the actionable population ~4x**: the "0% WR, MFE<0.25R" cells
  (n=13/8/6) mostly contain positions that had **already stopped out before H** — untouchable by
  any time-cut. Actionable (still-open) subset: train 5, mid 0, holdout 2.
- **The adjacent cell x0.25_h12 — the one the curves actually designated — failed with the exact
  trailing-sweep overfit shape**, and the trade logs show why: at 12h it cut **6 winners of 9 mid
  cuts** (forgone $7,618) and 2 of 5 holdout cuts (forgone $2,394 > saved $1,441), while cutting
  **zero winners in train**. The mechanism is horizon-fragile and regime-dependent, and train
  cannot warn.
- **EV bound does not clear**: savings/true-cut ≈ $604 vs cost/false-cut ≈ $1,233 → breakeven
  recovery probability ≈ 0.33. Observed 0/7 recoveries → 95% upper bound 0.35 > 0.33. The data
  cannot exclude an EV-neutral policy. **VERDICT: WATCH-ITEM — do not deploy.** Not an artifact
  (the mechanism is real, clean, and literature-backed), but not evidence-sufficient to become the
  system's first management edge.

## Methodology

- **Windows**: train 2018-01-01→2022-12-31, mid 2023-01-01→2024-12-31, holdout
  2025-01-01→2026-06-10 (pristine). Pre-registered grid, no extensions; acceptance = train AND
  holdout PF ≥ baseline (Rule 9 co-move).
- **Engine**: `bin/backtest_v11_standalone.py` via `_check_all_exits` patch, LC-only isolated
  config (v14rq archetype configs, `bypass_threshold=false` for threshold enforcement — the
  standard champion-battery scratch setup, production config untouched), $100K, 2bp commission,
  3bp slippage, V14L store.
- **Adjudication reruns** (`results/timecut_grid/_adjudication/rerun_logs.py`): baseline,
  x0.25_h24, x0.25_h12 × 3 windows with full trade logs. Determinism verified: reruns match the
  grid artifacts to full float precision in all compared cells.
- **Analysis** (`analyze.py`): PF recomputation from per-row gross P/L, per-position aggregation
  by `position_id`, fill-price verification against the store's close, exact set-difference of
  trade populations, per-cut-event attribution against the baseline outcome of the same position.
- Samples: LC 142/122/64 positions (train/mid/holdout); cut events 5/0/2 (h24), 11/9/5 (h12).

## Findings

### 1. Grid results (from `results/timecut_grid/liquidity_compression/`)

| cell | train PF (PnL) | mid PF | holdout PF (PnL) | pre-reg verdict | adjudicated |
|---|---|---|---|---|---|
| baseline | 1.14 ($11,714) | 1.33 | 1.14 ($2,982) | — | — |
| x0.25_h12 | 1.29 ($21,103) | 1.19 | 1.08 ($1,714) | fail | overfit shape, cut 8 OOS winners |
| **x0.25_h24** | **1.19 ($15,168)** | **1.33** | **1.19 ($4,016)** | PASS | **watch-item (2 holdout events)** |
| x0.25_h48 | 1.14 (=baseline, 0 cuts) | 1.33 (=baseline) | 1.15 ($3,212) | PASS | vacuous — tie + 1 event |
| x0.5_h12 | 1.22 | 1.23 | 1.05 | fail | overfit shape |
| x0.5_h24 | 1.17 | 1.29 | 1.08 | fail | mid+holdout down |
| x0.5_h48 | 1.14 | 1.32 | 1.15 | PASS | vacuous — near-tie |

wick_trap: all 6 cells fail (expected; its curves showed weak separation — losers tease green late).

### 2. A. Accounting — CLEAN

- Stats-file PF = trade-log recomputed PF exactly (e.g., holdout x0.25_h24: GP $25,064 /
  GL $21,048 = 1.1908, matches). Baseline reruns reproduce grid artifacts bit-for-bit.
- All 7 timecut exit rows carry `exit_reason=timecut_0.25R_24h` and fill at
  `close(exit_ts) × (1 − 3bp)` to <1e-6 relative error, with exit commission charged by
  `_close_position`. No favorable fills.
- Convention caveat: exits fill at the decision bar's close (standard for this engine); live would
  fill at next-bar open. For 7 events the close→open gap is noise, but live slippage on a
  market-out adds ~$10-20/event against the ~$604 saving.

### 3. B. Reshuffle confound — ABSENT (for the star cell)

Per window, baseline vs x0.25_h24 position populations: common = 142/122/64, only-baseline = 0,
only-variant = 0. PnL delta = 100% common-exit-delta ($3,454 / $0 / $1,034); non-cut common
positions show **zero** PnL differences. The gain is purely from cutting losers earlier — no
dedup/margin rerouting. (Caution for the family generally: at h12/mid one variant-only entry
appeared — cuts CAN free margin and admit new entries; h24's purity is a consequence of only 7
events, not a design guarantee.)

### 4. C. Mechanism — real, coherent, but RARE and thin

All 7 h24 cuts, with the baseline outcome of the same position:

| window | position (entry epoch) | cut PnL | baseline PnL | baseline exit |
|---|---|---|---|---|
| train | 1531004400 | −$390 | −$1,177 | stop_loss |
| train | 1556089200 | −$193 | −$1,454 | stop_loss |
| train | 1579255200 | −$335 | −$1,153 | 0.5R scale-out (after 24h) then stop |
| train | 1617631200 | −$1,125 | −$1,148 | stop_loss |
| train | 1631941200 | −$770 | −$1,334 | stop_loss |
| holdout | 1749502800 | −$95 | −$971 | stop_loss |
| holdout | 1750856400 | −$510 | −$668 | stop_loss |

7/7 baseline losers, 6/7 full stops. Savings: train $3,454 over 5 events ($691/event), holdout
$1,034 over 2 events ($517/event). Mid: **zero events** — the mid "co-move" in the pass criterion
is vacuous. MaxDD moves are cosmetic (train −11.1%→−10.5%, holdout −7.47%→−7.40%).

**Why the curves overstated it**: `path_conditional_study.py` computes `mfe_at{H}` as
`cum_mfe[min(H, len)−1]`, so positions that exited before H are counted at their exit-time MFE.
Splitting the "MFE<0.25R after 24h" cell by still-open-at-24h: train 13→**2** actionable, mid
8→**0**, holdout 6→**1** (offline replication; the engine, using slippage-adjusted entries and
its own bar counting, found 5/0/2). The famous 0%-WR cell is ~80% positions the stop already
killed — no exit policy can touch them.

### 5. The h12 autopsy — why this family cannot be trusted at grid resolution

| window | h12 cuts | winners cut | forgone | losers cut | saved |
|---|---|---|---|---|---|
| train | 11 | **0** | $0 | 11 | $9,389 |
| mid | 9 | **6** | $7,618 | 3 | ~$1,600 |
| holdout | 5 | **2** | $2,394 | 3 | $1,441 |

In-sample, h12 looked *strictly better* than h24 (more savings, zero winners cut). Out-of-sample it
beheaded winners of $2,011, $1,646, $1,504, $887, $750, $626... The "LC losers never get going"
signature is real in train but 2023+ winners routinely spend 12-24h below 0.25R before running.
Nothing in train distinguishes h24 from h12 except OOS outcome — which is precisely the
trailing-sweep lesson ([[trailing_sweep_verdict_2026_07_13]]): early "protection" clips OOS
winners, and the train window cannot price that risk.

### 6. D. Multiple comparisons + EV bound

- 12 trials (6 cells × 2 archetypes). Substantive passes: **1** (x0.25_h24); the two h48 "passes"
  are ties/near-ties (0-1 binding events) and count as neutral, not evidence. One-of-twelve, with
  the curve-designated neighbor (h12) failing in the overfit direction, on a +0.05 PF margin
  carried by 2 holdout events — this does not survive any reasonable trial-count haircut.
- **EV bound**: savings per true cut ≈ $604; cost per falsely-cut winner ≈ $1,233 (h12 events) →
  breakeven recovery probability p* ≈ 0.33. Observed recoveries at h24: 0/7 → 95% one-sided upper
  bound 0.35. **0.35 > 0.33**: the sample cannot exclude EV-neutral. Directional only, n far below
  the n≥30 separability bar.
- Regime stratification: all 7 events cluster in 2018/2019/2021 + June 2025; zero events in the
  2023-24 bull tape. The policy is inert exactly where LC earns most, and its false-positive risk
  (per h12) concentrates in bull tape — the regime where a 24h-recovery winner is most plausible.

## Recommendation

**WATCH-ITEM. Do NOT deploy. No config or code change proposed.** (Nothing to diff — production
untouched.)

- **Not an artifact**: accounting clean, zero reshuffle, deterministic, mechanism coherent and
  consistent with the MAE/MFE literature (Sweeney 1996; meta-labeling on post-entry features,
  López de Prado — see [[industry_study_post_entry_alpha_2026_07_14]]). The method class is
  validated; this instance is under-powered.
- **Not deployable**: 2 holdout events, 0 mid events, 1-of-12 substantive pass, EV bound not
  cleared, horizon fragility proven by the h12 autopsy, and it would be the FIRST live management
  edge — the evidentiary bar is highest exactly here.
- **Watch-item spec** (offline, zero live risk): keep `scripts/champion/timecut_grid.py` frozen as
  pre-registered. On each store extension (~6-monthly), rerun LC baseline + x0.25_h24 + x0.25_h12
  on the extended holdout and append to the cut-event ledger. **Deploy trigger**: cumulative
  actionable cut events n ≥ 15 with ≤ 1 baseline-winner among them (upper-bound recovery p < 0.25,
  safely under breakeven) AND h12 still failing (confirms the 24h boundary isn't drifting).
  At LC's observed event rate (~1-2/yr live-equivalent) this is a 2027+ decision; the offline
  ledger accrues faster than live would.
- **If it ever confirms**, deployment path: new ExitLogic rule (per-position running MFE in R,
  computed from bar highs vs entry; cut at market when `bars_held ≥ 24 and max_fav_R < 0.25`),
  LC-only, configured via the WORKING override path (user-JSON `exit_logic.liquidity_compression.*`
  — YAML exit params are dead-config, per [[trailing_sweep_verdict_2026_07_13]]), plus live-runner
  path-state persistence across restarts, plus a backtest-live parity check on the first fired cut.
  Live confirming evidence: cut positions' counterfactual (would-have-been stop) tracked in the
  outcomes log; any 2 recovered winners among the first ~6 live cuts kills it.

## Standing Orders (verbatim)

- **NEVER turn off bypass_threshold** — data collection mode is required for the foreseeable future
- **NEVER disable any archetype** — all 16 stay enabled to collect maximum live signal data
- **NEVER make production config changes** (bypass, disabled_archetypes, thresholds, archetype YAMLs) without explicit user approval
- **NEVER edit production code/configs directly** — recommendations and diffs only

(Study note: the isolated LC scratch config sets `bypass_threshold=false` inside
`configs/champion/tc_liquidity_compression.json` only — the standard battery pattern for
threshold-enforced standalone testing; the production config and live engine are untouched.)

## Sample Size & Honest Caveats

- **n=7 cut events over 8.5 years is directional only, not statistically separable** (bar is n≥30).
  The holdout pass is 2 events; the mid pass is 0 events.
- 0/7 sacrificed winners sounds perfect but bounds recovery probability only to <0.35 (95%),
  above the ~0.33 EV breakeven.
- Would it have worked in 2022 bear? Partially unknown — train includes 2018-2022 and 3 of 5 train
  cuts land in 2021; no cut fired in 2022 itself. Bear-regime behavior is effectively untested.
- Exit fills at decision-bar close; live market-outs pay spread/taker on top (~$10-20/event
  against $604/event savings — immaterial to the verdict, material to live parity).
- The per-row PF convention (Lesson #19) applies identically to both arms; position-level PF
  confirms the same deltas (holdout 1.140→1.198).

## What This Doesn't Test

- Next-bar-open fills for the cut (close-fill convention used).
- Interaction with the full 16-archetype book (LC-isolated runs only; in the full book, freed
  margin CAN admit other archetypes' entries — the h12/mid variant-only entry proves the channel
  exists, so a full-book rerun is mandatory before any deploy decision).
- Short side, other archetypes beyond wick_trap, X below 0.25R, partial (50%) cuts instead of
  full exits, or cut-then-reenter logic.
- Meta-labeling proper (sizing from post-entry features rather than a binary cut) — the natural
  Rule-8-shaped successor if the event ledger keeps confirming; a cut is the size=0 corner of that
  family.

## Files Modified

- **Written**: this doc; `results/timecut_grid/_adjudication/{rerun_logs.py,analyze.py}` +
  9 trade-log CSVs + 9 stats JSONs (scratch artifacts).
- **Production untouched**: no changes to `configs/`, `engine/`, `bin/`, or archetype YAMLs.
  Scratch config `configs/champion/tc_liquidity_compression.json` regenerated by the existing
  study tooling. Nothing committed (per instruction).

## Cross-references

[[industry_study_post_entry_alpha_2026_07_14]] (method-class validation: Sweeney MAE/MFE,
meta-labeling) · [[trailing_sweep_verdict_2026_07_13]] (the overfit shape h12 reproduced; dead
YAML exit params) · [[lc_battery_2026_07_14]] (LC's thin-but-real baseline edge) ·
[[winner_loser_forensic_2026_06_28]] (duration is the one robust discriminator — reverse-causal,
which is exactly why the actionable subset here collapsed to n=7) ·
[[unified_strategy_verdict_2026_07_13]]
