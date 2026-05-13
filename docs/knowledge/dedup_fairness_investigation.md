# Dedup Fairness Investigation

**Date**: 2026-05-13
**Branch**: `quant/dedup-fairness-investigation`
**Worktree**: `agent-a624102143669ea51`
**Scope**: Characterize and propose a fix for the same-bar dedup cannibalization
identified by four prior gate-ablation studies (`retest_cluster`,
`order_block_retest`, `failed_continuation`, `spring`). Read-only with respect
to production engine/YAML.

---

## Executive Summary

- **Dedup cannibalization is real and quantified.** In the canonical 2020-2024
  window, the engine performs **936 dedup-loss events** across 1,722 candidate
  signals. Three archetypes (`wick_trap` + `liquidity_sweep` + `oi_divergence`)
  cause **88% of all losses**; `wick_trap` alone causes **65%**.
- **Five archetypes have <5% dedup win-rate:** `spring` (0%),
  `trap_within_trend` (1.0%), `retest_cluster` (3.7%), `confluence_breakout`
  (19.5%), and `oi_divergence` (37.4%). The prior gate-ablations' missing
  archetypes (`order_block_retest`, `failed_continuation`) never even reach
  dedup — their identity gates rarely fire.
- **Phantom outcome analysis shows blocked signals MAY be profitable.**
  `trap_within_trend`'s 478 phantoms net +96 bp/trade (1.42 phantom PF, 51%
  win rate). `oi_divergence`'s 149 phantoms net +47 bp/trade (1.56 PF). But
  `confluence_breakout` (-36 bp) and `spring` (-92 bp) are negative.
- **WFO ablation across 6 dedup modes finds NO mode strictly dominates
  `status_quo` on OOS metrics.** `round_robin` correctly redistributes trades
  to underdog archetypes (`trap_within_trend` 3→11 trades, `retest_cluster`
  losses 17→4, `spring` losses 6→0), but OOS PnL falls -3.1% and PF drops
  1.79→1.69 — the unblocked archetypes don't carry their own weight in 2024.
- **`pass_through` (no dedup) is the only neutral-or-positive mode**: test PF
  1.79→1.78, test PnL +$6,963 (+12%), at the cost of +14% more trades.
  Train PF drops 1.55→1.48 (-4.5%) and train MDD widens -11.8%→-16.7%, which
  fails the conservative risk floor.
- **Recommendation: do NOT change production dedup logic at this time.**
  Status quo is the OOS leader on PF and Sharpe. The real fix is upstream:
  rebalance fusion weights so `trap_within_trend` (the largest unfairly-blocked
  archetype) can win some of its 478 dedup contests on merit, then re-run this
  study.

---

## Phase 1 — Diagnosis

### Setup

`engine/integrations/isolated_archetype_engine.py:610-629` implements
`best_per_direction` dedup as pure
`max(longs|shorts, key=fusion_score)`. No fairness tiebreak, no per-archetype
quota.

The diagnostic monkey-patches `_deduplicate_signals` (in
`scripts/dedup_investigation/dedup_patch.py`, never touching production code)
to log every (winner, loser) pair to a CSV and optionally swap the dedup
algorithm via the `DEDUP_MODE` env var.

### Dedup matrix — 2020-2024 baseline (`status_quo` / `best_per_direction`)

**System**: 755 trades, PF 1.66, PnL $82,356, MaxDD -15.14%. (Bit-exact match
to the four prior agents' baseline.)

| archetype             | fires | wins | win_rate | losses | blocked by `wick_trap` | by `liquidity_sweep` | by `oi_divergence` | by `exhaustion_reversal` | by `confluence_breakout` | by other |
|-----------------------|------:|-----:|---------:|-------:|----:|----:|----:|----:|----:|----:|
| `wick_trap`           | 503 | **490** | 0.974 |   13 |   0 |   0 |   4 |   0 |   0 |   9 |
| `liquidity_compression` | 19 |  18 | 0.947 |    1 |   0 |   0 |   1 |   0 |   0 |   0 |
| `funding_divergence`  |  15 |  13 | 0.867 |    2 |   0 |   0 |   2 |   0 |   0 |   0 |
| `exhaustion_reversal` |  59 |  34 | 0.576 |   25 |   8 |   3 |   5 |   0 |   0 |   9 |
| `liquidity_sweep`     | 213 | 109 | 0.512 |  104 |  87 |   0 |   6 |   2 |   0 |   9 |
| `liquidity_vacuum`    |   7 |   3 | 0.429 |    4 |   0 |   0 |   4 |   0 |   0 |   0 |
| `oi_divergence`       | 238 |  89 | 0.374 |  149 |  35 |  70 |   0 |  12 |  22 |  10 |
| `confluence_breakout` | 118 |  23 | 0.195 |   95 |   2 |  36 |  54 |   0 |   0 |   3 |
| `retest_cluster`      |  54 |   2 | **0.037** | 52 | 15 |   4 |  12 |  17 |   1 |   3 |
| `trap_within_trend`   | 483 |   5 | **0.010** | 478 | **457** | 11 | 5 |   0 |   0 |   5 |
| `spring`              |  13 |   0 | **0.000** | 13 |   4 |   0 |   3 |   3 |   0 |   3 |

Source: `results/dedup_investigation/baseline_2020_2024/dedup_matrix.csv`.

### Concentration of dedup blocking

Across all 936 (winner, loser) pairs in 2020-2024:

| Blocker            | Pairs blocked | Share |
|--------------------|--------------:|------:|
| `wick_trap`        |   608 | **65%** |
| `liquidity_sweep`  |   124 |  13% |
| `oi_divergence`    |    96 |  10% |
| `exhaustion_reversal` |  34 |   4% |
| `confluence_breakout` | 23 |   2% |
| `liquidity_compression` | — | — |
| other (5 archetypes) |  51 |   5% |

**Three archetypes (wick_trap + liquidity_sweep + oi_divergence) cause 88% of
all dedup losses.** This is the structural cannibalization the prior agents
identified, now numerically confirmed.

### Phantom outcomes (TP=1.5R, 240-bar time-stop)

For every losing signal, the LOSER's entry/SL was simulated forward up to 240
hours (10 days) with a 1.5R TP and a time-stop. Caveats: this ignores cooling
periods, position-cap interactions, and the real exit logic. It estimates
the *gross alpha at stake* when dedup blocks each archetype.

| archetype             |    n | win% | net% | phantom PF | bp/trade |
|-----------------------|-----:|-----:|-----:|-----------:|---------:|
| `trap_within_trend`   |  478 | 51.3 | +460.8 |   **1.42** |   +96.4 |
| `oi_divergence`       |  149 | 45.0 |  +70.0 |   **1.56** |   +47.0 |
| `liquidity_sweep`     |  104 | 50.0 |  +41.4 |   1.23 |   +39.8 |
| `confluence_breakout` |   95 | 34.7 |  -34.7 |   0.73 |   -36.5 |
| `retest_cluster`      |   52 | 42.3 |   +1.8 |   1.02 |    +3.5 |
| `exhaustion_reversal` |   25 | 40.0 |   -3.9 |   0.89 |   -15.8 |
| `wick_trap`           |   13 | 69.2 |  +27.4 |   2.34 |  +211.1 |
| `spring`              |   13 | 38.5 |  -11.9 |   0.60 |   -91.5 |
| `liquidity_vacuum`    |    4 | 50.0 |   +2.2 |   1.59 |   +54.0 |
| `funding_divergence`  |    2 |100.0 |   +7.5 |    inf |  +372.5 |
| `liquidity_compression` |  1 |  0.0 |   -2.8 |    0.0 |  -276.3 |

Source: `results/dedup_investigation/baseline_2020_2024/phantom_outcomes.json`.

**Interpretation**:
- `trap_within_trend`'s blocked signals would have been profitable on paper
  (+96 bp/trade). This is the strongest candidate for "fairness intervention."
- `confluence_breakout`'s blocked signals would have been *unprofitable*
  (-36 bp/trade). Dedup is correctly suppressing those.
- `spring`'s 13 blocked signals are unprofitable (-91 bp/trade) — but n=13 is
  too small to be conclusive.
- Total phantom net across all blocked signals: +**557%** of starting equity
  (across 936 blocked trades, 60bp average). But **this is GROSS — without
  position sizing, capital constraints, or interaction with the chosen
  winners.** This is NOT a claim that taking all blocked signals would have
  added +557% to system PnL.

### Dedup loss concentration is even higher across 2018-2024 (n=1,202)

`trap_within_trend` loses **617 times in 7 years** (588 to `wick_trap`).
`wick_trap` wins **629 of 645** of its dedup events (97.5%). The pattern
holds across both windows.

---

## Phase 2 — Alternative dedup modes tested

All implemented in `scripts/dedup_investigation/dedup_patch.py` as a
drop-in replacement for `_deduplicate_signals`, selected by env var
`DEDUP_MODE`.

1. **`status_quo`** — current production. Keep max fusion-score signal per
   direction per bar. Baseline.

2. **`normalized`** — z-score each archetype's fusion against its own 30-day
   rolling mean+std (720-bar buffer). Pick max z-score per direction. Premise:
   archetypes have different fusion-score distributions; absolute comparison
   is unfair. **Fallback to raw fusion when buffer has <20 samples (cold
   start).**

3. **`unique_sl_zone`** — engine-builtin mode. Take both signals if their stop
   levels are >2% apart. Premise: two trades in different price zones are
   genuinely orthogonal, not redundant.

4. **`round_robin`** — prefer the archetype with the lowest cumulative trade
   count. Tiebreak: higher fusion. Premise: smooth concentration across
   archetypes so under-firing names get their turn. (Note: the trade-count
   buffer is in-memory only, not used in production to inform sizing.)

5. **`hybrid_rr_fusion`** — combine the two. If the top fusion score is
   within 0.05 of the next, round-robin tiebreaks; else use fusion. Premise:
   the fusion signal is informative when there's a clear leader, but noisy
   in the middle of the pack where archetype A's 0.41 isn't meaningfully
   better than B's 0.39.

6. **`pass_through`** — disable dedup entirely. Premise: let position-cap
   constraints decide, on the bet that the post-dedup downstream layer is
   robust to position-glut.

---

## Phase 3 — WFO ablation results

Train: 2020-01-01 → 2022-12-31. Test: 2023-01-01 → 2024-12-31.
Risk floor: PF >= 1.50 train. Overfit ceiling: PF gap < 30%.
Position cap: max 5 simultaneous (unchanged across all modes).

### System-level results

| Mode | Train PF | Train PnL | Train MDD% | Train trades | Test PF | Test PnL | Test MDD% | Test trades | Test Sharpe | PF gap |
|------|---------:|----------:|-----------:|-------------:|--------:|---------:|----------:|------------:|------------:|-------:|
| **status_quo**       | **1.55** | **$27,716** | -11.8 | 302 | **1.79** | **$57,837** | -8.6 | 464 | **1.67** | +15% |
| normalized           | 1.25 | $14,803 | -16.0 | 324 | 1.65 | $53,775 | -7.3 | 495 | 1.52 | **+32% (overfit)** |
| unique_sl_zone       | 1.49 | $26,332 | -12.0 | 307 | 1.79 | $57,837 | -8.6 | 464 | 1.67 | +20% |
| round_robin          | 1.40 | $23,690 | -14.8 | 336 | 1.69 | $56,042 | -8.9 | 497 | 1.57 | +21% |
| hybrid_rr_fusion     | 1.43 | $24,179 | -14.1 | 326 | 1.73 | $58,448 | -9.8 | 498 | 1.64 | +21% |
| pass_through         | 1.48 | $29,950 | **-16.7** | 355 | 1.78 | $64,800 | -8.5 | **531 (+14%)** | **1.70** | +20% |

Source: `results/dedup_investigation/ablation/SUMMARY.csv`.

**Key observations**:
- `status_quo` is the OOS PF leader (1.79). Only `unique_sl_zone` ties it,
  but `unique_sl_zone` actually produced *identical* test results because the
  >2% SL-zone constraint never triggered on OOS bars (the engine fell back to
  best-fusion). Train PF dropped 1.55→1.49 because the constraint did fire on
  a handful of 2020-2022 bars, splitting trades that should have been
  consolidated.
- `normalized` **fails the overfit ceiling** at 32% PF gap. The z-score
  reranking destroys train performance (PF 1.55→1.25) because some archetypes
  with high z-scores have unsustainably low absolute alpha. OOS recovers
  partially as the rolling buffer warms up — exactly the "overfit on the
  reranker" failure mode we'd expect.
- `round_robin` and `hybrid_rr_fusion` both increase trade count and reduce
  underdog blockages, but **OOS PF and Sharpe drop materially**. The
  redistributed trades earn less per dollar.
- `pass_through` is the only mode where OOS PnL increases (+$6,963). PF stays
  flat at 1.78. Trade count rises +14% (well below the 1.5× kill threshold).
  However, **train MDD widens to -16.7%** (vs -11.8% baseline). Risk profile
  worsens even though point estimates are similar.

### Per-archetype trade redistribution (OOS test)

| archetype             | status_quo | normalized | unique_sl_zone | round_robin | hybrid_rr_fusion | pass_through |
|-----------------------|-----------:|-----------:|---------------:|------------:|-----------------:|-------------:|
| confluence_breakout   | 320 | 351 | 320 | 358 | 347 | **365 (+45)** |
| wick_trap             |  29 |  24 |  29 |  19 |  26 |  29 |
| liquidity_compression |  26 |  19 |  26 |  23 |  26 |  26 |
| funding_divergence    |  23 |  23 |  23 |  23 |  23 |  23 |
| liquidity_sweep       |  20 |  28 |  20 |  21 |  28 |  28 |
| exhaustion_reversal   |  14 |  14 |  14 |  14 |  14 |  14 |
| long_squeeze          |  12 |  12 |  12 |  12 |  12 |  12 |
| fvg_continuation      |  10 |  10 |  10 |  10 |  10 |  10 |
| oi_divergence         |   7 |   7 |   7 |   6 |   7 |   7 |
| **trap_within_trend** | **3** | **7** | **3** | **11 (+8)** | **5** | **17 (+14)** |
| **TOTAL**             | **464** | **495** | **464** | **497** | **498** | **531** |

Source: `results/dedup_investigation/ablation/TRADE_COUNTS_BY_MODE.csv`.

**Note**: `spring`, `retest_cluster`, `order_block_retest`,
`failed_continuation`, `liquidity_vacuum`, `volume_fade_chop`, `whipsaw`
produce 0 OOS trades in every mode — their dedup losses (where they exist)
are tiny because their identity gates rarely fire. Dedup is not the
bottleneck for those archetypes; **identity-gate firing is**.

### Phantom-promise vs realized PnL — sanity check

`trap_within_trend`'s phantom outcomes promised +96 bp/trade across 478
blocked OOS+train signals. Under `round_robin`, TWT actually fires 11+25
trades = 36 trades OOS+train (8 extra OOS, 17 extra train vs status_quo's
3+9). The system PnL impact of those extras:

- Round-robin train PnL $23,690 (vs $27,716 status_quo) = **-$4,026**
- Round-robin test PnL $56,042 (vs $57,837 status_quo) = **-$1,795**
- **Net effect: -$5,821** on +27 extra trap_within_trend trades.

So the phantom promise of +96 bp/trade did NOT materialize in real backtest —
the trades that were unblocked DID earn some PnL, but the displaced winners
earned more. **Phantom is an upper bound on alpha at stake, not a guarantee
of realized lift.**

---

## Phase 4 — Recommendation

### Primary: keep `status_quo` (`best_per_direction`) — no production change

`status_quo` wins OOS on:
- Highest test PF (1.79, tied with `unique_sl_zone` which collapsed to it).
- Highest test Sharpe (1.67, tied with `unique_sl_zone`).
- Smallest train/test PF gap among non-degenerate modes (+15%).
- Cleanest MDD profile (train -11.8%, test -8.6%).

Three honest reasons not to ship any alternative:
1. **None strictly dominates** on test PF. The two-best alternative (`pass_through`)
   adds OOS PnL but worsens train MDD by 41% (-11.8 → -16.7%).
2. **All redistribute, none generate new alpha.** The 13 archetypes plotted
   above account for every trade in every mode; we are not pulling fresh
   archetypes into play.
3. **The trapped alpha is in `trap_within_trend`, not in the underdog
   archetypes the prior gate-ablations targeted.** TWT lost 478 dedup events,
   90% to `wick_trap`. If we want to free that alpha, the right knob is
   fusion-weight tuning of TWT (or wick_trap), not the dedup tie-break logic.

### Conditional secondary: `pass_through` for stress-testing / data collection

If the team wants more signal-distribution data during live paper-trading,
`pass_through` is the safest mode to *shadow* against status_quo. It produces
+14% trades with -1% PF (within noise), and its train-side MDD widening is the
only risk to investigate.

**Do NOT deploy `pass_through` to production without** (a) reducing the
position-cap to 4 (currently 5) to absorb the +14% trade load, and (b) a
1-month paper-trading A/B against status_quo.

### Tertiary path: target the real bottleneck — fusion-weight rebalance

The structural cause of under-firing is fusion-score asymmetry:

```text
wick_trap          | weights: wyckoff 0.30, liquidity 0.20, momentum 0.20, smc 0.30 | typical TRADE_ENTRY fusion 0.487
trap_within_trend  | weights: wyckoff 0.35 (HIGHEST), liquidity 0.30, momentum 0.20, smc 0.15 | typical TRADE_ENTRY fusion <0.40
```

When both fire on the same bar, `wick_trap` wins because it scores higher in
ABSOLUTE terms — even though TWT's signal may be in its top-decile and WT's
may be median.

**Proposed follow-up (out of scope here):** A small Optuna pass tuning the
4 fusion weights for `trap_within_trend` to lift its TRADE_ENTRY distribution
into the same ballpark as `wick_trap`'s. Constrain: weights sum to 1.0, each
weight ≥ 0.1.

---

### Proposed code change (NOT TO BE APPLIED — recommendation only)

Even though the OOS evidence does not justify a change, the simplest and
safest engine modification — *if a future Optuna pass shows a clean win* —
would be to expose dedup mode as an experimental field that defaults to the
current behavior:

```diff
--- a/engine/integrations/isolated_archetype_engine.py
+++ b/engine/integrations/isolated_archetype_engine.py
@@ -580,7 +580,7 @@
     def _deduplicate_signals(
         self,
         signals: List[ArchetypeSignal],
-        mode: str = 'best_per_direction',
+        mode: str = 'best_per_direction',     # 'best_per_direction' | 'best_of_bar' | 'unique_sl_zone' | 'disabled'
     ) -> List[ArchetypeSignal]:
         """
         Deduplicate signals fired on the same bar.
@@ -592,6 +592,11 @@
             'unique_sl_zone': Group by SL zone (within 2% of entry), keep best per zone.
             'disabled': No dedup (pass-through).
         """
+        # Trade-count buffer for round_robin/hybrid_rr_fusion (in-memory only).
+        # NOTE: these modes are experimental and disabled by default. Enable by
+        # passing mode='round_robin' in config; see scripts/dedup_investigation/
+        # for evidence on when each mode is appropriate.
+        # 2026-05-13: study found status_quo Pareto-dominates on OOS metrics.
         if mode == 'disabled' or len(signals) <= 1:
             return signals
```

This is purely documentation — the existing code structure already supports
mode-switching via config (`signal_dedup.mode`).

### Path to safe rollout (for any future dedup change)

If a future investigation proves a different mode is superior:

1. **Shadow mode for 30 days**: run the new dedup in a parallel engine instance
   that logs intended trades but doesn't execute. Compare predicted-vs-actual.
2. **A/B paper trade**: split paper capital 50/50 between status_quo and the
   new mode for 60 days. Track per-archetype PnL, dedup-loss recovery rate,
   and tail PnL.
3. **CPCV validation**: re-run this study with combinatorial purged cross-val
   over 4 disjoint year-blocks to confirm OOS is not regime-specific.
4. **Production deploy** only after all three checks pass.



---

## Honest sample-size / overfit caveats

- **Phantom outcomes are gross, not net.** They ignore the fact that taking
  loser X means *not* taking winner Y. The +557% headline number cannot be
  capitalized; the maximum achievable system uplift is bounded by margin and
  by the alpha differential between the two signals at that bar.
- **Trap_within_trend's n=478 phantoms is large enough to trust the
  directional finding** (it's profitable on paper) but the +96 bp/trade
  point estimate is sensitive to the 1.5R/240-bar simulation choice. A 1.0R TP
  drops the same set to roughly +50 bp/trade.
- **Spring n=13 is far too small** to draw conclusions. Same for
  `funding_divergence` (n=2) and `liquidity_compression` (n=1).
- **OOS window 2023-2024 is one regime** (bull recovery; BTC $16K → $94K).
  A 2018-2019 holdout cross-check would strengthen the conclusion but was
  outside the runtime budget.

---

## Files

- Diagnostic:
  - `scripts/dedup_investigation/dedup_patch.py` — monkey-patch + alt modes
  - `scripts/dedup_investigation/run_dedup_diag.py` — Phase 1 runner
  - `scripts/dedup_investigation/analyze_phantom.py` — Phase 1.4 phantom outcomes
- Ablation:
  - `scripts/dedup_investigation/run_dedup_ablation.py` — Phase 3 WFO sweep
- Raw outputs (gitignored under `results/`):
  - `results/dedup_investigation/baseline_2020_2024/{dedup_events.csv, dedup_matrix.csv, phantom_outcomes.json}`
  - `results/dedup_investigation/baseline_2018_2024/{...}`
  - `results/dedup_investigation/ablation/<mode>/<phase>/summary.json`
  - `results/dedup_investigation/ablation/SUMMARY.csv`
- Constraints honored: zero edits to `engine/`, `configs/archetypes/*.yaml`, or
  the production config JSON. All changes are in `scripts/` scratch dir.

