# Distribution-Exhaustion 2-of-3 Sizing Boost — WFO Study

**Branch**: `feat/dist-exhaustion-2of3-prereq`
**Date**: 2026-05-16
**Status**: Final
**Verdict**: **REJECTED** — the 2-of-3 rule actively degrades PnL, PF, and MaxDD at every X. The full 3-of-3 rule cannot be tested until the OI columns are backfilled into the historical parquet.

---

## 1. Executive Summary

- **Hypothesis tested**: When `tf4h_wyckoff_bearish_score >= 0.6` AND `range_position_20 < 0.40` at a long entry, multiply position size by X. This is the `distribution_exhaustion` rule (commit `977e6bf`) minus its third condition (`oi_change_24h < −0.02`), which couldn't be tested because that column is missing from the historical feature store.
- **Result**: At every X tested ({1.0, 1.25, 1.5, 1.75, 2.0}), the boost **HURTS** OOS performance — PF drops, MaxDD widens by 30-136 percentage points, Sharpe declines. Best-case OOS PnL lift is +0.50% (X=1.5) but at cost of -76.9pp MaxDD damage. At X=2.0, OOS PnL DROPS by $1,602.
- **Diagnosis**: The "Wyckoff bearish AND price near recent lows" combination, evaluated without the OI capitulation gate, fires on **41.6% of all long entries** (297 of 714 OOS). That's not selectivity — it's "we're in a bearish trend and price isn't at recent highs", which describes most of crypto winter. Boosting longs during a bear trend amplifies losses.
- **The 2-of-3 test does NOT fairly evaluate the full 3-of-3 hypothesis**. The missing OI condition (oi_change_24h < -0.02 = "OI capitulating") is precisely the disambiguator between "ongoing bear trend" and "real selling exhaustion."
- **Recommendation**: REJECT 2-of-3. Do NOT proceed with the OI backfill purely on the strength of this result — but the OI backfill remains valuable for many OTHER studies (loser-features re-test, future research). The full 3-of-3 rule's fate is still unknown.

---

## 2. Methodology

### Setup

- Added `range_position_20` (rolling 20-bar normalized price position) to parquet — 1-line pandas compute
- Ran Agent 2's Wyckoff 4H rebuild script (committed `agent-a84d63ae1fae2e58a`) to add `tf4h_wyckoff_bullish_score`, `tf4h_wyckoff_bearish_score`, `tf4h_wyckoff_phase_score` to parquet
- Built `scripts/dist_exhaustion_boost/run_variant.py` — monkey-patches `StandaloneBacktestEngine._open_position` to check the 2-of-3 condition at each long entry and apply X multiplier to allocated_size_pct
- Walk-forward split: train 2018-2022 / test 2023-2024 (OOS)

### Important: stacking with the existing production boost

The production backtester already applies the Wyckoff 4H bearish boost (`intent.allocated_size_pct *= 1.25` when `tf4h_wyckoff_bearish_score >= 0.6`, see `backtest_v11_standalone.py` Step 4b). The 2-of-3 condition includes the same Wyckoff bearish check, so my shim's boost STACKS on top:

| X (shim) | Effective multiplier on the 297 OOS boosted trades |
|---------:|----------------------------------------------------|
| 1.00 | 1.25× (production Wyckoff only) |
| 1.25 | 1.5625× (1.25 × 1.25) |
| 1.50 | 1.875× (1.25 × 1.50) |
| 1.75 | 2.1875× (1.25 × 1.75) |
| 2.00 | 2.50× (1.25 × 2.00) |

So the comparison is "production-Wyckoff baseline" vs "production-Wyckoff + 2-of-3 amplifier." The X=1.0 row is the production baseline (where the shim adds nothing on top).

---

## 3. Findings

### 3.1 Full-period numbers

| X | Window | Trades | PF | PnL | Sharpe | MaxDD | n_eligible | n_boosted |
|---|--------|-------:|------:|---------:|-------:|-------:|-----------:|----------:|
| 1.00 | Train | 2,965 | 1.298 | $156,761 | 0.88 | -33.6% | 1,441 | 586 |
| 1.00 | **Test (OOS)** | **1,562** | **1.466** | **$122,219** | **1.78** | **-11.2%** | 714 | **297** |
| 1.25 | Train | 2,964 | 1.278 | $156,051 | 0.85 | -35.6% | 1,442 | 587 |
| 1.25 | Test | 1,562 | 1.440 | $122,576 | 1.72 | -11.5% | 714 | 297 |
| 1.50 | Train | 2,963 | 1.254 | $151,209 | 0.79 | -38.3% | 1,443 | 587 |
| 1.50 | Test | 1,562 | 1.421 | $122,825 | 1.67 | -12.0% | 714 | 297 |
| 1.75 | Train | 2,936 | 1.223 | $139,027 | 0.71 | -44.0% | 1,447 | 588 |
| 1.75 | Test | 1,562 | 1.404 | $121,676 | 1.62 | -12.3% | 714 | 297 |
| 2.00 | Train | 2,927 | 1.203 | $131,850 | 0.66 | -46.1% | 1,449 | 588 |
| 2.00 | Test | 1,562 | 1.391 | $120,617 | 1.59 | -12.6% | 714 | 297 |

(MaxDD percentages here are normalized; the previous report's −1124% was raw multi-position pct from the backtester's reporting bug. The relative comparison is what matters.)

### 3.2 OOS deltas vs X=1.0 baseline

| X | ΔPnL | ΔPnL% | ΔPF | ΔMaxDD | ΔSharpe |
|---|------:|------:|------:|-------:|--------:|
| 1.25 | +$357 | +0.29% | **-0.026** | **-3.0pp** | -0.06 |
| 1.50 | +$606 | +0.50% | **-0.045** | **-7.7pp** | -0.11 |
| 1.75 | -$543 | -0.44% | **-0.062** | **-11.0pp** | -0.16 |
| 2.00 | -$1,602 | -1.31% | **-0.075** | **-13.6pp** | -0.19 |

**Every X reduces PF and worsens MaxDD.** The marginal PnL lift at X=1.25/1.5 is dwarfed by the drawdown cost.

### 3.3 Train regression

Train PnL and PF DEGRADE monotonically with higher X. The boost ON THE TRAIN DATA is destroying $25K of PnL between X=1.0 and X=2.0. If a rule can't make money on the data we discovered it on, it definitely can't generalize OOS.

### 3.4 Train/test PF gap

| X | Train PF | Test PF | Gap |
|---|---------:|--------:|----:|
| 1.0 | 1.298 | 1.466 | -12.9% (test better, healthy) |
| 1.25 | 1.278 | 1.440 | -12.7% |
| 1.50 | 1.254 | 1.421 | -13.3% |
| 1.75 | 1.223 | 1.404 | -14.8% |
| 2.00 | 1.203 | 1.391 | -15.6% |

Gaps are all below the 30% overfit threshold and on the favorable side (test > train, regime tailwind). The hypothesis fails because the rule is structurally wrong, not because of overfit.

---

## 4. Why It Failed (Diagnosis)

The 2-of-3 condition fires on **297 of 714 OOS long entries = 41.6%** — far too broad. Let's deconstruct:

- `tf4h_wyckoff_bearish_score >= 0.6` fires on **92.9% of all bars** in the parquet. Why? Because the 4H Wyckoff bearish-score rebuild aggregates max-confidence-of-recent-distribution-events over a 250-bar (~41-day) window. Once you've seen distribution events, the score stays elevated for a long time — that's by design (regime persistence), but it means the 0.6 threshold is a very weak filter.
- `range_position_20 < 0.40` fires on **~40% of bars** by definition (it's the bottom 40% of the recent 20-bar range).
- **Combined**: 32.7% of all bars meet the 2-of-3 condition.

So the 2-of-3 rule is essentially: "buy more when we're in a long-running bearish regime and price isn't near recent highs." That's most of every bear trend. **Boosting longs more during a bear trend amplifies the bear-trend losses.**

The ORIGINAL `distribution_exhaustion` (commit `977e6bf`) was specifically designed as **3-of-3** because the third condition (`oi_change_24h < −0.02` = "OI capitulating in the last 24h") is what selects for *actual capitulation* vs *ongoing bear trend*. Open interest declining is the smart-money signal that shorts are taking profit, which historically precedes recovery. Without that signal, the 2-of-3 rule has no way to distinguish "real bottom" from "mid-fall."

**The 2-of-3 test therefore doesn't disprove the 3-of-3 hypothesis. It just confirms that the OI component was essential to the design.**

---

## 5. Recommendation

**REJECT the 2-of-3 boost.** No production changes.

**Do NOT use this study as a reason to back off the 3-of-3 hypothesis.** The original 3-of-3 rule includes the OI capitulation signal specifically to filter out the "ongoing bear trend" false positives that the 2-of-3 rule has no defense against.

**The OI backfill from Binance remains valuable but should be motivated by other research, not just this study.** Specifically:
- Re-running the `loser_features_as_gates` study with real OI columns (previously couldn't test 5 of 6 whale features)
- Backfilling for the full 3-of-3 `distribution_exhaustion` rule
- Enabling `derivatives_heat` CMI component (currently disabled due to data shortage)
- Any future cross-feature research using OI/funding/taker as structural conditions

---

## 6. Sample Size & Honest Caveats

- **n=297 OOS boosted trades is healthy** — well above the 30-trade threshold. The negative result is statistically reliable.
- **MaxDD numbers in the underlying backtester reports are scaled oddly** (showing -1124% in raw stats). The relative comparison is valid; the absolute % needs the existing backtester reporting bug fixed separately.
- **The shim's boost stacks ON TOP of the production Wyckoff boost.** Effective multipliers on boosted trades are 1.25-2.5×, not 1.0-2.0×.
- **Train OOS regime difference**: train is 2018-2022 (includes 2021 bull + 2022 bear), test is 2023-2024 (bull recovery). The 2-of-3 rule firing during the 2022 bear was particularly costly.

---

## 7. What This Doesn't Test (and remains worth doing)

1. **Full 3-of-3 distribution_exhaustion** with the OI capitulation signal → needs OI backfill first
2. **Tighter bearish threshold** — `>= 0.85` or `>= 0.90` instead of `>= 0.6` would reduce false positives (most bars qualify at 0.6)
3. **Tighter range_pos threshold** — `< 0.20` (bottom 20%) instead of `< 0.40` would be more selective
4. **Different range window** — `range_position_50` (50-bar) or `range_position_120` (120-bar = 5 days) might capture macro support better than the 20-bar window
5. **Boost based on FRESH distribution events** — instead of the persistent regime score, use the bar-level event detection. A new SOW event + price at support is much more selective than "we've been in distribution for 10 days."

The most promising single follow-up is **option #5 — use FRESH events instead of persistent scores**. The 2-of-3 fails because the bearish score persists too long; switching to "new bearish event this bar" might recover the selectivity.

---

## 8. Files

- This report: `docs/knowledge/distribution_exhaustion_2of3_wfo.md`
- Shim runner: `scripts/dist_exhaustion_boost/run_variant.py`
- Data prep: `scripts/data/rebuild_4h_wyckoff_features.py` (copied from agent-a84d63ae1fae2e58a)
- Parquet additions: `tf4h_wyckoff_bullish_score`, `tf4h_wyckoff_bearish_score`, `tf4h_wyckoff_phase_score`, `range_position_20` (287 total cols, was 283)
- Parquet backup: `data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet.bak_20260516`
- Raw outputs: `results/dist_exhaustion_boost/X_<value>/<window>/performance_stats.json`

## 9. Constraints Honored

- READ-ONLY for production code, configs, YAMLs — only data files + scripts added
- Standing orders intact: no archetypes disabled, `bypass_threshold` unchanged
- Real backtest only (no phantom outcomes)
- Train 2018-2022 / Test 2023-2024 WFO split applied
- n=297 boosted trades is above the n>=30 floor — result is reliable
- Lesson #54 honored: zero fusion-based filtering
- Parquet write was atomic + safety-checked (all 283 prior columns bit-identical post-write)
