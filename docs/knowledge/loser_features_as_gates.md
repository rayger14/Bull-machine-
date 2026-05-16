# Loser Features as Winning Gates — Hypothesis Study
**Branch**: `quant/loser-features-as-gates`
**Date**: 2026-05-14
**Status**: HYPOTHESIS REJECTED (with one partial / sample-bounded exception)

---

## 1. Executive Summary

- **Hypothesis tested**: Features computed by losing archetypes (`oi_divergence`, `long_squeeze`) detect real market phenomena (crowded longs, funding extremes, OI divergences) and should improve OTHER archetypes' outcomes when used as confirmation / anti-gates.
- **Verdict**: REJECTED for the core "funding extreme as anti-long gate" claim. Trades with `funding_Z >= 1.5` actually have **higher** PF (2.33) than trades without (1.77) across all longs in 2020–2024. The "crowded-longs" signal is dominated by positive-funding *bull-continuation* periods in this dataset, not by impending squeezes.
- **One marginal survivor**: `liquidity_sweep` + `funding_Z >= 1.0 block` passes WFO (train dPnL +$3.2K, test dPnL +$0.2K) but **n_blocked_test = 4** — below the n<30 directional-only floor and dominated by a single 2021 outlier (-$3.2K).
- **Feature store has NO Binance whale columns**: `oi_change_4h`, `oi_change_24h`, `oi_price_divergence`, `ls_ratio_extreme`, `taker_imbalance`, `funding_oi_divergence` are **absent** from `BTC_1H_LATEST.parquet` (61,306 bars × 283 cols, 2018-2024). Only `funding_Z` and `funding_rate` are usable from the loser archetypes' feature catalog. `oi` exists but is a hardcoded constant (1e9). **The hypothesis cannot be fully tested with the current feature store.**
- **No production changes recommended.** YAML diffs withheld per zero-tolerance rules. Phase 1 catalog is the durable contribution; the cross-test matrix is documented for reproducibility.

---

## 2. Methodology

### Data
- Trade log: `results/cross_archetype/anti_signals/baseline/trade_log.csv` (755 trades, 2020-01-14 → 2024-12-26)
- Feature store: `data/features_mtf/BTC_1H_LATEST.parquet` (283 cols, 2018-2024) — joined via `merge_asof` on entry timestamp (backward, 1H tolerance). 100% coverage on `funding_Z`, `rsi_14`, `volume_zscore`, `bos_*`.
- Merged dataset persisted at `results/cross_archetype/loser_features/trades_with_features.parquet`.

### Walk-Forward Split
- Train: 2020-01 → 2022-12 (302 trades)
- Test (OOS): 2023-01 → 2024-12 (453 trades)
- Note: Train window is smaller than test window because most trades in this log are 2023-2024. Sample-size caveats applied throughout.

### Reject Criteria
- Train PF improvement and Test PF improvement both required (`dpnl > 0` on both)
- OOS n < 30 → directional only
- Trade drop > 50% on test → reject (overfit)
- PF gap (|train_pf − test_pf| / train_pf) > 30% → flag, not auto-reject (small samples inflate)

### Loser-archetype identity gates (from `engine/archetypes/logic.py` + YAML)
| Archetype | Identity gate (logic.py) | Hard gates (YAML) | Effective in feature store? |
|---|---|---|---|
| **funding_divergence** (S4) | `binance_funding_rate < -0.0001` OR `funding_Z < -1.0` | funding_Z ≤ -0.5, `funding_oi_divergence=1`, `ls_ratio_extreme ≤ -0.5`, `oi_price_divergence ≥ 0.01`, `oi_change_4h ≤ 0.05` | **Only funding_Z** usable (other 4 NaN) |
| **long_squeeze** (S5) | `binance_funding_rate > 0.0001` OR `funding_Z > +1.0` | funding_Z ≥ 0.5, `rsi_14 ≥ 60`, `ls_ratio_extreme ≥ 1.5`, `funding_oi_divergence = -1`, `vol_shock ≤ 0.10`, `accumulation_at_support = false` | **Only funding_Z + RSI** usable |
| **oi_divergence** (S9) | NONE in logic.py — pure YAML gate | `oi_change_4h ≤ -0.02`, `oi_change_24h ≤ -0.03`, `volume_zscore ≥ 0.0`, `rsi_14 ≤ 35`, `taker_imbalance ≤ 0.1`, `distribution_at_resistance = false` | **None of the OI/taker gates usable** — falls back to volume_zscore + rsi_14 only |

---

## 3. Findings

### Phase 1 — Loser-archetype feature catalog (effect size vs other-archetype baseline)

Cohen's d = (mean_loser − mean_other) / std_other. Sample sizes from the 755-trade log.

#### oi_divergence (n=10 in-log; pnl=-$4,641; wr=20%)
| Feature | mean(loser) | mean(other) | Cohen's d | Note |
|---|---:|---:|---:|---|
| volume_zscore | 4.764 | 0.984 | **+1.61** | Extreme volume spike |
| bos_bearish | 0.286 | 0.038 | **+1.29** | Bearish BOS at entry |
| rsi_14 | 40.13 | 57.20 | **−1.06** | Oversold |
| atr_pct | 0.747 | 0.548 | +0.47 | Elevated volatility |
| funding_rate | 0.0001 | 0.0000 | +0.64 | Slight positive |
| Others (funding_Z, fvg_*, pti) | ~baseline | — | <0.35 | Not differentiating |

#### long_squeeze (n=34; pnl=-$2,669; wr=53%)
| Feature | mean(loser) | mean(other) | Cohen's d | Note |
|---|---:|---:|---:|---|
| funding_rate | 0.0011 | 0.0001 | **+4.03** | Extreme positive funding |
| funding_Z | 3.35 | -0.07 | **+2.42** | Extreme z-score |
| rsi_14 | 75.92 | 56.08 | **+1.22** | Overheated |
| volume_zscore | 2.61 | 0.96 | +0.69 | Elevated |
| tf1h_pti_score | 0.13 | 0.20 | **−0.71** | Lower PTI |
| fvg_bullish | 0.32 | 0.18 | +0.37 | Mild bullish FVG |

**Key insight from catalog**: long_squeeze fires on genuinely distinct funding/RSI extremes (d > 2). oi_divergence fires on extreme volume + bearish BOS + low RSI (but the OI/taker fields it really needs are MISSING from the feature store, so it's effectively a "high-volume-low-RSI" detector in practice).

### Phase 2 — Cross-test gate matrix (TRAIN only, 2020-2022)

Selection: archetype + gate combinations where gated_n ≥ max(5, 0.4 × base_n). Effect = train_dpnl.

Top positive train effects:
| archetype | gate | base→gated n | base_pf | gated_pf | train_dpnl |
|---|---|---|---:|---:|---:|
| liquidity_sweep | funding_Z < 1.5 (block xpos) | 33→30 | 2.95 | 12.43 | **+$3,760** |
| liquidity_sweep | funding_Z < 2.0 | 33→31 | 2.95 | 8.63 | +$3,245 |
| confluence_breakout | funding_Z > -1.0 (block xneg) | 169→145 | 0.73 | 0.73 | +$754 |
| confluence_breakout | rsi_14 < 75 | 169→168 | 0.73 | 0.75 | +$585 |
| confluence_breakout | bos_bearish == 0 | 169→162 | 0.73 | 0.74 | +$607 |

Top negative train effects (CONFIRMATIONS that these features are NOT good gates):
| archetype | gate | gated_pf | train_dpnl |
|---|---|---:|---:|
| funding_divergence | volume_z < 3.0 | 1.90 | **−$4,670** |
| wick_trap | rsi_14 < 75 | 4.76 | **−$8,348** |
| funding_divergence | rsi_14 < 75 | 3.00 | −$2,279 |

Full matrix at `results/cross_archetype/loser_features/phase2_gate_matrix.parquet`.

### Phase 3 — WFO validation (TRAIN 2020-2022 + TEST 2023-2024)

Only PASS combinations (both train_dpnl > 0 AND test_dpnl > 0):

| candidate | arch | feat | thr | train_n→gated | gated_pf | train_dpnl | test_n→gated | gated_pf | test_dpnl |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | confluence_breakout | rsi_14 ≤ 42 (block) | low_blocks | 169→113 | 0.83 | **+$4,470** | 319→253 | 1.85 | **+$3,977** |
| 2 | liquidity_sweep | funding_Z ≥ 1.0 (block) | high_blocks | 33→28 | 11.97 | +$3,223 | 20→16 | 4.65 | +$203 |
| 3 | confluence_breakout | rsi_14 ≥ 66 (block) | high_blocks | 169→151 | 0.75 | +$1,931 | 319→303 | 1.64 | +$424 |

### Phase 3.B — Across-all-longs hypothesis check

The cleanest test of the user's core hypothesis ("funding extreme = avoid longs") across all 13 long archetypes pooled (711 trades):

| gate | train_dpnl | test_dpnl | PASS? |
|---|---:|---:|---|
| funding_Z ≥ 1.5 BLOCK | +$912 | **−$12,160** | FAIL |
| funding_Z ≥ 2.0 BLOCK | −$213 | −$5,302 | FAIL |
| funding_Z ≥ 2.5 BLOCK | −$213 | −$4,234 | FAIL |
| rsi ≥ 80 BLOCK | −$4,192 | −$15,558 | FAIL |
| rsi ≥ 78 BLOCK | −$5,147 | −$19,307 | FAIL |
| volume_zscore ≥ 4.0 BLOCK | **−$18,921** | −$8,843 | FAIL |
| volume_zscore ≥ 3.0 BLOCK | **−$20,792** | −$11,477 | FAIL |
| bos_bearish==1 BLOCK | **−$34,843** | +$778 | FAIL |
| AND-combo funding≥2 ∧ rsi≥75 | +$2,443 | −$1,457 | FAIL |
| AND-combo funding≥1.5 ∧ vol_z≥3 | +$984 | −$1,331 | FAIL |

**The cross-archetype pooling shows the hypothesis fails clearly.** The OOS losses are large and consistent. The "crowded-longs-imminent-squeeze" interpretation does not hold in BTC 2020-2024 data — funding extremes correlate with bull-continuation, not tops.

### Phase 4 — Regime stratification on top survivor

**funding_Z ≥ 1.5 BLOCK long** (the most-cited part of the hypothesis):

| regime | blocked n | base_pf | gated_pf | dpnl |
|---|---:|---:|---:|---:|
| risk_on | 37 | 2.35 | 2.30 | **−$11,525** |
| neutral | 3 | 0.99 | 1.03 | +$498 |
| risk_off | 6 | 1.27 | 1.28 | −$111 |
| crisis | 2 | 2.93 | 2.72 | −$109 |

**Decisive**: in risk_on regime (where most trades happen), blocking funding-extreme longs *destroys* $11.5K of PnL. The "crowded longs" detection is dominated by *winning* bull-continuation trades.

#### Detailed view of the three Phase-3 survivors

##### Candidate 1: confluence_breakout RSI ≤ 42 BLOCK
- TRAIN n 169→113, blocks 56 trades worth −$4,470 (PF stays 0.73→0.83)
- TEST n 319→253, blocks 66 trades worth −$3,977 (PF 1.59→1.85)
- Blocked regime: 86% risk_off / 14% neutral
- Blocked year mix: 56 (2022) / 37 (2023) / 29 (2024)
- **Caveat**: this is largely a "block bear-regime breakdown entries" gate. RSI < 42 is a generic indicator, not specifically a loser-archetype-feature. The gate "looks like" oi_divergence's mean RSI (40, d=-1.06) but it's confounded with generic bear-market avoidance.
- **PF gap**: 124% (train 0.83 vs test 1.85) — train arm is so deeply negative that PF gap is misleading. Absolute dPnL is robust both windows.

##### Candidate 2: liquidity_sweep funding_Z ≥ 1.0 BLOCK
- TRAIN n 33→28, blocks 5 trades worth −$3,223 (PF 2.95→11.97)
- TEST n 20→16, blocks 4 trades worth −$203 (PF 2.71→4.65)
- Blocked: 8 risk_on / 1 neutral
- Blocked year mix: 2020:3 ($+22), **2021:2 ($−3,245)**, 2024:4 ($−203)
- **Caveat**: the entire TRAIN benefit ($3,223) comes from 2 trades in 2021. Test benefit is $203 over 4 trades. **n_blocked_test = 4 → directional only by the n<30 floor.** Cannot deploy on this evidence.

##### Candidate 3: confluence_breakout RSI ≥ 66 BLOCK
- TRAIN dpnl +$1,931, TEST dpnl +$424
- Trade drop on test: only 5% (16 of 319). Marginal effect.
- **Caveat**: tiny effect, neither convincing nor disqualifying.

---

## 4. Recommendation

### Production changes: **NONE**

Per zero-tolerance rules, recommendations + diffs only. None of the surviving candidates clear the bar:

1. The clean cross-archetype hypothesis (funding_Z extreme → anti-long gate) **fails OOS** with substantial losses (-$12K on test for funding_Z ≥ 1.5, -$19K for rsi ≥ 78).
2. The marginal survivor (liquidity_sweep + funding_Z ≥ 1.0 block) has OOS n=4 blocked trades, below the n<30 directional floor.
3. The other survivors (confluence_breakout RSI gates) are essentially "block bear-market entries" — generic features confounded with bear-regime avoidance, not specifically a loser-archetype detection.

### Withheld YAML diff (DO NOT APPLY) — for documentation only

If someone insists on testing the marginal liquidity_sweep finding, the diff would be:

```yaml
# configs/archetypes/liquidity_sweep.yaml
# CAUTION: based on 4 blocked trades on test set (n<30 → directional only)
# Could regress in any future regime where funding-extreme longs are bull-continuation
hard_gates:
  - feature: funding_Z
    op: max
    value: 1.0
    nan_policy: skip
    description: "Block liquidity_sweep longs when funding overheated (long_squeeze territory). LOW CONFIDENCE n=4 test."
```

Strong recommendation: **do not apply** — the train benefit is concentrated in a single 2021 outlier and the OOS sample is too small to justify a production gate.

### Alternative investigations to pursue

1. **Rebuild the feature store with the missing 8 Binance whale columns** before re-attempting this hypothesis. The hypothesis cannot be fairly tested while `oi_change_4h`, `funding_oi_divergence`, `ls_ratio_extreme`, `taker_imbalance` are missing.
2. **Re-examine confluence_breakout** independently — the RSI ≤ 42 and RSI ≥ 66 gates both passed WFO with modest gains, suggesting CB needs its own gate tightening (not a cross-archetype phenomenon).
3. **Lesson #54 reinforced**: features-as-gates research should focus on **per-archetype identity-gate tightening**, not cross-archetype shared anti-gates. The structural diversity of archetypes means a feature that helps one archetype often hurts another.

---

## 5. Sample Size & Honest Caveats

| concern | impact |
|---|---|
| Trade log is **already filtered** (post-hard-gates) | The "blocked" trades we see are the survivors of existing gates. Adding another gate on top inherits the existing selection bias. |
| Train window has only 302 trades; test has 453 (test > train) | Inverted, because most strategy activity happened post-2023. WFO interpretation needs care. |
| **liquidity_sweep n_blocked_test = 4** | Directional only by the n<30 floor; cannot deploy. |
| oi_divergence in log: only n=10 | Effect-size catalog is suggestive but not statistically reliable. |
| Binance whale features (`oi_change_4h`, etc.) absent from feature store | The hypothesis as stated cannot be tested in full — only `funding_Z` + `rsi_14` + `volume_zscore` proxies available. |
| Confluence_breakout dominates the log (488/755 trades) | Any cross-archetype result is heavily biased by CB behavior. |
| One-shot train/test split (not full CPCV or rolling WFO) | OOS evidence is one window of 2 years. Could be regime-specific. |

---

## 6. What This Doesn't Test

- **Backtest validation**: this study uses the existing trade log as a sample. It does **not** re-run `bin/backtest_v11_standalone.py` with modified YAML gates. Phantom-outcome considerations (Lesson #54 / no fusion filters) are honored by avoiding any fusion-based gate; only structural feature thresholds were tested.
- **Confirmation gates** (feature_high *confirms* long) yielded zero meaningful survivors and were not deeply explored beyond Phase 4.B's "require funding_Z ≤ X" check, which clearly hurt PnL across the board.
- **Short-side archetypes**: testing loser-feature gates for shorts was out of scope — only `long_squeeze` is a short archetype, and gating other shorts on its own feature would be tautological.
- **The actual S9 oi_divergence detection logic**: because `oi_change_4h`, `oi_change_24h`, `ls_ratio_extreme`, `taker_imbalance` are NaN in the feature store, S9's real detection cannot be tested as a cross-gate.
- **Time-of-day / day-of-week effects** on gate efficacy.
- **Live (post-2024) data**: live-mode performance of `funding_divergence` (winning) and `long_squeeze` (losing) was cited by the user, but live trade logs were not pulled into this analysis.

---

## 7. Files Modified

### Read-only inspections (no production edits)
- `engine/archetypes/logic.py` (lines 952-998 inspected — _check_S4/S5)
- `configs/archetypes/oi_divergence.yaml`
- `configs/archetypes/long_squeeze.yaml`
- `configs/archetypes/funding_divergence.yaml`
- `docs/knowledge/MEMORY.md`
- `docs/knowledge/structural_checks.md`

### Scratch results written
- `results/cross_archetype/loser_features/trades_with_features.parquet` (755 trades joined with feature store)
- `results/cross_archetype/loser_features/phase2_gate_matrix.parquet` (40 gate × archetype combos)
- `results/cross_archetype/loser_features/phase3_wfo.parquet` (7 candidates with train/test metrics)
- `docs/knowledge/loser_features_as_gates.md` (this report)

### Branch
- `quant/loser-features-as-gates` (no production files modified — only this report + scratch results)

---

## TL;DR for the user

You asked: "Do the FEATURES computed by losing archetypes (oi_divergence, long_squeeze) work as confirmation gates for OTHER archetypes?"

**Answer**: No — at least not as cross-archetype anti-gates with the features available in the current store. The two interpretable findings:

1. **The intuitive form fails OOS**: blocking longs on `funding_Z ≥ 1.5` (long_squeeze's territory) loses **$12K** in 2023-2024. Crowded longs in 2020-2024 BTC are dominated by bull-continuation, not by impending squeezes.

2. **The one marginal survivor is too small**: liquidity_sweep + `funding_Z ≥ 1.0 block` passes WFO but only blocks 4 trades in test (worth −$200) — below the n<30 floor.

The hypothesis would be **worth re-testing** once the feature store is rebuilt with the 8 Binance whale columns (`oi_change_4h`, `funding_oi_divergence`, `ls_ratio_extreme`, `taker_imbalance`). Right now, oi_divergence's actual detection logic cannot be tested as a cross-archetype gate because its core features are NaN.

No production changes recommended.
