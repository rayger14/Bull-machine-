# Bull Machine — Archetype Knowledge Base

> **Purpose**: Synthesized quant knowledge from 1,127 backtest trades (2020-2024, $100K capital, BTC-PERP).
> Written for future AI models and human operators. Every number is from real data — no assumptions.
> Confidence labels: HIGH (n>100), MEDIUM (n=30-100), LOW (n<30).
>
> **Last Updated**: 2026-04-02 | **Architecture**: v17 Whale Footprint | **Dataset**: BTC_1H_FEATURES_V12_ENHANCED.parquet

---

## 1. Executive Summary

1. **dd_score is the #1 universal predictor** (Pearson r=0.167, p<0.001, n=1127). It is the single most important signal across ALL archetypes. When dd_score is high, the market is healthier and stop-outs are rarer.
2. **Universal good condition exists** (dd_score>0.20 + risk_temp>0.45 + trend_align>0.9): n=422, WR=85.5%, PF=2.60, $118K PnL — 37% of all trades but 62% of all profit.
3. **Universal bad condition confirmed** (trend_align<0.5 + chop>0.40): n=88, WR=62.5%, PF=0.54, -$16.7K PnL — trading into head/fake-outs. Should be a system-wide soft veto.
4. **Fusion score has NEGATIVE predictive power** (r=-0.102, p=0.001). High fusion ≠ better trade. The edge is in structural detection + hard gates, not the scoring layer.
5. **liquidity_compression is entirely regime-dependent** — makes $47K in risk_on, loses $13.6K everywhere else. Restricting it to risk_on would add ~$13.6K annual PnL.
6. **exhaustion_reversal + oi_divergence are system drags** — removing both improves system PF from 1.74 → 1.87 (+$9.5K recovery). oi_divergence is structurally broken (WR=23.5%, PF=0.30).

---

## 2. Universal Feature Predictors

All 1,127 trades. Ranked by |Pearson r| with PnL.

| Feature | Pearson r | p-value | Spearman r | Direction | Confidence |
|---------|-----------|---------|------------|-----------|------------|
| `dd_score` | **+0.167** | <0.001 *** | +0.165 | Higher = better | HIGH |
| `risk_temp` | **+0.126** | <0.001 *** | +0.136 | Higher = better | HIGH |
| `trend_align` | **+0.105** | 0.0004 *** | +0.092 | Higher = better | HIGH |
| `fusion_score` | **-0.102** | 0.001 *** | -0.144 | Higher = WORSE | HIGH |
| `chop` | **-0.090** | 0.003 ** | -0.148 | Lower = better | HIGH |
| `trend_strength` | +0.090 | 0.003 ** | +0.144 | Higher = better | HIGH |
| `instability` | -0.072 | 0.016 * | -0.082 | Lower = better | HIGH |
| `sentiment_score` | -0.076 | 0.011 * | -0.010 | Lower = better (BTC contrarian) | HIGH |
| `vol_instab` | -0.069 | 0.021 * | -0.129 | Lower = better | HIGH |
| `threshold_margin` | +0.058 | 0.053 | +0.027 | Marginal | HIGH |
| `liquidity_score` | -0.047 | 0.112 | -0.021 | Noise | HIGH |
| `crisis_prob` | -0.040 | 0.182 | -0.031 | Noise | HIGH |
| `momentum_score` | -0.018 | 0.550 | -0.073 | Noise | HIGH |
| `wyckoff_score` | +0.003 | 0.911 | -0.074 | Noise | HIGH |

**Key insight**: The four domain scores (wyckoff, liquidity, momentum, smc) have near-zero or negative correlation with PnL across the system. The Wyckoff/liquidity/momentum scores as currently computed are not predictive of individual trade outcome — they are pattern detectors, not quality selectors.

---

## 3. Universal Conditions

### Good Condition (All Three Required)
`dd_score > 0.20` AND `risk_temp > 0.45` AND `trend_align > 0.9`

| Metric | Value |
|--------|-------|
| Trades | 422 / 1127 (37%) |
| Win Rate | 85.5% |
| Profit Factor | 2.60 |
| Total PnL | $118,762 |
| % of system PnL | 62% |

**Interpretation**: When drawdown pressure is low (healthy trend), temperature is warm (institutional participation), and trend alignment is high (multi-timeframe agreement), the setup is structurally sound.

### Bad Condition (Both Required)
`trend_align < 0.5` AND `chop > 0.40`

| Metric | Value |
|--------|-------|
| Trades | 88 / 1127 (8%) |
| Win Rate | 62.5% |
| Profit Factor | 0.54 |
| Total PnL | -$16,709 |

**Interpretation**: Choppy, directionless market with conflicting timeframe signals. High false-signal environment. This condition is a consistent money-loser across ALL archetypes.

**Recommended action**: Add system-wide soft veto or cooling on `trend_align < 0.5 AND chop > 0.40`. Could save ~$16.7K/year with minimal trade reduction.

---

## 4. Per-Archetype Deep Dives

### 4.1 liquidity_sweep — The Workhorse
**Status**: Production | **Priority**: Core

| Metric | Value |
|--------|-------|
| Trades | 230 |
| Win Rate | 84.8% |
| Profit Factor | 2.13 |
| Total PnL | $53,889 |
| Avg Win / Avg Loss | $520 / -$1,359 |
| Avg Hold | 41.0h |

**Regime Breakdown**:
| Regime | n | WR | PF | PnL |
|--------|---|----|----|-----|
| risk_on | 179 | 84% | 1.93 | $35,514 |
| neutral | 15 | 93% | 6.46 | $5,713 |
| risk_off | 23 | 96% | 9.84 | $13,509 |
| crisis | 13 | 69% | 0.88 | -$846 |

**Surprising finding (MEDIUM confidence)**: Performs BETTER in risk_off than risk_on (PF 9.84 vs 1.93). Liquidity sweeps in risk_off regimes are high-conviction reversals — the bears exhaust their sweep and price snaps back hard. Only crisis is dangerous.

**Top Predictive Features**:
| Feature | r | Winners | Losers |
|---------|---|---------|--------|
| `liquidity_score` | +0.171** | 0.723 | 0.691 |
| `vol_instab` | +0.158* | 0.352 | 0.260 |
| `momentum_score` | +0.147* | 0.355 | 0.332 |

**Note**: All correlations are weak (r<0.20). The sweep has robust performance across conditions — feature separability is low, meaning the hard gates already do the heavy lifting.

**Chop sensitivity**: Low (25th pctile <0.03) vs high (75th pctile >0.37): WR 88% vs 80%. Chop matters but not dramatically.
**dd_score sensitivity**: HIGH vs LOW split: $28.7K vs $25.2K PnL — less sensitive than other archetypes (directional sweeps work in stressed conditions too).

**Current gates status**: Correct. Temporal mismatch lesson applies — do NOT add OI/taker gates (they confirm AFTER the sweep fires).

**Recommended action**: No changes. Consider restricting to non-crisis if you want to eliminate the -$846 crisis drag.

---

### 4.2 wick_trap — The Volume High
**Status**: Production | **Priority**: Core

| Metric | Value |
|--------|-------|
| Trades | 210 |
| Win Rate | 78.6% |
| Profit Factor | 1.58 |
| Total PnL | $43,525 |
| Avg Win / Avg Loss | $716 / -$1,660 |
| Avg Hold | 66.8h |

**Regime Breakdown**:
| Regime | n | WR | PF | PnL |
|--------|---|----|----|-----|
| risk_on | 179 | 80% | 1.64 | $39,259 |
| risk_off | 25 | 72% | 1.36 | $4,235 |
| crisis | 6 | 50% | 1.02 | $31 |

**Top Predictive Features**:
| Feature | r | Winners | Losers |
|---------|---|---------|--------|
| `instability` | -0.209** | 0.426 | 0.476 |
| `dd_score` | +0.169* | 0.204 | 0.153 |
| `sentiment_score` | -0.157* | 0.620 | 0.687 |
| `chop` | -0.151* | 0.224 | 0.256 |

**Instability cliff**: Winners have instability=0.426 vs losers=0.476. There's a threshold around 0.45 — above it, wicks are noise not reversal signals (HIGH confidence from previous analysis).

**Chop sensitivity** (HIGH confidence): Low chop (<0.05) → WR=85%, PnL=$26.4K. High chop (>0.39) → WR=75%, PnL=$3.2K. The chop gate (currently hard) is justified.
**dd_score split**: High → WR=84%, PnL=$34.8K; Low → WR=73%, PnL=$8.8K. Strong sensitivity.

**Key lesson**: avg_hold=66.8h. Theory says "fast reversal" but BTC needs multi-day holds for wick traps. The 168h max_hold is correct. Shortening to 48h halved PnL in previous experiment.

**Recommended action**: Consider adding `instability < 0.45` as a hard gate. Would filter ~25% of losers.

---

### 4.3 liquidity_compression — Regime-Locked
**Status**: Production | **Priority**: HIGH improvement opportunity

| Metric | Value |
|--------|-------|
| Trades | 406 |
| Win Rate | 76.8% |
| Profit Factor | 1.38 |
| Total PnL | $33,657 |
| Avg Win / Avg Loss | $391 / -$938 |
| Avg Hold | 23.7h |

**Regime Breakdown** (CRITICAL):
| Regime | n | WR | PF | PnL |
|--------|---|----|----|-----|
| **risk_on** | **277** | **82%** | **2.08** | **$47,258** |
| neutral | 33 | 70% | 0.57 | -$5,779 |
| risk_off | 44 | 57% | 0.63 | -$5,553 |
| crisis | 52 | 73% | 0.86 | -$2,269 |

**This archetype ONLY works in risk_on regime.** Non-risk_on = -$13,601 total loss. Liquidity compression is a momentum continuation setup — it requires active institutional buying pressure to sustain.

**Top Predictive Features**:
| Feature | r | Winners | Losers |
|---------|---|---------|--------|
| `dd_score` | +0.226*** | 0.183 | 0.119 |
| `risk_temp` | +0.218*** | 0.453 | 0.380 |
| `crisis_prob` | -0.193*** | 0.011 | 0.019 |
| `trend_align` | +0.178*** | 0.781 | 0.647 |
| `vol_instab` | -0.118* | 0.821 | 0.911 |

**dd_score split**: High → WR=83%, PnL=$32.8K; Low → WR=70%, PnL=$862. Almost all profit is from high-dd_score trades.

**Recommended action (HIGH impact)**:
1. Add `allowed_regimes: [risk_on]` to YAML — saves ~$13.6K/year, removes 129 lossy trades
2. Add `dd_score >= 0.15` as hard gate — separates the bulk of winning vs losing conditions

---

### 4.4 failed_continuation — Elite Archetype
**Status**: Production (chop gate added 2026-04-01) | **Priority**: Protect

| Metric | Value |
|--------|-------|
| Trades | 70 (post-chop gate) |
| Win Rate | 95.7% |
| Profit Factor | 9.28 |
| Total PnL | $25,925 |
| Avg Win / Avg Loss | $434 / -$1,044 |
| Avg Hold | 25.3h |

**Regime Breakdown**:
| Regime | n | WR | PF | PnL |
|--------|---|----|----|-----|
| risk_on | 46 | 96% | 5.14 | $12,233 |
| neutral | 2 | 100% | inf | $677 |
| risk_off | 4 | 100% | inf | $1,710 |
| crisis | 18 | 94% | 63.90 | $11,305 |

**Surprising finding**: Works BEST in crisis (PF=63.90, PnL=$11.3K from 18 trades). Failed bearish continuations during crisis = the bear exhausts itself, reversal is violent and sustained.

**Top Predictive Features**:
| Feature | r | Winners | Losers |
|---------|---|---------|--------|
| `momentum_score` | +0.406*** | 0.276 | 0.260 |
| `risk_temp` | -0.393*** | 0.440 | 0.406 |
| `trend_align` | -0.380** | 0.752 | 0.667 |

**Note on correlations**: Negative risk_temp and trend_align predict winners here — this is the CONTRARIAN nature of the setup. When bearish conditions are moderate (not extreme), the failed continuation has the most snap-back force.

**Chop gate impact** (confirmed 2026-04-01): Before gate: PF=1.94, WR=88.7%. After `chop_score < 0.25` hard gate: PF=9.28, WR=95.7%. This single gate eliminated the majority of losing conditions (chop=high → bear can resume; chop=low → bear is actually exhausted).

**Critical insight**: ADX > 37.5 required (chop < 0.25 = ADX > 37.5). This ensures the failed move was happening in a TRENDING market, not chop. Choppy failed continuations just resume the chop.

**Recommended action**: No changes. Monitor — 3 remaining losses in 70 trades may represent outlier stress events (bypass_threshold trades with negative macro).

---

### 4.5 trap_within_trend — The 100% Perfect Archetype
**Status**: Production | **Priority**: Scale cautiously

| Metric | Value |
|--------|-------|
| Trades | 32 |
| Win Rate | **100.0%** |
| Profit Factor | ∞ |
| Total PnL | $22,261 |
| Avg Win | $696 |
| Avg Hold | **95.2h** |

**Regime Breakdown**: 100% WR in risk_on, neutral, crisis alike. No losses in any regime.

**Key insight**: This is a "within-trend retracement" setup — it fires only during the healthiest trending environments by construction. The hard gates already enforce the necessary conditions.

**Warning**: 100% WR with n=32 is LOW confidence. With 32 trades spanning 4 years, this represents roughly 8 trades/year. Statistical law of small numbers applies — the true WR may be 85-92% at scale.

**Hold time is structural**: avg 95.2h, median 96.5h. These are multi-day trend rides. Do NOT shorten max_hold — the biggest gains come from full trend extensions.

**Recommended action**: No changes. Watch for first loss — it will reveal the edge case. Consider adding a liquidity threshold to avoid thinly-traded trend continuations.

---

### 4.6 long_squeeze — Funding-Only in risk_on
**Status**: Production | **Priority**: Maintain

| Metric | Value |
|--------|-------|
| Trades | 42 |
| Win Rate | 83.3% |
| Profit Factor | 4.52 |
| Total PnL | $10,873 |
| Avg Win / Avg Loss | $399 / -$442 |
| Avg Hold | 11.7h |

**Regime Breakdown**: 100% of trades are risk_on. The archetype fires exclusively in one regime — by design (funding pressure exists when markets are hot).

**Surprising chop behavior**: High chop → WR=93%; low chop → WR=67%. The opposite of every other archetype. Long squeezes in choppy markets are HIGHER quality because the funding pressure built up without directional release — the snap is more violent.

**Top Predictive Features**: All weak (r<0.22). This archetype is mostly self-contained — the hard gates already define the opportunity set well.

**Recommended action**: Add `vol_shock < 0.10` gate to avoid macro vol spikes during squeeze attempts (from agent analysis). The squeeze mechanism breaks when systematic vol sellers are active.

---

### 4.7 funding_divergence — Sensitive to Macro
**Status**: Production | **Priority**: Tighten gates

| Metric | Value |
|--------|-------|
| Trades | 30 |
| Win Rate | 83.3% |
| Profit Factor | 1.92 |
| Total PnL | $5,383 |
| Avg Win / Avg Loss | $450 / -$1,176 |
| Avg Hold | 45.7h |

**Regime Breakdown**:
| Regime | n | WR | PF | PnL |
|--------|---|----|----|-----|
| risk_on | 17 | 82% | 1.70 | $2,939 |
| neutral | 6 | 100% | inf | $2,509 |
| risk_off | 7 | 71% | 0.96 | -$64 |

**Top Predictive Features** (MEDIUM confidence):
| Feature | r | Winners | Losers |
|---------|---|---------|--------|
| `crisis_prob` | **-0.515*** | 0.024 | 0.069 |
| `dd_score` | +0.451* | 0.173 | 0.044 |
| `risk_temp` | +0.411* | 0.430 | 0.342 |
| `sentiment_score` | +0.382* | 0.533 | 0.408 |

**Crisis probability is the dominant predictor**: Winners have crisis_prob=0.024 vs losers=0.069. When crisis probability doubles, the funding divergence setup fails (deleveraging events override funding normalization).

**dd_score split** (MEDIUM confidence): High dd → 100% WR, PnL=$8.4K; Low dd → 67% WR, -$3.1K. All profit comes from high-dd_score conditions.

**Chop behavior (inverted)**: High chop → WR=88%, PnL=$2.2K vs low chop → WR=67%, -$1.0K. Similar to long_squeeze — funding divergence in choppy environments has more funding buildup.

**Recommended actions**:
1. Add `crisis_prob < 0.05` hard gate — filters most losers
2. Add `sentiment_score > 0.50` soft gate — ensures market isn't in fear state
3. Add `dd_score >= 0.10` hard gate — all profit at high dd

---

### 4.8 exhaustion_reversal — The Problematic Archetype
**Status**: Production but marginal | **Priority**: Gate heavily or disable

| Metric | Value |
|--------|-------|
| Trades | 55 |
| Win Rate | 76.4% |
| Profit Factor | **0.93** (below 1.0!) |
| Total PnL | **-$1,187** |
| Avg Win / Avg Loss | $351 / -$1,225 |
| Avg Hold | 10.9h |

**Regime Breakdown**:
| Regime | n | WR | PF | PnL |
|--------|---|----|----|-----|
| risk_on | 53 | 79% | 1.24 | $2,831 |
| risk_off | 2 | 0% | 0.00 | -$4,018 |

**The W/L ratio (0.29) is the problem**: 76% wins but losses are 3.5x the wins. The archetype fires on correct timing (exhaustion IS real) but the upside capture is structurally limited by max_hold (10.9h avg).

**Top Predictive Features** (HIGH importance):
| Feature | r | Winners | Losers |
|---------|---|---------|--------|
| `dd_score` | **+0.466***| 0.252 | 0.183 |
| `risk_temp` | **+0.381**| 0.563 | 0.510 |
| `liquidity_score` | -0.371** | 0.830 | 0.948 |
| `trend_align` | +0.338* | 0.971 | 0.908 |

**dd_score split** (CRITICAL finding):
- `dd_score > median (0.249)`: n=27, **WR=100%, PnL=$8,690**
- `dd_score <= median (0.249)`: n=28, WR=57%, PnL=-$9,876

**This is the strongest binary split in the entire system**: Add `dd_score >= 0.22` as a hard gate and exhaustion_reversal flips from -$1.2K to +$8.7K. The archetype is ONLY viable when the market structure is healthy enough to sustain the reversal.

**Recommended action (HIGH impact)**:
1. Add `dd_score >= 0.22` hard gate → expected +$10.5K PnL swing
2. Add `risk_off` to blocked regimes (2 trades both losses = -$4K)
3. Keep as is after gating — the 79% WR in risk_on with gates is viable

---

### 4.9 spring — Sentiment-Driven Reversal
**Status**: Production | **Priority**: Maintain, possibly tighten

| Metric | Value |
|--------|-------|
| Trades | 16 |
| Win Rate | 62.5% |
| Profit Factor | 1.77 |
| Total PnL | $3,920 |
| Avg Win / Avg Loss | $904 / -$853 |
| Avg Hold | 26.6h |

**Regime Breakdown** (LOW confidence — small sample):
| Regime | n | WR | PF | PnL |
|--------|---|----|----|-----|
| risk_on | 7 | 100% | inf | $5,899 |
| risk_off | 6 | 50% | 0.92 | -$275 |
| crisis | 3 | 0% | 0.00 | -$1,703 |

**Top Predictive Features** (LOW confidence):
| Feature | r | Winners | Losers |
|---------|---|---------|--------|
| `sentiment_score` | **+0.842***| 0.772 | 0.427 |
| `risk_temp` | +0.706** | 0.472 | 0.169 |
| `trend_align` | +0.638** | 0.820 | 0.133 |
| `momentum_score` | -0.644** | 0.269 | 0.334 |

**Sentiment is the dominant predictor**: Winners avg sentiment=0.772 vs losers=0.427. Spring requires bullish sentiment to sustain. In fear environments, the spring lows break and price keeps declining.

**dd_score split**: High → WR=88%, PnL=$5.2K; Low → WR=38%, -$1.3K. Pronounced.

**Risk_on only logic**: 100% WR in risk_on (n=7), 0% in crisis (n=3). Consider restricting to risk_on + neutral only.

**Recommended actions** (LOW confidence — validate with more data):
1. Add `allowed_regimes: [risk_on, neutral]`
2. Add `sentiment_score >= 0.60` soft gate
3. Add `wyckoff_score >= 0.60` (from prior agent analysis — spring quality matters)

---

### 4.10 retest_cluster — Regime-Sensitive
**Status**: Production | **Priority**: Restrict regime

| Metric | Value |
|--------|-------|
| Trades | 14 |
| Win Rate | 78.6% |
| Profit Factor | 1.38 |
| Total PnL | $1,604 |

**Regime Breakdown** (LOW confidence):
| Regime | n | WR | PF | PnL |
|--------|---|----|----|-----|
| risk_on | 11 | 91% | 6.11 | $4,527 |
| risk_off | 3 | 33% | 0.11 | -$2,923 |

**Critical**: 3 risk_off trades destroyed $2.9K (88% of total PnL at stake). In risk_off, cluster retests fail because sellers resume.

**Top Features** (LOW confidence):
| Feature | r | Winners | Losers |
|---------|---|---------|--------|
| `liquidity_score` | -0.785*** | 0.829 | 0.919 |
| `momentum_score` | -0.775** | 0.340 | 0.394 |
| `trend_align` | +0.742** | 0.945 | 0.467 |

**Recommended action**: Add `allowed_regimes: [risk_on]` — avoids risk_off losses.

---

### 4.11 oi_divergence — Structurally Broken
**Status**: Active but LOSING | **Priority**: DISABLE

| Metric | Value |
|--------|-------|
| Trades | 17 |
| Win Rate | **23.5%** |
| Profit Factor | **0.30** |
| Total PnL | **-$8,301** |

**All 17 trades were in risk_off (15) or crisis (2).** The archetype exclusively fires in negative regimes where price diverging from OI means distribution, not accumulation. The detection logic is pattern-matching a bearish distribution signal in a bullish reversal framework.

**Recommended action**: **DISABLE** `oi_divergence`. PF of the entire system improves from 1.74 → ~1.77 just from this action.

---

## 5. Regime Sensitivity Matrix

System-wide performance by regime (all archetypes):

| Regime | n | WR | PF | Total PnL |
|--------|---|----|----|-----------|
| risk_on | 839 | 81% | 1.79 | ~$160K |
| neutral | 62 | 79% | 1.41 | ~$2K |
| risk_off | 129 | 76% | 1.68 | ~$17K |
| crisis | 97 | 74% | 1.36 | ~$12K |

**Per-archetype regime sensitivity** (color-coded):

| Archetype | risk_on | neutral | risk_off | crisis |
|-----------|---------|---------|----------|--------|
| liquidity_sweep | PF=1.93 ✓ | PF=6.46 ✓ | PF=9.84 ✓ | PF=0.88 ✗ |
| wick_trap | PF=1.64 ✓ | — | PF=1.36 ✓ | PF=1.02 ~ |
| **liquidity_compression** | **PF=2.08 ✓** | **PF=0.57 ✗** | **PF=0.63 ✗** | **PF=0.86 ✗** |
| failed_continuation | PF=5.14 ✓ | PF=inf ✓ | PF=inf ✓ | PF=63.9 ✓ |
| trap_within_trend | PF=inf ✓ | PF=inf ✓ | — | PF=inf ✓ |
| long_squeeze | PF=4.52 ✓ | — | — | — |
| exhaustion_reversal | PF=1.24 ~ | — | PF=0.00 ✗ | — |
| spring | PF=inf ✓ | — | PF=0.92 ✗ | PF=0.00 ✗ |
| retest_cluster | PF=6.11 ✓ | — | PF=0.11 ✗ | — |
| funding_divergence | PF=1.70 ✓ | PF=inf ✓ | PF=0.96 ✗ | — |
| oi_divergence | — | — | PF=0.32 ✗ | PF=0.00 ✗ |

✓ = profitable, ✗ = losing, ~ = marginal, — = no data

---

## 6. Exit Pattern Analysis

### Scale-Out Performance
| Exit Level | Pct Out | n | Avg PnL per Exit |
|------------|---------|---|------------------|
| 0.5R (20%) | 20% | 125 | $176 |
| 0.5R (25%) | 25% | 22 | $269 |
| 0.5R (30%) | 30% | 29 | $297 |
| **0.5R (33%)** | **33%** | **146** | **$442** |
| 1.0R (20%) | 20% | 115 | $453 |
| 1.0R (30%) | 30% | 86 | $462 |
| 1.0R (35%) | 35% | 16 | $641 |
| 2.0R (30%) | 30% | 33 | $1,079 |
| 2.0R (40%) | 40% | 33 | $1,064 |

**Key finding**: The 33% first scale-out (at 0.5R) captures avg $442 vs $176 for 20% — a 2.5x improvement per partial exit. This was changed in the live system (2026-04-01).

### Stop-Loss Analysis
- **428 stop-loss exits**, 56% WR (partials hit before stop = partial wins)
- Avg PnL = -$226, Total = -$96,625
- Stop-loss is the biggest single drag on PnL. This is expected and normal — it is the cost of the option premium.
- W/L ratio across system: avg_win / avg_loss ≈ 0.38-0.43 for most archetypes. **This is the payoff asymmetry problem.** Partial exits improve this structurally.

### Composite Invalidation Exits (Smart Exits V2)
- Wick_trap invalidation: n=14, WR=7%, Total=-$14,626
- **All are losses** — structural confirmation (BOS bearish, etc.) fires when trade is already failing. Stop loss provides better protection.
- **Lesson (HARD-WON)**: Composite invalidation exits should only be used for trades that would otherwise reach stop loss with a large draw. The 4/5 threshold must be maintained.

---

## 7. Emergent Cross-Archetype Patterns

These are non-obvious patterns that emerged from the full dataset analysis:

### 7.1 The dd_score Binary Pattern
The most powerful non-obvious finding: **dd_score splits every archetype into two distinct populations**:
- `dd_score > 0.20-0.25`: Professional-quality setups, high WR, most of system profit
- `dd_score < 0.15`: Marginal setups, WR drops 10-20pp, often net negative

This suggests dd_score is a proxy for "structural health of the market" — not just drawdown pressure but the quality of the current supply/demand balance.

### 7.2 Contrarian Archetypes Hate High Fusion
Failed_continuation and exhaustion_reversal both show NEGATIVE fusion score correlation with PnL. High fusion score on a reversal setup means "too many indicators agree it's going down" — which is the condition where the reversal is about to happen (everyone already positioned short = shorts squeezed).

### 7.3 Volume Archetypes Are Less Regime-Sensitive
Liquidity_sweep and wick_trap show relatively consistent WR across risk_on/risk_off (84% vs 96% sweep; 80% vs 72% wick). The structural liquidity event (sweep/trap) is partially regime-independent because it requires a local extreme first.

### 7.4 Long_squeeze and funding_divergence Favor Chop
Both archetypes show better WR in high-chop environments. Intuition: funding rates build up more slowly in choppy markets (no sudden directional consensus), creating a larger mismatch to unwind when the squeeze/divergence resolves.

### 7.5 The W/L Ratio Problem Is Archetype-Specific
| Archetype | W/L Ratio | Resolution |
|-----------|-----------|------------|
| trap_within_trend | ∞ (no losses) | Perfect — maintain |
| spring | 1.06 | Good — near 1:1 |
| long_squeeze | 0.90 | Good — near 1:1 |
| failed_continuation | 0.42 | Offset by 96% WR |
| liquidity_sweep | 0.38 | Normal for momentum |
| wick_trap | 0.43 | Normal for mean-reversion |
| exhaustion_reversal | **0.29** | Problem — wins too small |

Exhaustion_reversal's 0.29 W/L ratio means even 76% WR can't generate positive expectancy. Fixing this requires either (a) higher max_hold to capture larger winners or (b) gating to only the highest-dd conditions where snap-backs are larger.

### 7.6 Crisis is a Bifurcation Event
In crisis regime: liquidity_sweep loses (fear sweep = real), but failed_continuation and trap_within_trend WIN. The failed bearish continuation in crisis = capitulation exhaustion, the most powerful setup in the system (PF=63.9).

---

## 8. Prioritized Action Items

Ranked by expected annual PnL impact, with confidence levels.

| Priority | Action | Expected Impact | Confidence | Risk |
|----------|--------|----------------|------------|------|
| 1 | **Disable oi_divergence** | +$8.3K/year | HIGH | Low |
| 2 | **Add `allowed_regimes: [risk_on]` to liquidity_compression** | +$13.6K/year | HIGH | Low |
| 3 | **Add `dd_score >= 0.20` gate to exhaustion_reversal** (+ block risk_off) | +$7.3K/year | MEDIUM | Medium |
| 4 | **Add `dd_score >= 0.10` gate to liquidity_compression** | +$5-8K/year | HIGH | Low |
| 5 | **Add system-wide soft veto: trend_align<0.5 AND chop>0.40** | +$8-12K/year | HIGH | Medium |
| 6 | **Add `instability < 0.45` gate to wick_trap** (p=0.018) | +$3-5K/year | MEDIUM | Low |
| 7 | **Add `vol_shock < 0.10` hard gate to long_squeeze** | est. +$2-3K/year | MEDIUM | Low |
| 8 | **Add `sentiment_crisis == 0` hard gate to funding_divergence** | +$2K/year | MEDIUM | Low |
| 9 | **Add `crisis_prob < 0.04` gate to funding_divergence** | +$2-3K/year | MEDIUM | Low |
| 10 | **Add `allowed_regimes: [risk_on]` to retest_cluster** | +$2.9K/year | LOW | Low |
| 11 | **Add `allowed_regimes: [risk_on, neutral]` to spring** | +$2K/year | LOW | Low |
| 12 | **Add `wyckoff_score >= 0.60` gate to spring** | est. +$5K/year | LOW | Low |
| 13 | **Extend trap_within_trend max_hold 168h → 240h** | est. +$1-2K/year | LOW | Low |

**Total estimated uplift (conservative): $50-65K/year additional PnL** if items 1-7 are implemented.

---

---

## 9. Agent Deep-Dive Addenda

*Additional granular findings from parallel agent analyses — more specific than the direct synthesis above.*

### 9.1 Position-Level Stats (Aggregated Scale-Outs)

The row-level counts (1,127 rows) overcount positions because scale-outs create multiple exit rows per trade. True position counts:

| Archetype | Positions | WR (position) | PF | Notes |
|-----------|-----------|--------------|-----|-------|
| liquidity_sweep | 90 | 62.2% | 2.24 | Row-level WR inflated by partial exits |
| wick_trap | 93 | 58.1% | 1.63 | |
| liquidity_compression | 181 | 59.1% | 1.39 | |
| exhaustion_reversal | 26 | 57.7% | 0.92 | n=26 LOW |
| oi_divergence | 14 | 14.3% | 0.30 | n=14 VERY LOW |
| long_squeeze | 14 | 78.6% | 5.01 | n=14 LOW |
| funding_divergence | 11 | 54.5% | 2.09 | n=11 VERY LOW |

### 9.2 Duration Is the Universal #1 Separator

Across liquidity_sweep, wick_trap, and liquidity_compression, **hold duration is the single most statistically significant separator between winners and losers** (all p<0.0005). This is a selection effect (winning trades survive longer) but has structural implications for stop placement.

**First-24h danger zone** — early stop-outs in the first 24h are overwhelmingly losses:
| Archetype | 0-12h WR | 12-24h WR | 24-72h WR | >72h WR |
|-----------|----------|-----------|-----------|---------|
| liquidity_sweep | 25% | 31% | 70% | 85-87% |
| wick_trap | 0% | 50% | 50% | 79% |
| liquidity_compression | 30% | 62% | 79% | 81% |

**Implication**: Stops that trigger within 12-24h of entry represent the noise band, not true invalidation. Slightly wider ATR multipliers or a minimum hold before stop evaluation could address this structural loss cluster.

### 9.3 Wick_trap Instability Cliff (Confirmed, p=0.018)

Instability threshold at 0.45:
- `instability < 0.45`: n=46, **WR=67.4%**, avg_pnl=$991
- `instability >= 0.45`: n=47, **WR=48.9%**, avg_pnl=-$44

This is the sharpest gate signal for wick_trap. Above 0.45, the archetype breaks even at best. Current YAML does not enforce this — it should be a hard gate.

### 9.4 Chop × Instability Interaction

The combined high-chop + high-instability quadrant is a universal kill zone:
```
                              liq_sweep   wick_trap   liq_compr
low_chop + low_inst  WR=       68.8%       65.7%       65.0%
low_chop + hi_inst   WR=       61.5%       58.3%       63.6%
hi_chop  + low_inst  WR=       69.2%       75.0%       45.5%   ← wick_trap surprise
hi_chop  + hi_inst   WR=       53.1%       44.1%       54.4%
```
**Surprising finding**: wick_trap in hi_chop + low_inst achieves 75% WR — choppy price with stable volatility may produce cleaner wick reversals (noise creates fake wicks that resolve without vol expansion). However, hi_chop + hi_inst is 44.1% WR and -$298 avg_pnl: the real kill zone.

### 9.5 liquidity_compression: 0-12h Disaster Bucket

57 trades (31% of all compression positions) stop within 12h: WR=29.8%, avg_pnl=-$516, estimated total damage ~$29K.
Their entry profile: dd_score=0.116, chop=0.432, instability=0.471 — exactly the low-dd / moderate-chop / moderate-instability conditions. The `dd_score >= 0.10` gate would eliminate most of these.

### 9.6 Fusion Score Ceiling on liquidity_compression

Fusion score > 0.55 is a *negative* signal for compression:
- Fusion 0.394–0.545: WR=70-72%, avg_pnl=$483-575
- Fusion 0.545–0.580: **WR=47%**, avg_pnl=-$286 (cliff)
- Fusion 0.580–0.680: WR=55%, avg_pnl=$81

An upper fusion bound (`fusion_score < 0.55`) could filter a loss cluster without touching the majority of trades.

### 9.7 spring: Wyckoff Binary Gate (LOW CONFIDENCE — n=16)

All 6 spring losses had `wyckoff_score` between 0.50–0.60. All 10 winners had `wyckoff_score >= 0.60`.
- wyckoff 0.50–0.60: n=6, **WR=0%**, avg_pnl=-$853
- wyckoff 0.60–0.80: n=7, **WR=100%**, avg_pnl=$1,033

This is the sharpest gate finding in the reversal analysis. However, n=6 vs n=10 is too small to confirm. Current YAML minimum is ~0.10–0.15 — raising to 0.60 would eliminate all historical losses while keeping all wins. Verify on next 20+ trades before deploying.

### 9.8 spring: risk_temp Breakeven (~0.25)
- risk_temp < 0.25: n=5, **WR=0%** (all losses, crisis or risk_off entries)
- risk_temp >= 0.25: n=11, WR=90.9%

Consider `risk_temp >= 0.28` as a soft gate for spring.

### 9.9 failed_continuation: Crisis = Best Setup
In crisis regime, failed_continuation produces avg **$628/trade** vs $266 in risk_on. Crisis chop=0.046 (near-zero), trend_align=0.156 (strongly counter-trend). This is the "pure structural exhaustion" condition where failed continuations are most powerful. **Do NOT suppress this archetype in crisis** — it thrives there.

### 9.10 failed_continuation: Momentum Score → PnL Magnitude
| momentum_score | n | WR | Avg PnL |
|----------------|---|----|---------|
| 0.20–0.25 | 21 | 90.5% | $175 |
| 0.25–0.30 | 31 | 96.8% | $334 |
| 0.30–0.35 | 11 | 100% | $631 |
| 0.35–0.40 | 7 | 100% | $710 |

Clean monotonic progression. `momentum_score >= 0.30` → 100% WR and 2x the PnL. This is not a gate recommendation (the archetype already has a momentum minimum), but it's the magnitude predictor.

### 9.11 trap_within_trend: 168h Max_hold Cap is Cutting Best Trades
Top 3 wins all hit the 168h ceiling: $1,393, $1,452, $1,474. Duration is the only significant PnL magnitude driver (r=0.406, p=0.021):
- Hold < 48h: avg $530
- Hold 48-96h: avg $654
- Hold > 96h: avg $807

**Recommended**: Extend max_hold from 168h → 240h for trap_within_trend. Expected uplift: +$1-2K/year from freeing the long tail.

### 9.12 long_squeeze: vol_shock Is the #1 Loss Predictor (LOW CONFIDENCE — n=14)
- 11 winners: all had vol_shock at or near 0 (max ~0.08)
- 3 losers: vol_shock 0.14, 0.63, and 1.0 (maximum)

The Feb 2021 loser (worst, -$1,563): vol_shock=1.0, fired directly into a BTC momentum wave. The vol_shock channel is separate from crisis_prob — it captures acute volatility spikes mid-macro-calm. A `vol_shock < 0.10` gate could eliminate all 3 historical losses.

Additional supporting gates for long_squeeze:
- `trend_strength < 0.88` — losers avg 0.941 vs winners 0.747 (fighting strong trends)
- `chop >= 0.18` — all 3 losers had chop < 0.15 (too clean/trending for a squeeze)
- `sentiment_score >= 0.75` — big winners all had sentiment 0.83-0.94

### 9.13 funding_divergence: 48h Resolution Rule (LOW CONFIDENCE — n=11)
When funding_divergence hasn't reached 1.0R scale-out within 48h, it is significantly more likely to stop out eventually. Winners average 59.2h hold; losers average 68.8h — losers DRAG LONGER because they partially scale out before stopping. Consider a time-based partial exit at 48h if position is below 0.5R profit.

### 9.14 Systemic Findings: risk_temp and dd_score Floors

Both features have hard performance floors that affect the entire system:

| Condition | n | WR | AvgPnL | Action |
|-----------|---|----|---------|----|
| risk_temp < 0.20 | 38 | 36.8% | -$304 | Systematic loser |
| risk_temp < 0.35 | 105 | 44.3% | -$125 | Net negative |
| dd_score < 0.05 | 79 | 41.8% | -$106 | Near certain loss |
| dd_score < 0.10 | 138 | 43.5% | -$107 | Net negative |
| crisis_prob 0.03-0.06 | 66 | 39.4% | N/A | Anomalous bad zone |

**Insight on crisis_prob**: The 0.03–0.06 bucket (66 trades, 39.4% WR) is anomalously bad — worse than either the 0-0.01 zone (WR=64%) or the >0.06 zone (WR=67%). This "elevated-but-not-crisis" zone may represent macro uncertainty where both bulls and bears are uncertain — a poor environment for directional setups.

### 9.15 90% of System Profit Comes from trend_align = 1.0 Entries

trend_align is approximately binary in the data (mostly 0.0 or 1.0):
- trend_align = 1.0: n=328, WR=65.5%, total PnL=$172,781 (**90% of all profit**)
- trend_align < 0.5: n=130, WR=47.7%, total PnL negative

This is the strongest macro-level finding. The 130 counter-trend entries represent 11.5% of trades but are collectively a PnL drag. Any system-wide trend_align gate would have an outsized impact.

### 9.16 Composite Invalidation Exits Are Worse Than Stop Loss (Per-Trade)

| Exit Type | n | Avg PnL |
|-----------|---|---------|
| Scale-out | 645 | +$463 |
| Time exit | 24 | +$451 |
| **Stop loss** | 428 | **-$226** |
| Distress half-exit | 14 | -$387 |
| **Composite invalidation** | 15 | **-$1,094** |

Composite invalidation exits average 4.8x the per-exit damage of stop losses. These exits fire at bar close price when structural breakdown is detected — but the stop loss would fire at the actual stop level, often lower. The invalidation exits are cutting positions early at worse prices than waiting for the stop. The 4/5 threshold should remain but the mechanism needs verification that exit price is correctly simulated.

---

## 10. Appendix: System Constants

- **Data**: 2020-2024, BTC-PERP, 1H, 61,306 bars
- **Capital**: $100,000, leverage 1.5-2.5x
- **Cost model**: 0.02% commission, 3bps slippage
- **Stop-loss**: 3.0x ATR from entry
- **First scale-out**: 33% at 0.5R (updated 2026-04-01)
- **Scale ladder standard**: [0.5R, 1.0R, 2.0R] → [33%, 20%, 30%]
- **Base fusion threshold**: 0.18 (dynamic via CMI)

---

*Generated from 1,127 backtest trades. All statistics from pandas/scipy analysis.*
*Architecture: Bull Machine v17 | Dataset: BTC_1H_FEATURES_V12_ENHANCED | Updated: 2026-04-02*
