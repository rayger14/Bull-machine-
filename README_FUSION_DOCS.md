# Bull Machine Fusion & Threshold Documentation

**Created**: 2026-03-11  
**Updated**: 2026-03-11

Three comprehensive guides document how the Bull Machine fusion scoring and dynamic threshold system works. This index helps you find the right document for your need.

## Quick Navigation

### I want to understand HOW the system works
**→ Read: `FUSION_THRESHOLD_ARCHITECTURE.md`** (15 KB, technical deep-dive)

This is the authoritative technical specification covering:
- Feature store analysis (9 columns with statistics)
- 6-step fusion score pipeline
- CMI v0 (Contextual Market Intelligence) formulas
- Dynamic threshold computation
- 3 worked examples (bull/bear/crisis scenarios)
- Per-archetype thresholds

**Best for**: Understanding system internals, debugging complex issues, academic reference

### I need a quick formula or reference
**→ Read: `FUSION_QUICK_REFERENCE.md`** (5.9 KB, one-page cheat sheet)

This compact reference includes:
- Core formulas (fusion score, threshold)
- Per-archetype threshold table
- Risk temperature & instability components
- Feature requirements
- Emergency protections
- Quick debugging checklist

**Best for**: Quick lookups while coding, debugging, ad-hoc analysis

### I need to change system behavior
**→ Read: `CMI_CONFIG_TUNING_GUIDE.md`** (9.1 KB, operational guide)

This step-by-step guide covers:
- Parameter locations in config
- 4 core parameter groups with examples
- 4 tuning scenarios (permissive/selective/bear/crisis)
- Implementation steps
- Log monitoring patterns
- Validation checklist

**Best for**: Adjusting system performance, tuning for specific market conditions

---

## Key Architecture Summary

### Fusion Score Pipeline (6 Steps)

```
[Feature Store] → [Pattern Detection] → [Runtime Boosts] → [Crisis Penalty] → [Threshold Filter]
  0.3-0.65        gates (veto)         wyckoff/smc/    crisis_penalty      arch_threshold
                                       temporal (0.8-1.8x) (0.6-1.0x)         0.25-0.75+
```

### Dynamic Threshold Formula

```
arch_threshold = base_arch_threshold 
               + (1 - risk_temp) * temp_range        [bear penalty]
               + instability * instab_range           [chop penalty]

Range: 0.25 (bull) to 0.75+ (crisis)
```

### Three Market Conditions

| Condition | risk_temp | instability | crisis_prob | threshold | Result |
|-----------|-----------|-----------|-----------|-----------|--------|
| **Bull** | 0.80 | 0.20 | 0.01 | 0.28 | Permissive ✓ |
| **Bear** | 0.20 | 0.60 | 0.16 | 0.60 | Selective ✗ |
| **Crisis** | 0.05 | 0.85 | 0.90 | 0.72+ | Emergency ✗✗✗ |

---

## Per-Archetype Thresholds

| Archetype | Base | Rationale |
|-----------|------|-----------|
| retest_cluster | 0.12 | Sparse pattern, low gates → needs help |
| trap_within_trend | 0.15 | Moderate gates, high quality |
| liquidity_vacuum | 0.15 | Moderate gates, high quality |
| failed_continuation | 0.15 | Moderate gates, high quality |
| liquidity_sweep | 0.15 | Moderate gates, moderate PF |
| wick_trap | 0.18 | Frequent signals, highest gates |

---

## Core Parameters to Adjust

### Primary (Most Impact)
- `per_archetype_base_threshold` — Signal frequency per archetype
- `temp_range` — Bear market penalty (0.35-0.55)
- `crisis_coefficient` — Crisis impact on fusion (0.3-0.5)

### Secondary (Fine-tuning)
- `instab_range` — Choppy market penalty (0.10-0.25)
- `emergency_crisis_threshold` — When to cap sizing
- `emergency_size_multiplier` — How much to reduce sizing

### Tertiary (Advanced)
- `cmi_weights` — Rebalance components of risk_temperature
- `base_max_positions` — Concurrent position limit

---

## Configuration Location

All parameters in: `configs/bull_machine_isolated_v11_fixed.json`

```json
{
  "adaptive_fusion": {
    "enabled": true,
    "base_threshold": 0.18,
    "per_archetype_base_threshold": {
      "trap_within_trend": 0.15,
      "liquidity_vacuum": 0.15,
      "retest_cluster": 0.12,
      "wick_trap": 0.18,
      "failed_continuation": 0.15,
      "liquidity_sweep": 0.15
    },
    "temp_range": 0.48,
    "instab_range": 0.15,
    "crisis_coefficient": 0.4,
    "cmi_weights": {
      "trend_align": 0.45,
      "trend_strength": 0.25,
      "sentiment_score": 0.15,
      "dd_score": 0.10,
      "derivatives_heat": 0.05
    }
  }
}
```

---

## Variant A Change (2026-02-12)

**What**: temp_range increased from 0.40 to 0.48 (+20%)

**Why**: Surgically filter marginal bear market trades

**Effect**:
- Bear thresholds increased by +0.08
- 43 of 58 losing trades killed (74% loss reduction)
- Only 2 winners affected (0.3% impact)
- Profit Factor: 1.80 → 1.82 (+1%)
- Sharpe: 1.56 → 1.67 (+7%)

---

## Quick Start: Adjust System Behavior

### Too Many Signals (Too Permissive)
```bash
# Increase selectivity
↑ per_archetype_base_threshold (+0.02-0.03)
↑ temp_range (+0.05-0.10)
↑ crisis_coefficient (+0.1)
```

### Too Few Signals (Too Selective)
```bash
# Increase permissiveness
↓ per_archetype_base_threshold (-0.02-0.03)
↓ temp_range (-0.05-0.10)
↓ crisis_coefficient (-0.05)
```

### Losing Money in Bear Markets
```bash
# Stronger bear penalties
↑ temp_range (+0.05-0.10)
↑ crisis_coefficient (+0.1)
↓ emergency_crisis_threshold (-0.05-0.10)
```

### Max Drawdown Too High
```bash
# More risk-averse
↑ crisis_coefficient (+0.1)
↓ emergency_size_multiplier (-0.10-0.20)
↓ base_max_positions (-1)
```

---

## Testing Workflow

1. **Edit config**
   ```bash
   vim configs/bull_machine_isolated_v11_fixed.json
   ```

2. **Validate JSON**
   ```bash
   jq . < configs/bull_machine_isolated_v11_fixed.json
   ```

3. **Quick test (Q1 2023)**
   ```bash
   python3 bin/backtest_v11_standalone.py \
     --config configs/bull_machine_isolated_v11_fixed.json \
     --data data/btc_1h_2023_Q1.csv 2>&1 | tee backtest.log
   ```

4. **Check metrics**
   ```bash
   grep "Total signals:\|Total Trades:\|Profit Factor:" backtest.log
   ```

5. **Expected baseline for Q1 2023**
   - Trades: ~10-15
   - Profit Factor: ~1.8
   - Win Rate: ~82%

---

## Feature Requirements

CMI v0 uses 9 critical features (100% coverage required):

✓ `price_above_ema_50` — EMA-based trend  
✓ `ema_50_above_200` — EMA-based trend  
✓ `adx` / `adx_14` — Momentum  
✓ `fear_greed_norm` — Sentiment [0,1]  
✓ `drawdown_persistence` — Drawdown context [0,1]  
✓ `rv_20d` — Realized volatility  
✓ `chop_score` — Choppiness  
✓ `wick_ratio` — Wick extremes  
✓ `volume_z_7d` — Volume extremes  

**DO NOT USE**: `sma_50`, `sma_200` (NaN in 2020-2021)

---

## Production Baseline (OOS 2023-2024)

Current system with 6 archetypes, per-arch thresholds, 3x leverage:

| Metric | Value |
|--------|-------|
| **Profit Factor** | 1.82 |
| **Return** | +73.8% |
| **Sharpe** | 1.67 |
| **Max Drawdown** | -9.1% |
| **Total Trades** | 656 |
| **Win Rate** | 82.0% |

Full period (2020-2024) with 1.5x leverage:
- Profit Factor: 1.48
- Return: +102.6%
- Sharpe: 0.96
- Trades: 1,586

---

## Debug Log Patterns

Monitor backtest logs for these key lines:

```
[THRESHOLD] bar=... | dynamic_threshold=X.XXX | risk_temp=X.XXX | instability=X.XXX | crisis_prob=X.XXX

[SIGNALS] bar=... | raw=... | post_filter=... | post_threshold=... | passed=...

[FILTER] archetype rejected: fusion=X.XXX*penalty=X.XXX=X.XXX < threshold=X.XXX

[TRADE_ENTRY] bar=... | archetype | fusion=X.XXX | threshold=X.XXX
```

---

## Document Structure

```
README_FUSION_DOCS.md (this file)
├─ FUSION_THRESHOLD_ARCHITECTURE.md
│  └─ 6,500+ words, technical deep-dive
│     ├─ Feature store analysis
│     ├─ Fusion score pipeline (6 steps)
│     ├─ CMI v0 breakdown
│     ├─ Threshold formulas
│     └─ 3 worked examples
│
├─ FUSION_QUICK_REFERENCE.md
│  └─ Compact cheat sheet
│     ├─ Core formulas
│     ├─ Parameter tables
│     ├─ Threshold examples
│     └─ Debugging checklist
│
└─ CMI_CONFIG_TUNING_GUIDE.md
   └─ Operational guide
      ├─ Parameter reference
      ├─ 4 tuning scenarios
      ├─ Implementation steps
      └─ Validation checklist
```

---

## Quick Links

- **Backtest engine**: `bin/backtest_v11_standalone.py` (CMI v0 at line 560-700)
- **Archetype logic**: `engine/archetypes/logic_v2_adapter.py` (fusion computation)
- **Config file**: `configs/bull_machine_isolated_v11_fixed.json`
- **Feature store**: `data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet` (283 cols)

---

## FAQ

**Q: Why different thresholds per archetype?**  
A: Low-gate archetypes (retest_cluster) need help to reach min quality. High-gate archetypes (wick_trap) need selectivity to avoid oversaturation.

**Q: Why does risk_temperature matter?**  
A: It drives the primary dynamic adjustment to thresholds. 0=bear (high threshold), 1=bull (low threshold).

**Q: Why is instability separate from risk_temperature?**  
A: risk_temp measures bias/direction. instability measures noise/chop. Both independent, both matter.

**Q: What's the Variant A change?**  
A: temp_range increased from 0.40 to 0.48, increasing bear penalties by 20% to surgically kill marginal bear trades.

**Q: What happens in a crisis?**  
A: Threshold rises (harder to pass) + fusion scores reduced by crisis_penalty (weaker signals) + emergency sizing cap (0.5x).

---

**Last Updated**: 2026-03-11  
**Maintainer**: Claude AI  
**Status**: Complete and production-ready
