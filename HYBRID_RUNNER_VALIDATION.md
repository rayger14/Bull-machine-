# Hybrid Runner Validation Checklist

**Purpose**: Validate execution parity between batch backtests and bar-by-bar hybrid_runner before production deployment.

**Philosophy**: The hybrid_runner exercises all stateful bits that batch backtests can "average out" (cooldown, regime flips, partial exits, macro veto timing).

---

## Why Bar-by-Bar Validation Matters

Bar-by-bar (event-driven) is closest to production because it tests:

✅ **Stateful Components**:
- Cooldown windows (entry blocking after recent trade)
- Rolling ATR updates (20-bar lookback)
- Regime flips mid-sequence (VIX crossing 20→25)
- Partial exits (50% at 1.0R, trail remainder)
- Trailing stops (ATR-based with regime adaptation)

✅ **Temporal Ordering**:
- Macro veto timing (veto toggles between bars)
- Config switching (VIX-policy: VIX<18=aggressive, >22=conservative)
- ML fusion patches applied at decision time with only past data

✅ **No Look-Ahead**:
- Features computed causally (only data ≤ current bar)
- Macro signals snapshot at bar timestamp
- Model artifact frozen (no retraining mid-backtest)

---

## Config Compatibility Fix ✅ DONE

**File**: `utils/config_compat.py`

**Problem**: Naming mismatch ("liquidity" vs "hob") between configs and engine

**Solution**: Alias map in loader without refactoring engine

```python
from utils.config_compat import normalize_config_for_hybrid

cfg = json.load(Path(config_path).open())
cfg = normalize_config_for_hybrid(cfg)  # Apply aliases + validate weights
```

**Status**: ✅ Tested with `BTC_live.json` - passes validation

---

## Quick Bar-by-Bar Validation Plan

### Step 1: Freeze Inputs
- Same candles (data/features/v18/BTC_1H.parquet)
- Same macro cache (VIX, MOVE, DXY snapshots)
- Same ML model artifact (LightGBM hash recorded)
- Fixed random seed (if any stochastic components)

### Step 2: Two Runs on Q3 2024 (July 1 - Sept 30)

**A. Batch Mode** (Current):
```bash
python3 bin/optimize_v19.py \
  --asset BTC \
  --mode custom \
  --config configs/v18/BTC_live.json \
  --start 2024-07-01 \
  --end 2024-09-30 \
  --output q3_batch.json
```

**B. Hybrid Mode** (Bar-by-Bar):
```bash
python3 bin/live/hybrid_runner.py \
  --asset BTC \
  --start 2024-07-01 \
  --end 2024-09-30 \
  --config configs/v18/BTC_live.json \
  --output q3_hybrid.json
```

### Step 3: Acceptance Criteria

Compare `q3_batch.json` vs `q3_hybrid.json`:

| Metric | Tolerance | Pass/Fail |
|--------|-----------|-----------|
| Trade Count | Δ ≤ 5% | |
| Win Rate | Δ ≤ 2pp | |
| Profit Factor | Δ ≤ 0.05 | |
| Total Return | Δ ≤ 3% | |
| Equity Curve RMSE | ≤ 0.35% of balance | |
| Max Drawdown | Δ ≤ 2% | |

**If all pass**: Extend to full-year 2024

**If any fail**: Debug decision log to find divergence point

---

## Hybrid Runner Acceptance Checklist

### ✅ No Look-Ahead
- [ ] ML features use only data ≤ current bar
- [ ] Macro signals snapshot at bar timestamp (no "future" macro ticks)
- [ ] Model artifact frozen (no retraining mid-backtest)
- [ ] VIX/MOVE/DXY values loaded from historical cache, not live

### ✅ Warmup Period
- [ ] Enforce warmup bars for ATR (20 bars minimum)
- [ ] Enforce warmup for volatility modules (20 bars)
- [ ] Enforce warmup for temporal ACF/LPPLS (100 bars minimum)
- [ ] Forbid entry signals during warmup

### ✅ Stop Logic Parity
- [ ] Clarify on-touch vs close-through for OB/pHOB
- [ ] ATR stops consistent with batch (use close-through)
- [ ] Trailing stops update bar-by-bar, not tick-by-tick

### ✅ Sizing Parity
- [ ] Same risk model (dynamic ADX-based vs fixed fractional)
- [ ] Same max position size (20% of capital)
- [ ] Same max margin utilization (50%)

### ✅ Fees/Slippage Parity
- [ ] Identical fee model (10 bps maker + 5 bps slippage)
- [ ] Applied on entry and exit
- [ ] No hidden costs in hybrid that batch doesn't have

### ✅ Macro Caching
- [ ] VIX, MOVE, DXY, Oil, Gold, Yields loaded from historical files
- [ ] Timestamps aligned to 1H bar closes
- [ ] No forward-looking macro data (interpolation if missing)

### ✅ Performance
- [ ] Per-bar compute ≤ 50ms target (1H bars allow more than 1min data)
- [ ] Cache LPPLS/ACF or precompute in feature store
- [ ] No blocking I/O during simulation

### ✅ Determinism
- [ ] Log model artifact hash (LightGBM file MD5)
- [ ] Log config hash (JSON content hash)
- [ ] Log code version (git SHA)
- [ ] Same inputs → same outputs (reproducible)

### ✅ Audit Log
For every bar, log structured decision:
```python
decision_log.append({
  "ts": bar_ts,
  "signal": signal or "none",
  "fusion": round(fusion_score, 3),
  "breakdown": {"wyckoff": w, "liquidity": l, "momentum": m, "macro": ma},
  "vetoes": active_veto_tags,  # ["VIX>30+DXY>105", "YieldInversion"]
  "config_id": cfg_id,          # file + patch hash
  "cooldown": cooldown_active,
  "position_state": pos_state,  # flat/long/short + size
  "macro_snapshot": {"VIX": vix, "MOVE": move, "DXY": dxy}
})
```

---

## Minimal Instrumentation Required

Add to `bin/live/hybrid_runner.py`:

```python
# At top
decision_log = []
model_hash = hashlib.md5(Path('models/fusion_optimizer.lgb').read_bytes()).hexdigest()
config_hash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

# Per bar
decision_log.append({
    "ts": bar_ts,
    "signal": signal,
    "fusion": fusion_score,
    "breakdown": fusion_breakdown,
    "vetoes": active_vetoes,
    "config_id": f"{cfg_id}_{config_hash[:8]}",
    "cooldown": cooldown_active,
    "position": position_state,
    "macro": {"VIX": vix, "MOVE": move, "DXY": dxy}
})

# At end
with open(f"{output_prefix}_decisions.json", 'w') as f:
    json.dump({
        "model_hash": model_hash,
        "config_hash": config_hash,
        "git_sha": git_sha,
        "decisions": decision_log
    }, f, indent=2)
```

---

## When to Use Batch vs Hybrid

### Use Batch For:
- ✅ Exploration (sweeping 100s of configs)
- ✅ Sensitivity studies (threshold ladders, weight grids)
- ✅ Research on new alpha ideas (faster iteration)

### Use Hybrid For:
- ✅ Final validation before production
- ✅ Execution parity testing
- ✅ Config switching logic (VIX-policy)
- ✅ ML fusion patch application timing
- ✅ Paper trading simulation

---

## Run Order Recommended

1. **Q3 2024 Hybrid vs Batch** → Compare (35s each)
   - Target: ≤5% trade count Δ, ≤2pp WR Δ, ≤0.05 PF Δ

2. **Full 2024 Hybrid vs Batch** → Compare (5-7 min)
   - Confirm parity holds over longer period

3. **2022-2025 Hybrid** (one pass) → Produce equity curve (46 min)
   - Generate $10k → $? P&L, PF, WR, Sharpe
   - This is the "production simulation"

4. **VIX-Policy Config Switching** → Re-run 2024 (5-7 min)
   - VIX <18: ETH_live_aggressive.json
   - VIX 18-22: BTC_live.json
   - VIX >22: ETH_live_conservative.json or pause

---

## Parity Assertion Script (Optional CI Check)

**File**: `tests/test_parity.py`

```python
import pandas as pd
import pytest

def test_q3_2024_parity():
    batch = pd.read_json('q3_batch.json')
    hybrid = pd.read_json('q3_hybrid.json')

    # Trade count
    assert abs(len(batch) - len(hybrid)) / len(batch) <= 0.05, "Trade count Δ >5%"

    # Win rate
    batch_wr = (batch['pnl'] > 0).mean() * 100
    hybrid_wr = (hybrid['pnl'] > 0).mean() * 100
    assert abs(batch_wr - hybrid_wr) <= 2.0, f"WR Δ >2pp: {batch_wr:.1f}% vs {hybrid_wr:.1f}%"

    # Profit factor
    batch_pf = batch[batch['pnl'] > 0]['pnl'].sum() / abs(batch[batch['pnl'] < 0]['pnl'].sum())
    hybrid_pf = hybrid[hybrid['pnl'] > 0]['pnl'].sum() / abs(hybrid[hybrid['pnl'] < 0]['pnl'].sum())
    assert abs(batch_pf - hybrid_pf) <= 0.05, f"PF Δ >0.05: {batch_pf:.3f} vs {hybrid_pf:.3f}"

    # Equity curve RMSE
    batch_equity = batch['pnl'].cumsum()
    hybrid_equity = hybrid['pnl'].cumsum()
    rmse = ((batch_equity - hybrid_equity) ** 2).mean() ** 0.5
    assert rmse <= 35.0, f"Equity RMSE >{rmse:.2f} (>0.35% of $10k)"

    print("✅ All parity checks passed!")
```

Run with: `pytest tests/test_parity.py -v`

Fail CI if tolerance exceeded → forces fixing execution divergence before merge

---

## Current Status

✅ **Config compatibility**: DONE (`utils/config_compat.py`)
✅ **Phase 1 ML**: DONE (fusion optimizer + enhanced macro)
⏳ **Hybrid validation**: PENDING (needs Q3 2024 run)

**Next Steps**:
1. Run Q3 2024 hybrid_runner (35s)
2. Compare to batch results
3. If parity passes → Phase 1 COMPLETE with execution validation
4. If parity fails → Debug decision log, fix divergence

---

## Expected Timeline

- Config compat: ✅ DONE (10 min)
- Q3 2024 hybrid run: 35 seconds
- Parity comparison: 5 minutes
- Full 2024 hybrid run: 5-7 minutes
- Documentation: 15 minutes

**Total: ~30 minutes to full hybrid validation**

---

**Document Version**: 1.0
**Status**: Ready for Q3 2024 validation
**Last Updated**: 2025-10-14
