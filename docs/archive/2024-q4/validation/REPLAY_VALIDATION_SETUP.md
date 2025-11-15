# Replay Validation Setup - MVP to Live Path

## Status: Ready for Hybrid Replay Testing

Generated: 2025-10-19

---

## 1. Frozen Best Configs (v3 Replay 2024)

### BTC - Best v3 Config
- **File:** `configs/v3_replay_2024/BTC_2024_best.json`
- **Source:** Rank #1 from `reports/optuna_results/BTC_knowledge_v3_best_configs.json`
- **Expected Metrics (2024):**
  - PNL: $1,940.26
  - Trades: 17
  - Source period: 2024-01-01 to 2024-12-31

### ETH - Best v3 Config
- **File:** `configs/v3_replay_2024/ETH_2024_best.json`
- **Source:** Rank #1 from `reports/optuna_results/ETH_knowledge_v3_best_configs.json`
- **Expected Metrics (2024):**
  - PNL: $1,952.73
  - Trades: 29
  - Source period: 2024-01-01 to 2024-12-31

### SPY - Equity-Tuned Best Config
- **File:** `configs/v3_replay_2024/SPY_2024_equity_tuned.json`
- **Source:** Rank #1 from `reports/optuna_results/SPY_equity_tuned_best_configs.json`
- **Expected Metrics (2024):**
  - PNL: $774.24
  - Trades: 4
  - Source period: 2024-01-01 to 2024-12-31

---

## 2. Feature Stores (2024 Full Year)

### BTC Feature Store
- **File:** `data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet`
- **Size:** 1.4M
- **Last Built:** Oct 19 03:32 (latest rebuild with M1/M2 Wyckoff)
- **Features:** 69 columns (1D→4H→1H MTF, Wyckoff, SMC/HOB, FRVP, Momentum, PTI, Macro)

### ETH Feature Store
- **File:** `data/features_mtf/ETH_1H_2024-01-01_to_2024-12-31.parquet`
- **Size:** 1.4M
- **Last Built:** Oct 19 03:32 (latest rebuild with M1/M2 Wyckoff)
- **Features:** 69 columns (1D→4H→1H MTF, Wyckoff, SMC/HOB, FRVP, Momentum, PTI, Macro)

### SPY Feature Store
- **File:** `data/features_mtf/SPY_1H_2024-01-01_to_2024-12-31.parquet`
- **Size:** 134K
- **Last Built:** Oct 19 17:05 (latest rebuild with M1/M2 Wyckoff + adaptive max-hold)
- **Features:** 69 columns (1D→4H→1H MTF, Wyckoff, SMC/HOB, FRVP, Momentum, PTI, Macro)

---

## 3. Validation Gates (Acceptance Criteria)

### Parity Check (Replay vs Backtest)
For each asset replay:
- **PNL:** Within ±2-5% of backtest PNL for same period
- **Trade Count:** Within ±20% of backtest trade count
- **Exit Reasons Mix:** Similar distribution (max_hold vs trailing vs neutralization)
- **Timestamps:** Entry/exit times roughly align with backtest
- **Adaptive Events (SPY only):** Extension count > 0, events logged to JSON

### Bar-by-Bar Sanity (Visual)
- Run 1 week sample per asset at 60x speed with debug prints enabled
- Verify fusion score, tier assignment, wyckoff_phase, pti, macro visible
- Confirm adaptive max-hold triggers on SPY (show effective_max_hold in logs)
- Check MTF alignment (1D→4H→1H causal forward-fill)

### Shadow-Live Week Test
- Run 1 full week in shadow mode (live feed, no capital)
- Log all trades, fusion scores, regime changes
- After week ends, backtest the same week with same code+config
- Compare metrics: PNL, trade count, win rate, max DD
- **Acceptance:** Live-vs-backtest drift ≤ 5%

---

## 4. Code Freeze (Tag: v1.9.1-replay)

### Core Engine
- `bin/backtest_knowledge_v2.py` - KnowledgeAwareBacktest with adaptive max-hold
- `lib/features/mtf_feature_builder_v2.py` - MTF feature store builder (M1/M2 Wyckoff)
- `lib/knowledge/smc_hob_v2.py` - SMC/HOB/BOMS/CHOCH/FVG detection
- `lib/knowledge/frvp_detector.py` - FRVP/POC/VA/HVN analysis
- `lib/knowledge/m1m2_wyckoff_detector.py` - Wyckoff M1/M2 spring/markup detection
- `lib/knowledge/pti_detector.py` - PTI (Policy Trend Indicator)
- `lib/knowledge/macro_pulse_detector.py` - Macro Echo/Pulse regime detection

### Hybrid Runner (Next: To Be Implemented)
- `bin/live/hybrid_runner.py` - 60x replay mode + shadow-live mode
- Must support:
  - `--mode replay` (tick through feature store at 60x speed)
  - `--mode shadow` (live feed, shadow trading, no capital)
  - `--features <parquet_path>` (load cached features for replay)
  - `--config <json_path>` (frozen config params)
  - `--speed <multiplier>` (e.g., 60x for replay)

---

## 5. Next Steps (Replay Validation Path)

### Step 1: Implement Hybrid Replay Runner ⏳
- Add `--mode replay` flag to `bin/live/hybrid_runner.py`
- Tick source: iterate rows of feature store in timestamp order
- Sleep multiplier: 1/60 for 60x speed (or disabled, just advance time per bar)
- On each tick: compute fusion → hooks → entries/exits exactly like live
- Write same trade/telemetry logs as live mode

### Step 2: Run 60x Replay Tests 📊
```bash
# BTC 2024 Replay
python3 bin/live/hybrid_runner.py \
  --asset BTC \
  --from 2024-01-01 --to 2024-12-31 \
  --features data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet \
  --config configs/v3_replay_2024/BTC_2024_best.json \
  --mode replay --speed 60x \
  --shadow true --log reports/replay/BTC_2024_60x.log

# ETH 2024 Replay
python3 bin/live/hybrid_runner.py \
  --asset ETH \
  --from 2024-01-01 --to 2024-12-31 \
  --features data/features_mtf/ETH_1H_2024-01-01_to_2024-12-31.parquet \
  --config configs/v3_replay_2024/ETH_2024_best.json \
  --mode replay --speed 60x \
  --shadow true --log reports/replay/ETH_2024_60x.log

# SPY 2024 Replay
python3 bin/live/hybrid_runner.py \
  --asset SPY \
  --from 2024-01-01 --to 2024-12-31 \
  --features data/features_mtf/SPY_1H_2024-01-01_to_2024-12-31.parquet \
  --config configs/v3_replay_2024/SPY_2024_equity_tuned.json \
  --mode replay --speed 60x \
  --shadow true --log reports/replay/SPY_2024_60x.log
```

### Step 3: Validate Parity ✅
- Compare replay results to backtest expected metrics
- Check acceptance gates (PNL ±5%, trade count ±20%, etc.)
- Diff logs for any mismatched trades
- Fix ordering/rounding issues if found

### Step 4: Bar-by-Bar Sanity (1 Week Sample) 🔍
- Run Nov 1-7, 2024 per asset with debug prints ON
- Verify MTF features, fusion, tier, adaptive logic visible
- Confirm no look-ahead, proper regime transitions

### Step 5: Shadow-Live Week Test 🕵️
- Flip to `--mode shadow` (live feed, current week)
- Run for 1 full week (no capital, logs only)
- After week ends, backtest the same week
- Compare live vs backtest: drift ≤ 5%

### Step 6: Tiny-Capital Go-Live 💸 (Optional)
- If shadow passes, enable tiny capital (0.25-0.5% risk per trade)
- Kill switch: if live PF < 1.2 or max DD > 2%, stop and analyze
- Monitor for 1-2 weeks before scaling

---

## 6. Domain Verification Status

All domains wired and validated:

✅ **Wyckoff (M1/M2):** Advanced detector implemented, phases vary (markup, spring, markdown, transition, B), M1/M2 signals detected
✅ **SMC/HOB:** Order blocks, BOMS, CHOCH, FVG detection present (note: BOMS/CHOCH/FVG rare by design)
✅ **FRVP:** POC, VA, HVN, LVN detection, volume profile levels computed
✅ **Momentum:** ADX, RSI, Squiggle integrated in fusion scoring
✅ **PTI (Policy Trend Indicator):** Wired, varies in volatile regimes (confirmed 2022 bear test)
✅ **Macro (Echo/Pulse):** Macro sentiment detection, fixed VIX levels, equity-tuned configs reduce over-filtering
✅ **MTF Downcast:** 1D→4H→1H verified causal, forward-fill rules correct, no look-ahead

---

## 7. Risk Considerations

### Drawdown Management
- **Current configs:** Optimized for 2024 bull market
- **2025 risk:** If market regime shifts (bear/choppy), configs may need re-tuning
- **Mitigation:** Monitor live DD closely; if live DD > 2x backtest DD, pause and re-optimize

### Adaptive Max-Hold (SPY)
- **Current:** SPY adaptive logic extends holds 24h → 48h/72h in markup phases
- **Risk:** If market becomes choppy, extensions may hurt
- **Mitigation:** Instrumentation logs all extension events; can disable `adaptive_max_hold=False` if needed

### Parity Drift
- **Expected:** Replay should match backtest within ±5% PNL
- **If drift > 5%:** Investigate ordering, float rounding, or exit sequence differences
- **Common causes:** Position sizing path, stop-loss rounding, or partial exit timing

---

## 8. Feature Store Metadata

All stores built with:
- **MTF Downcast:** 1D Wyckoff → 4H → 1H (causal forward-fill)
- **Wyckoff M1/M2:** Advanced detector, spring/markup signals
- **SMC/HOB:** Order blocks, BOMS, CHOCH, FVG (rare events expected)
- **FRVP:** Volume profile, POC, VA, HVN, LVN
- **Momentum:** ADX, RSI, Squiggle
- **PTI:** Policy Trend Indicator (activity varies by regime)
- **Macro:** Echo/Pulse (fixed VIX for 2024 equity-tuned configs)

Columns per store: **69 features**

---

## 9. Code Tag (for replay/backtest parity)

**Tag:** `v1.9.1-replay`
- Adaptive max-hold logic implemented (bin/backtest_knowledge_v2.py:385-442)
- Instrumentation added (AdaptiveHoldEvent dataclass, JSON logging)
- SPY test script fixed (bin/test_spy_adaptive_maxhold.py uses correct 2024-12-31 file)
- Best configs frozen (configs/v3_replay_2024/)

**Commit message:**
```
feat(mvp): freeze v3 configs and feature stores for replay validation

- Extract rank #1 configs: BTC v3, ETH v3, SPY equity-tuned
- Freeze 2024 feature stores (BTC, ETH, SPY with M1/M2 Wyckoff)
- Tag v1.9.1-replay for parity testing
- Ready for hybrid replay runner implementation
```

---

## 10. Success Metrics (Definition of Done)

Replay validation PASSES when:
1. ✅ All 3 assets (BTC, ETH, SPY) replay at 60x with PNL within ±5% of backtest
2. ✅ Trade counts within ±20% of backtest expectations
3. ✅ Exit reason distribution matches (no unexpected max-hold or stop-loss floods)
4. ✅ Bar-by-bar sanity shows MTF + adaptive logic executing correctly
5. ✅ Shadow-live week test shows <5% drift vs backtest of same week
6. ✅ Adaptive max-hold events logged for SPY (count > 0)

When all gates pass → Ready for tiny-capital go-live testing.

---

**Status:** Config freeze complete ✅ | Next: Implement hybrid replay runner
