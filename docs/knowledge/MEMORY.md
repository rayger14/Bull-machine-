# Bull Machine Project Memory

## User Preferences
- [Backtest output format](feedback_backtest_output.md) — always show date range, starting equity, avg risk per trade

## Key Architecture

- **v17 Whale Footprint Architecture**: 16+1 archetypes + hard_gates YAML + 301-col feature store + Optuna ATR + dedup + heuristic fixes + whale conflict penalty + $100K
- **Standalone Backtester**: `bin/backtest_v11_standalone.py` - primary backtest engine (--commission-rate 0.0002, --slippage-bps 3, --start-date, --initial-cash 100000)
- **Config**: `configs/bull_machine_isolated_v11_fixed.json` - production config (dynamic threshold adaptive fusion)
- **Feature Store**: `data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet` (61,306 bars x 301 cols, 2018-2024, +8 Binance OI/funding/LS/taker features)
- **Symlink**: `BTC_1H_LATEST.parquet` → BTC_1H_FEATURES_V12_ENHANCED.parquet (SM-REBUILT)
- **Hard Gates**: `configs/archetypes/*.yaml` each has `hard_gates:` section, enforced by `archetype_instance.py:_check_hard_gates()`
- **ML Retraining**: DISABLED (AUC=0.585, near-random)

## Current Results (2020-2024, Post-Optuna Gate Optimization, $100K)

| Metric | Pre-Optuna Gates | **Post-Optuna Gates** | Change |
|--------|-----------------|----------------------|--------|
| **PF** | 1.68 | **2.06** | **+23%** |
| **PnL** | $111K | **$111K** | Same |
| **Trades** | 649 | **476** | -27% |
| **Win Rate** | 79.0% | **81.5%** | +2.5pp |
| **MaxDD** | -7.9% | **-6.3%** | +20% better |
| **Sharpe** | 1.17 | **1.33** | +14% |

### Per-Archetype (Post-Optuna, ranked by PnL)
| Archetype | Trades | PF | PnL | Direction |
|---|---|---|---|---|
| liquidity_sweep | 98 | 2.21 | $26.6K | long |
| retest_cluster | 117 | 1.92 | $23.1K | long |
| wick_trap | 109 | 1.59 | $21.9K | long |
| failed_continuation | 33 | 13.47 | $13.1K | long |
| trap_within_trend | 18 | 3.54 | $8.1K | long |
| liquidity_vacuum | 12 | inf | $7.1K | +$0.3K | long |
| funding_divergence | 18 | 2.61 | $6.4K | -$0.7K | long |
| fvg_continuation | 51 | 1.36 | $3.8K | +$0.3K | long |
| spring | 19 | 1.62 | $3.7K | **+$5.5K** (was -$1.8K) | long |
| oi_divergence | 2 | 0.04 | -$2.8K | — | long |
| order_block_retest | 24 | 0.75 | -$2.7K | — | long |
| failed_continuation | 6 | 0.35 | -$2.8K | — | long |
| long_squeeze | 25 | 0.13 | -$6.8K | -$0.3K | short |

## Whale Footprint System (2026-02-24)

### What Was Wired
1. **Whale conflict penalty** (`archetype_instance.py:_compute_whale_conflict()`): Direction-aware 4-signal institutional data check. Penalties: 0=1.0, 1=0.95, 2=0.90, 3=0.85, 4=0.80.
2. **Per-archetype YAML gates**: funding_divergence (+2 gates: funding_oi_divergence, ls_ratio_extreme), long_squeeze (+2 gates: ls_ratio_extreme, funding_oi_divergence, removed frozen_bypass)
3. **Derivatives heat CMI component**: Wired in backtester + live (oi_momentum, funding_health, taker_conviction). **DISABLED** (weight=0.0) pending more data.
4. **oi_divergence archetype**: Enabled (shadow mode, only 3 trades so far)

### What Didn't Work (Reverted)
- **Liquidity sweep OI gates**: Temporal mismatch — sweep is detected DURING event, OI confirmation comes AFTER. Reverted.
- **Spring OI/taker gates**: Same temporal mismatch — at spring detection, selling IS the mechanism. Reverted.
- **Aggressive whale conflict thresholds**: -0.03 OI, -0.3 taker were too sensitive. Tightened to -0.05 OI, -0.5 taker.
- **CMI weight redistribution (10% derivatives_heat)**: Changed threshold calculations for all bars. Reverted to 0% weight.

### Key Insight: Temporal Mismatch
YAML soft gates check the CURRENT bar's features, but institutional confirmation (OI rising, selling subsiding) comes AFTER the pattern fires. Only add OI/taker gates for concurrent-state checks (funding rate IS concurrent), not post-event confirmations.

## ExitLogic Wiring (2026-02-25)

### What Was Wired
1. **ExitLogic class** (`exit_logic.py`) wired into backtester + live runner
2. **_PositionAdapter** bridges TrackedPosition to ExitLogic's Position interface
3. **Entry metadata** captured at open: wick_low, spring_low, ob_low, funding_z, oi_delta, adx, volume
4. **Per-archetype exit configs** in 16 YAML files + create_default_exit_config()
5. **Hard stop-loss stays inline** (fill-at-stop-level behavior) — ExitLogic handles everything else

### What Didn't Work (Fixed)
- **Invalidation exits**: -$54K from 66 wick + 14 spring exits — ALL losses. Close < wick_low is too sensitive, stop loss provides better protection. **DISABLED**.
- **Reason-gone exits**: Untested, disabled pending calibration
- **Regime-scaled R-levels**: Bull regime pushed [0.5,1.0,2.0] to [0.65,1.3,2.6], positions missed scale-outs. **Set all scale_level_multiplier=1.0**.
- **Regime-scaled time exits**: risk_off cut hold to 50%, crisis to 25% — net negative. **Set all time_exit_factor=1.0**.
- **Short max_hold (48h wick_trap)**: Halved PnL vs 168h. **Restored top-4 archetypes to proven 168h defaults**.

### Key Lesson: Theory vs Backtest
Per-archetype exit CHARACTER (48h for "fast reversal" wick_trap) was wrong — BTC moves slowly, even "fast" setups need multi-day holds. The proven 168h/[0.5,1.0,2.0]/[0.2,0.2,0.3]/1.0R/2.0x defaults work well for top-4 archetypes. Differentiated configs work for special archetypes (funding_divergence: 240h/4-levels).

## Heuristic Fixes Applied (2026-02-24)

1. **FROZEN_FEATURES corrected**: Removed `funding_Z`, `funding_rate` (have REAL data since 2020). Added `fusion_smc`, `tf4h_squiggle_confidence`, `tf4h_fvg_present` (truly frozen).
2. **squiggle_confidence 0.5→0.0**: Was frozen at 0.5, inflating momentum_score by +0.167. Now defaults to 0.0.
3. **fusion_smc 0.5→fallback**: Was frozen at 0.5 (neutral). Now computes from BOS/CHOCH/FVG components.
4. **chop_score formula fixed**: `1-ADX/100` → `max(0, 1-ADX/50)`. Median 0.74→0.49.
5. **liquidity_score fallback fixed**: New uses vol_z, atr_pct, tf1h_fvg, oi_change.
6. **drawdown_persistence default 0.9→0.5**: Old default biased bearish. Now neutral.
7. **ML temporal 0.5→feature-based**: Reads tf1h_temporal_score.

## Smart Exits V2 (2026-02-25)

### What Was Wired (all in `exit_logic.py`)
1. **Composite Invalidation V2** (`_check_invalidation` + `_compute_invalidation_score`): 5-feature score (BOS bearish, tf4h BOS, RSI<30, EMA slope<-0.001, volume_ratio<0.5). Threshold 4/5. Enabled for wick_trap, retest_cluster only. 9 exits, -$14K.
2. **Distress Half-Exit** (`_check_distress_exit`): 5-feature score (dd_score<0.10, chop>0.40, RSI<35, EMA slope down, vol_ratio<0.5). Threshold 4/5. If underwater (-0.2R) for 3-24h → exit 50%. All archetypes. 14 exits, -$7K.
3. **Chop-Aware Trailing** (in `_update_trailing_stop`): chop>0.45 → 0.75x trailing, chop>0.35 → 0.88x. Global.

### What Didn't Work (Tightened)
- **3/5 composite thresholds**: Too permissive — invalidation fired 34 times (all losses, -$25K), distress fired 71 times (all losses, -$30K). Combined -$55K destroyed $38K of PnL. **Raised both to 4/5** — now only 23 combined exits, selective and net positive.

### Key Insight: Composite Exit Threshold Must Be High
Academic research says multi-indicator exits outperform single-indicator by 20-40%. TRUE — but the threshold matters enormously. 3/5 catches too many false positives (positions that would recover). 4/5 is selective enough to only exit truly broken positions.

## Wyckoff V2 Recalibration (2026-02-18)

### What Was Fixed
1. **ST debounce + tighter thresholds**: lookback 30→15, volume_z 0.5→0.0, proximity 5%→3%, 10-bar spacing. ST dropped from 9,817→270 (97% reduction).
2. **SM st_count limit**: Max 2 STs per structure (breaks infinite self-loop ACCUM_ST→ACCUM_ST).
3. **Spring B activation**: Band 0.5-1.0%→0.2-1.5%, 3-bar recovery (was same-bar), lower quartile recovery (was mid-range). Spring B: 0→72 detections.
4. **SM spring tolerance**: 1% tolerance for Spring B vs sc_low (was strict `low < sc_low`).
5. **UT/UTAD SM relaxation**: Within 1% of bc_high (was strict `high > bc_high`), allow from more states (DISTRIB_ST, DISTRIB_SOW, DISTRIB_UT). UT: 1→5, UTAD: 1→5.
6. **Phase determination**: Confidence-gated (SC/BC conf>0.3 or 2+ support events). ST no longer dominates Phase A.
7. **Config key fixes**: st_lookback_max→st_lookback, st_price_tolerance→st_low_proximity, sos_range_pct_min→sos_breakout_margin. Removed vestigial sc_wick_min, bc_wick_min, utad_rsi_min.
8. **Live alignment**: _CFG_1H, _CFG_4H, _CFG_1D all updated with recalibrated params.

### Results (PF +3.2%, PnL +$29K, MaxDD improved 6.8%)
- Bull/Bear ratio: 16:1 → 1.7:1 (target 2-4:1)
- Spring archetype flipped from loser (-$1.8K) to winner (+$3.7K, PF=1.62)
- Liquidity sweep PF improved: 1.89→2.23 (+$24K PnL improvement)

### Key Insight: Debounce + ST Count Are The Critical Fixes
The 10-bar debounce and SM max-2-STs-per-structure together killed the self-loop that caused 71% Phase A. Academic literature has no Wyckoff calibration paper — best thresholds synthesized from PyQuantLab VSA, loopofM, QuantVue, Wyckoff Analytics open-source implementations.

## CMI Weights: Data-Informed (2026-02-21)

```
risk_temp:   dd_score=50%, trend_align=30%, sentiment=15%, trend_strength=5%, derivatives_heat=0% (READY but disabled)
instability: chop=40%, wick_score=25%, vol_instab=25%, adx_weakness=10%
crisis:      base_crisis=45%, sentiment_crisis=45%, vol_shock=10%
```
Config: base=0.18, temp_range=0.38, instab_range=0.15, crisis_coeff=0.50

## Signal Dedup + Optuna ATR (2026-02-23/24)

- **Dedup DEPLOYED**: `signal_dedup.mode: best_per_direction` — max 1 long + 1 short per bar
- **Optuna ATR COMPLETE**: 100 trials, Best #87, WFE=107%
- **Key: wide stops for volume archetypes** — wick_trap/liq_sweep/retest_cluster all optimal at 3.4x ATR
- **Dedup modes**: best_of_bar | best_per_direction | unique_sl_zone | disabled

## Binance Futures Features (2026-02-24)

- **8 columns**: oi_value, oi_change_4h, oi_change_24h, oi_price_divergence, binance_funding_rate, funding_oi_divergence, ls_ratio_extreme, taker_imbalance
- **Coverage**: 100% for 2022-2024, 8.5% for 2021, NaN before 2021
- **Live wired**: `live_feature_computer.py` imports BinanceFuturesAPI with 5-min cache

## Critical Lessons (HARD-WON)

1. NEVER Add Bonuses to Fusion Scoring
2. Double-Penalty Problem (backtester already has fusion_thresholds_by_regime)
3. Small Sample PF is Misleading (need 200+ trades)
4. Wider Exits Hurt (76% hit SL before 2R)
5. More Positions = More DD
6. Selectivity > Deployment
7. Default Cost Model Overcharges — 2bps comm, 3bps slip, $100K capital
8. regime_preferences as fusion multiplier + varying thresholds = DOUBLE-GATING
9. **ProbabilisticRegimeDetector has broken normalizations** — use direct EMA-based scoring
10. **Feature store regime_label is ML garbage** — backtester derives CMI-based regime at init
11. **Fusion multiplier approach fails** — **Dynamic threshold is the right architecture**
12. **SMA features NaN for 2020-2021** — use EMA features which have 100% coverage
13. **SMA regime labels REMOVED** — replaced with CMI-derived bull/bear/neutral/crisis
14. **Never hardcode feature values** — use CoinGecko API or NaN fallback
15. **CMI must be orthogonal to archetype fusion** — prevents double-counting/overfitting
16. **Never let accumulation cancel crisis** — catastrophic dip-buying
17. **NaN guard on feature extraction** — `_get()` must check `val != val` for NaN
18. **Diagnostic defaults must match config** — live_feature_computer.py hardcoded defaults
19. **Per-exit-row stats are misleading** — scale-out exits create multiple small win rows
20. **Adverse-regime trades are marginal-threshold trades** — raising temp_range kills them
21-39. (See previous entries — unchanged)
40. **Heuristic fixes improve PF 22% with half the trades** — PF 1.27→1.55
41. **Whale YAML gates need concurrent-state features ONLY** — OI/taker gates fail for sweeps/springs because the pattern fires DURING the event, but whale confirmation comes AFTER. Only funding state (concurrent) works in gates.
42. **Whale conflict penalty must be light (5% per signal)** — 10% per signal killed $54K from wick_trap. Softened to 5%/10%/15%/20% tiers.
43. **bypass_threshold NOT implemented in backtester** — config says bypass=true but backtest code ignores it. Threshold always active.
44. **derivatives_heat needs >2 years of data** — CMI weight redistribution changed thresholds for ALL bars (including 2020-2021 with no OI data). Disabled until more OI data accumulates.
45. **Invalidation exits are net negative** — wick/spring invalidation (close < entry wick_low) fires before stop loss, closing at close price instead of waiting. ALL 80 invalidation exits were losses (-$54K total). Stop loss provides better protection.
46. **Regime-scaling exit R-levels kills profit** — bull regime pushed scale-out levels 30% higher ([0.5→0.65, 1.0→1.3, 2.0→2.6]), positions missed targets and stopped out. Scale-out levels must be ABSOLUTE.
47. **Per-archetype max_hold must be data-driven not theory-driven** — "fast reversal" wick_trap at 48h halved PnL vs proven 168h. BTC moves slowly; avg hold is 33h but outlier winners need 5-7 days. Theory says short, data says long.
48. **ExitLogic has TWO regime adjustment functions** — `_get_regime_exit_adjustments()` scales scale_level_multiplier, `_get_regime_adjusted_params()` scales time_exit_factor. Both must be neutralized; the old inline code had NO regime exit scaling.
49. **Composite exit threshold 4/5 not 3/5** — at 3/5, invalidation fired 34x (all losses, -$25K) and distress fired 71x (-$30K). At 4/5: only 9+14=23 exits, net positive (+$7K PnL, +3.7% Sharpe). Higher confidence = fewer false positives on positions that would recover.
50. **Chop-aware trailing is always positive** — tightening trailing by 12-25% when chop_score > 0.35-0.45 catches profit before whipsaw reversal. Research-backed (Clare et al. 2013: volatility-adaptive stops reduce DD 45-65%).
51. **Wyckoff ST self-loop is the #1 detection bug** — SM allowed ST→ST indefinitely, causing 16% of all bars to be "ST" and 71% Phase A. Fix: st_count limit (max 2) + 10-bar debounce + volume_z<0 (below mean, not just "low"). This single fix chain improved PF from 1.58→1.63.
52. **Config key mismatches are silent failures** — `st_lookback_max` was ignored because detector reads `st_lookback`. Python dict.get() returns default silently. Always verify config key names match the cfg.get() calls in the detector functions.
53. **Spring B needs multi-bar recovery on 1H** — same-bar close > range_mid is impossible on 1H BTC (too coarse). 3-bar window + lower quartile target (not mid-range) activated 72 spring events from 0.
54. **Fusion score has NEGATIVE predictive power** — Pearson r=-0.082 (p=0.018), Spearman r=-0.122 (p<0.001). ALL domain scores (wyckoff, liquidity, momentum, smc) have near-zero correlation with PnL. The edge comes from structural detection + hard gates, not fusion scoring.
55. **Hard→soft gate_mode revert saves PF** — retest_cluster hard mode dropped PF 1.43→1.18 (marginal-but-profitable signals blocked). order_block_retest wyckoff 0.05→0.20 + hard mode killed ALL 24 trades→0. Reverted both.
56. **Volume gate on wick archetypes is neutral-positive** — wick_trap + trap_within_trend with `volume_zscore >= 0.0` (median vol) filters low-quality wicks. PF impact: -2% to +3%.
57. **2-bar entry spacing > 4-bar** — 4-bar spacing blocked 87 profitable entries ($15K PnL loss). 2-bar only blocks 38 truly correlated entries. Live + backtest both wired.
58. **Perp position sizing must be margin-based, not notional-based** — 3 bugs fixed: (a) leverage was double-applied to notional (risk/stop already gives correct notional), (b) cash deduction was full notional instead of margin (notional/leverage), (c) position cap was on notional % instead of margin %. Leverage determines MARGIN requirement, NOT risk. Risk = notional × stop_distance. Correct: `notional = risk / stop` (no leverage multiply), `margin = notional / leverage`, `cash -= margin`.
59. **max_margin_per_position_pct = 0.35** — At 1.5x leverage: max notional = $52.5K, margin = $35K. At 2.5% stop: risk = $1,312 (1.3%). Can run ~3 concurrent positions. Reference correct impl: `pnl_tracker_v2.py:138-152`.

60. **Boost vs Filter empirical accept rate (May 2026)** — across 10+ quant studies this cycle:

   | Change type | Tested | Accepted | Rate |
   |-------------|-------:|---------:|-----:|
   | Sizing/exit boosts (Type A) | 2 | 2 | **100%** |
   | Hard/soft gates (Type B) | 3 | 0 | 0% |
   | Mode flips (Type C) | 1 | 0 | 0% |
   | CMI/regime tuning (Type D) | 3 | 0 | 0% |
   | Direction flips (Type E) | 1 | 0 | 0% |

   **Only sizing boosts** (TP Tier 1 +7.19%, 3-of-3 dist_exhaustion +2.26%) survived WFO. Filters/gates/mode-flips/CMI-tuning all failed via one of: **(a) dedup-reshuffling** (gate shrinks target archetype, dedup re-routes bars, system PnL rises but target loses); **(b) selection asymmetry** (regime gates filter the WINS, not losses — long_squeeze gates this 3 times); **(c) gate-immune architecture** (soft+bypass coexistence makes hard-gate tightening inert — CB-specific).

   Codified in `.claude/agents/quant-analyst.md` as Zero-Tolerance Rules 7-10 (May 18, 2026). When proposing any new gate, always also propose the BOOST equivalent and test both. Default to boost.

61. **Dedup-reshuffling false signal** — system PnL gain WITHOUT target-archetype PnL gain = false signal. Validated May 18 with LC + `oi_change_24h ≤ -0.02` hard gate: system +$2,595, target archetype −$5,465. The system gain was just dedup re-routing bars to OTHER archetypes — the rule did NOT do what we intended. Auto-reject codified as Rule 7.

62. **Train regression + OOS gain = overfit signature** — derivatives_heat at 10% weight produced +$1,221 OOS but −$25,954 train on May 18. Often happens when the OOS window has a regime tailwind aligning with noise the variant happened to pick up. Codified as Rule 9: both train AND OOS must move in the same direction.

## Structural Overhaul (2026-03-02)

### What Was Done
1. **Phase 1: YAML gate tightening** (PF 1.51→1.75, +15.9%)
   - wick_trap: gate_mode soft→hard, added volume_zscore >= 0.0
   - trap_within_trend: gate_mode soft→hard, added volume_zscore >= 0.0
   - spring: wyckoff min 0.10→0.15, pti min 0.05→0.10
   - long_squeeze: RSI min 50→60
   - liquidity_sweep: removed frozen boms_strength gate (kept liquidity_score only)
   - retest_cluster: REVERTED hard mode (PF 1.43→1.18)
   - order_block_retest: REVERTED wyckoff 0.20 + hard mode (killed all trades)
2. **Phase 2: ATR distance + entry spacing** (PF 1.75→1.74, net neutral)
   - ATR min distance: stop must be >= 0.5x ATR from entry
   - Same-direction entry spacing: 2-bar minimum between long/short entries
3. **Phase 3: Domain score logging** — wyckoff/liquidity/momentum/smc/gate_penalty in trade_log.csv
4. **Phase 4: Engine Health dashboard** — EngineHealth.tsx with bypass/threshold/positions/gates

### Backup
- Git tag: `pre-structural-overhaul`
- Backup dir: `configs/archetypes_backup_pre_overhaul/`
- File backups: `*.pre_overhaul` for archetype_instance.py, v11_shadow_runner.py, backtest_v11_standalone.py

## Feature Store Known Issues (CORRECTED 2026-02-24)
- `tf4h_fvg_present`: Always 0 (truly frozen)
- `boms_strength`, `tf1d_boms_strength`, `tf4h_boms_strength`: ~2.3% non-zero (mostly frozen)
- `fusion_smc`: Always 0.5 (truly frozen — fallback computation used instead)
- `tf4h_squiggle_confidence`: Always 0.5 (truly frozen — defaults to 0.0 now)
- `funding_Z`, `funding_rate`: **REAL DATA** since 2020 (NOT frozen)
- `chop_score`: **RECOMPUTED** with max(0, 1-ADX/50) formula
- BOS features sparse: tf1h_bos_bullish 3.9%, tf1h_bos_bearish 3.2%

## Dashboard
- **Current**: v7.6 React SPA (React 19 + TypeScript + Vite 6 + Tailwind CSS 4)
- **Live**: http://165.1.79.19:8081
- **Build**: `cd dashboard && npm run build` → `dashboard/dist/`
- **v7.6**: Added WhaleIntelligencePanel — shows raw Binance OI/funding/LS/taker data + engine-derived signals + conflict status with explanations

## Deployment
- **IP**: 165.1.79.19 | **SSH**: `ssh -i ~/.ssh/oracle_bullmachine ubuntu@165.1.79.19`
- **Deploy**: `./deploy/deploy.sh` (code + restart coinbase-paper + dashboard)
- **Monitor**: `sudo journalctl -u coinbase-paper -f`
- **Services**: coinbase-paper (800MB limit, $100K) + dashboard (8081, 200MB limit)

## CMI Weight Sync (2026-03-04) — FIXED

Live code had stale hardcoded CMI weights. Backtester reads from config (correct). Fixed:
- `live_feature_computer.py`: instab (0.35/0.25/0.20/0.20→0.40/0.10/0.25/0.25), crisis (0.60/0.20/0.20→0.45/0.10/0.45), diag logs (temp_range 0.48→0.38, instab_range 0.20→0.15)
- `v11_shadow_runner.py`: all .get() defaults aligned (risk_temp dd_score 0.15→0.50, etc.)
- **Lesson 60**: Live CMI weights must match config — 3 independent files compute CMI (backtester, feature_computer, shadow_runner). Only backtester reads config; other two hardcode. Always grep for weight values after changing config.

## Dashboard Health (2026-03-04)
- **WORKING**: Wyckoff (all 13 events, MTF, state machine), macro APIs, threshold, positions, signals, trades
- **COSMETIC GAPS**: Oracle synthesis not implemented (shows "synthesizing..."), Wyckoff conviction breakdown empty
- **FIXED**: CMI weights now aligned, diagnostic logging params corrected

## STANDING ORDERS (DO NOT OVERRIDE)
- **NEVER turn off bypass_threshold** — user wants data collection mode ON for the foreseeable future
- **NEVER disable archetypes** — all 16 stay enabled to collect maximum live signal data
- **NEVER make production config changes** (bypass, disabled_archetypes, thresholds) without explicit user approval

## Live Deployment
- **2026-06-30 LIVE PAPER CONFIG** (`configs/champion_paper.json`, service `--config`): FULL BOOK — all 16 archetypes enabled, bypass_threshold=True (data collection), archetype_config_dir=archetypes_v14rq so wick_trap + trap_within_trend now FIRE at re-quantiled 0.43 (were locked out at 0.72). [History: briefly deployed lean core2-only, then user reverted to full-book-plus-unlocked.] deploy.sh excludes deploy/ so the service file installs MANUALLY (scp+systemctl). Breakeven-1R exit = phase 2 (not deployed).

## Post-Entry Alpha 2026-07-14
- [Industry study: post-entry alpha](industry_study_post_entry_alpha_2026_07_14.md) — verified pro playbook: (1) META-LABELING (sizing layer on frozen strategy, fed post-entry+regime features — never retune entries), (2) MAE/MFE trade management from own excursion distributions (validates our method), (3) BREADTH: at N=2 strategies, adding a 3rd uncorrelated edge beats all tuning, (4) TC² — short live windows are mostly noise, (5) execution/maker alpha ~2bp/side, (6) NEVER retune on live losses (trial counter irreversible)
- [Path-conditional verdict](path_conditional_verdict_2026_07_14.md) — post-entry info IS real (LC: MFE<0.25R@12h = 0% winners; losers 2x early MAE) but the time-cut policy = WATCH-ITEM not deploy: only 7 actionable cuts in 8.5yr (mid window vacuous 0 events), 1-of-12 substantive pass, EV bound not cleared (0/7 recoveries vs 0.33 breakeven). Curve-methodology gap found (mfe_at{h} counts already-stopped positions, overstates actionable pop ~4x). Deploy trigger pre-registered: n>=15 cut events, <=1 baseline-winner. wick_trap: all 6 cells fail (losers tease green)

## Winning Strategies Scoreboard
- [LC battery 2026-07-14](lc_battery_2026_07_14.md) — liquidity_compression validated as a REAL BUT THIN second edge: holdout PF 1.14 (+$3.0K), perfect train/holdout co-move (1.14/1.14, no overfit signature), 75% WR, positive 5/7 years, bear-weak like everything. Book = wick_trap (champion 1.43) + LC (thin 1.14) + funding_divergence (garnish). LC's live lead (+$2.9K all-time, +$4.9K/30d) is consistent with its profile, not luck

## Unified Strategy 2026-07-13
- [Trailing sweep verdict](trailing_sweep_verdict_2026_07_13.md) — REJECTED: monotone overfit gradient (train 1.37->1.76 while holdout 1.43->0.96 as trailing starts earlier). ALSO: YAML exit_logic.trailing_start_r is DEAD CONFIG (bit-identical runs); working path = user-JSON exit_logic override. OPTIMIZATION FRONTIER NOW CLOSED: every lever tested, wick_trap as-is is a local maximum in all directions; new info comes only from live evidence, U3 watch item, V15 rebuild (BOS backfill), or structurally new instruments
- [Edge-within-the-edge hunt](wick_trap_edge_within_2026_07_13.md) — NO exploitable entry-state discriminator inside wick_trap (4th independent confirmation; level features don't discriminate; strong train AUCs evaporate OOS). Only convergent candidate (low-chop+high-adx, also seen in live-50) REJECTED by sizing co-move test (train down/holdout up). Remaining dimensions: exit-side (trailing_start_r sweep) + U3 watch item
- [Unified strategy verdict](unified_strategy_verdict_2026_07_13.md) — level anchoring FAILS the pre-registered bar (U1 sweep starves to n=3; U2 quality violates co-move; U3 near-support intriguing but n=23). THE UNIFIED STRATEGY IS WICK_TRAP, by proof: all knowledge already converged into it. V14L store + live level features = permanent infra; U3 = watch item at n>=30

## Audit-1 Repairs 2026-07-13
- [Repair batch + re-baseline](audit1_repair_batch_2026_07_13.md) — BOS emitters FIXED (attr mismatch, 0%->4% fire rate; V14 stored columns still 0 until rebuild); impossible thresholds rescaled; never-validated live-only gates removed; instability gate activated; ls_ratio true z-score. RE-BASELINE: wick_trap validation HOLDS IDENTICALLY (holdout PF 1.43); liquidity_sweep DOWNGRADED (broken-gate penalty was accidentally protective — repaired holdout PF 0.61, edge was artifact). wick_trap = the book's ONLY validated edge

## Triple Audit 2026-07-10
- [Engine integrity audit](engine_integrity_audit_2026_07_10.md) — 6 CRITICAL: BOS/CHoCH emitters dead on ENTIRE live path (fvg_continuation can never fire, SMC domain blind book-wide); 2 mathematically impossible gates (liquidity_vacuum dead; liquidity_sweep permanently -50% soft-penalized: wick_lower_ratio>=1.3 vs feature max 1.0); rsi_divergence gates never in any backtest but block 96-98% live (kills failed_continuation + volume_fade_chop); ls_ratio_extreme live/store scale mismatch. 5 archetypes have NEVER signaled since unlock — all traced to dead components, not rarity. WYCKOFF + MOMENTUM VERIFIED WORKING (user fear cleared)
- [Strategy book review](strategy_book_review_2026_07_10.md) — book is functionally ~1.5 strategies of 17: 2 dead-by-direction (neutral archetypes return None in detect()), 3 dead-by-data (BOS), inert-gate epidemic; thesis incoherences (CB has NO breakout test, TWT no trend test, spring accepts UTAD distribution-tops 1860:826, ER fires more on overbought); 35-46% entry clustering = single factor (long-BTC flush-buying). Ranked hypotheses H1-H6; re-baseline REQUIRED before fixing dead gates (validated configs include inert gates)
- [Last-50 live forensic](live_50_forensic_2026_07_10.md) — nothing entry-observable separates W/L (fusion inversion replicated 3RD time: enforced-threshold 1W/9L -$7.6K); outcome=BTC direction (15/15 win when BTC rose, 2/35 when fell); CB no-chase FAILED its OOS replication (downgraded); recovery +$7.2K since 06-30 driven by tape turn (LC+CB); only loss-structure-matching lever = downtrend trade-SKIP (offline-validated, needs approval)

## Validated Champion
- [wick_trap CPCV 2026-07-09](wick_trap_cpcv_2026_07_09.md) — PASS 15/15 combinations positive, median PF 1.41, WORST alternate history +$2,421. wick_trap_v14rq standalone (unstacked) has now cleared EVERY offline gate: battery, pristine holdout (PF 1.43), stacking adjudication, CPCV. Remaining evidence = LIVE (already unlocked in full book @0.43). Satellite strategy: bleeds modestly in deep bears; no overlay improves it

## Direction Fix
- [Downtrend study 2026-07-02](downtrend_study_2026_07_02.md) — DEFENSIVE SKIP works, naive SHORT does not. skip longs when price < 200-day mean: 2022 -$43K->$0, MaxDD 51%->16.4%, full PnL $163K->$198K, PF 1.16->1.29 (holdout -$14K->-$8.4K, still neg = bleed-STOP not profit engine). SHORT feasibility: fwd returns during downtrends are ~coin flip (49% neg, ~0% mean) — being below 200d does NOT predict imminent downside; naive regime-short has NO edge, would lose to fees. DEPLOY the skip; do NOT build a blind short (needs a precise setup, unproven).

## Live Trade Forensics
- [CB forensic 2026-07-04](cb_forensic_2026_07_04.md) — last-10 live CB: fusion zero separation (again). Live pattern (winners buy the 7d-low washout, losers chase extended) VALIDATED AT SCALE only as loss-concentration: entries >4% above 7d low = 410 pos, -$78/avg, -$32K total; near-low ~breakeven. No feature makes CB a WINNER (42% WR) — "don't chase" avoids the bleeding, doesn't create edge
- [Last-30 live forensic 2026-07-02](live_trade_forensic_2026_07_02.md) — net -$10,362 (9W/21L), ALL long into a -22.8% BTC decline. AT ENTRY effectively RANDOM: no feature predicts win/loss; fusion is INVERTED (the 5 trades clearing the enforced threshold were ALL losers, -$5,886). Dominant failure mode: 18 longs straight to stop (-$18.2K). 6 real runners (+$7.9K) prove triggers work when market cooperates. Good archetypes (wick_trap/liq_sweep) ABSENT (locked out live-path). Turning bypass OFF would delete all 9 winners + keep 5 worst losers. Lever = direction/regime SKIP in downtrends + fix live-path; exits minor (only 3 gave back a +1R winner).

## Experiment Results
- [Stack validation verdict 2026-07-08](stack_validation_verdict_2026_07_08.md) — 9-cell grid (book x protection, REAL engine keys, no monkey-patches): ALL cells FAIL pre-registered acceptance. **RETRACTS both 06-29 BE results**: the monkey-patch mutated pos.stop_loss → R denominator zeroed → scale-outs/trailing silently DISABLED post-BE = accidental "ride policy" (max_win 2.5x, WR -16pts — impossible for a stop-ratchet), not breakeven. Honest production BE = 1 trade / +$15 across ~337 (trailing@1R instantly 2.9 ATR better than entry). skip200 on wick_trap = Rule 9 FAIL (train 1.37→1.71 UP, holdout 1.43→1.10 DOWN; removed trades 32W/4L = the below-200d washout winners ARE the edge). Honest best: wt_only__none (holdout PF 1.43 +$6.8K n=70) > core2__none (liq_sweep adds +$62 OOS = passenger). Per-archetype protection assignment MOOT (skip's only consumer fails floors anyway). Next: CPCV wt_only__none; re-register "ride policy" as clean new hypothesis; trailing_start_r sweep (0.5/0.75/1.0) covers the unprotected +0.41..1.0R window
- [Structural embed experiments](feedback_structural_embeds.md) — momentum gates regress, don't use on reversal archetypes
- [Wyckoff fix status](project_wyckoff_fix_status.md) — live fix committed, feature store rebuild pending
- [Champion strategy: wick_trap + exhaustion_reversal pair](champion_strategy_pair_2026_06_10.md) — passed the full V12 battery (positive every year 2020-24) but FAILED the 2025-26 holdout; 200-HOUR vs 200-DAY regime-label gap remains a valid finding
- **[RETRACTED 2026-07-08 — pos.stop_loss-mutation artifact, see stack_validation_verdict_2026_07_08.md]** [Breakeven full-system + core2 validation 2026-06-29] — wick_trap BE@1R confirmed real even in full-16 book (its own PnL +$10K: $18.8K->$29K, 200->157 trades) BUT full-system total DROPS $163K->$154K (freed slots filled by junk = dedup interaction). In LEAN core2 book the gain FLOWS THROUGH: core2+BE(wicktrap) = train PF 1.48->1.62, holdout 1.30->1.42 (+$8K), full PF 1.41->1.54, MaxDD 17->15.9% — co-move up. DEPLOYABLE config = core2 + wick_trap-BE (lean book required); still long-only/bears bleed; needs CPCV + approval
- **[RETRACTED 2026-07-08 — the "no artifact" claim was wrong: accounting was internally consistent but the patch measured a different exit policy (scale-outs/trailing disabled post-BE), not breakeven; production BE = +$15/1 trade. See stack_validation_verdict_2026_07_08.md]** [Breakeven-stop verdict 2026-06-29](breakeven_study_verdict_2026_06_29.md) — BE@1R: accounting clean (no artifact) but only WICK_TRAP is a real edge (gross profit flat, gross loss -13% = textbook loss-cut). liquidity_sweep = reshuffle luck (gross profit went UP +25% — impossible for a stop-tightener; n=29<30); spring = in-sample rescue (WR 65->29% — the V12 invalidation failure pattern). NOT deployable; BE changes entry population (not clean A/B). Next: wick_trap CPCV + full-system re-run w/ frozen entries
- [MFE analysis 2026-06-28](mfe_analysis_2026_06_28.md) — hypothesis REFUTED: we capture 84% of winners high-water mark (give back only 0.2R) — NOT a take-profit problem. Real leak: 14% of LOSERS (exh_rev 24%) were +1R before reversing to a loss. Trades top out ~1.4R (only 23% reach 2R) so far targets are vestigial. Fix = breakeven stop after +1R (testable exit study)
- [Winner-loser forensic 2026-06-28](winner_loser_forensic_2026_06_28.md) — archetypes ALREADY win 65-78% individually; entry features barely separate win/loss (AUC ~0.5, flip OOS = Lesson #54 confirmed); the ONE robust discriminator is duration (winners run, losers stop fast = EXIT lever, reverse-causal); 2-3 narrow OOS-real entry edges (trap_within_trend with-trend+away-from-support; liquidity_sweep fusion). "Win more" = exits, not triggers
- [Risk-overlay verdict 2026-06-27](risk_overlay_verdict_2026_06_27.md) — CUT-THE-BOOK confirmed: core2 (wick_trap_v14rq+liquidity_sweep) holdout PF 1.30 +$6.8K, MaxDD 17% vs full16 PF 1.16 MaxDD 51% holdout-NEGATIVE. BUT +$6.8K is ~all wick_trap (liq_sweep breakeven); wider stops + bear-sizing are OVERFIT TRAPS (in-sample up, holdout down); bears still bleed. Deploy=capital-allocation overlay (16 logged/2 funded), needs CPCV + approval (Standing Order #2). Next: exit scale-out winner-capture study
- [Live validation plan 2026-06-24](live_validation_plan_2026_06_24.md) — decision: keep collecting + validate wick_trap_v14rq OFFLINE before any live change (normal 2% sizing on deploy); harness built (scripts/champion/validate_live_offline.py), 0 trades on 5d (insufficient, needs weeks); latent finding: wick_trap `instability` gate INERT (reads `instability`, engine emits `instability_score` — skipped in all backtests incl. holdout)
- [Sizing studies verdict 2026-06-16](sizing_studies_verdict_2026_06_16.md) — sizing CANNOT flip a long-only strategy bear year (0/13 variants flip 2018/21/22; confirmed twice). wick_trap_v14rq already clears a professional bar (full PF 1.46, holdout 1.43). Recommendation P1+P3: adopt professional bar + build trade-SKIP regime kill-switch (size=0 untestable — engine mis-accounts zero-qty positions). P2 short-side deferred
- [V14 champion hunt results](v14_champion_hunt_2026_06_11.md) — 0 passers on honest data; wick_trap RESURRECTED via re-quantiled threshold (holdout +$6.8K PF 1.43 n=70, the best honest asset; bear years still bleed); V12-era pair construction does NOT transfer (legs' regime personalities flipped between feature spaces); trap_within_trend edge was a V12 artifact; 3 pre-registered sizing studies queued (S1 macro, S2 day_type, S3 level-quality)
- [ES failed-breakdown knowledge transfer](trader_knowledge_failed_breakdown_es_transfer_2026_06_11.md) — trade the RECLAIM never the break; acceptance (time holding the level) separates trap-reversal from dead-cat; level QUALITY is the edge; range-until-proven-trend day classification (the missing intermediate regime — explains 2022 + May-Jun 2026 losses); structural stops past the swept level; 4 pre-registered hypotheses for V14 WFO
- [Industry study: wick trap detection](industry_study_wick_trap_detection_2026_06_11.md) — our detector is built backwards: field standard is LEVEL-ANCHORED sweeps (wick-beyond/close-back-inside a prior swing/pool, NO wick-ratio gates anywhere); Osler 2005: stop runs CASCADE for hours (never enter on the sweep bar), reversals live AT levels; volume gate unvalidated either way; full redesign spec inside
- [Industry study: backtest-live parity](industry_study_backtest_live_parity_2026_06_11.md) — verified 3-layer playbook (single code path / feature logging + PSI-JS drift checks / shadow reconciliation) with concrete CI thresholds (PSI 0.1 warn, 0.25 fail; JS 0.1; range assertions) and a 6-item ranked adoption plan
- [liquidity_score root cause](liquidity_score_root_cause_2026_06_10.md) — wick_trap + trap_within_trend (the #2/#3 backtest earners) are STRUCTURALLY DEAD in live: their liquidity_threshold 0.72 is unreachable on live-path features (live max 0.675 vs V12 q90 0.72). Fix = full live-path rebuild 2018→2026 + re-quantile all thresholds
- [Holdout verdict + pipeline divergence](holdout_verdict_2026_06_10.md) — CRITICAL: wick_trap fires 0 trades on live-pipeline features (bridge test: 17 on V12 vs 0 on live-path, same market; 0 in 158 real live trades). V12 backtests overstate what live can express. V13_EXTENDED store (2018→2026-06, live-path 2025+) is the new validation substrate; scripts/rebuild/ is the reusable pipeline

## Next Priorities
1. **Feature store rebuild** — fix scoring for sparse events + Optuna re-optimize (see wyckoff fix status)
2. **Disable proven losers** — long_squeeze (PF=0.08, -$8.1K), order_block_retest (PF=0.62, -$4.2K)
3. **Enable derivatives_heat when OI data reaches 3+ years** — currently 2 years (2022-2024)
4. **Implement Oracle synthesis** — `_build_oracle_synthesis()` method in coinbase_runner.py

## Data Infrastructure
- [Derivatives data backfill method](derivatives_data_backfill_method.md) — `data.binance.vision` is the ONLY working source for historical OI/funding/LS/taker. Binance REST API and OKX are geo-blocked or 30-day-limited. CDN is geo-unblocked. Script at `scripts/data/backfill_binance_vision_derivatives.py`. Coverage 2020-09 onwards (62%).

## Python/System Notes
- Use `python3` not `python` on this system
- Always guard against NaN in stop_loss, take_profit, and position sizing
- Feature store scripts must coordinate - check what symlink points to
- [Emergent knowledge & short hypotheses](project_emergent_knowledge.md) — EMA/Wyckoff divergence signal, distribution idle time, short archetype candidates
- [Never sed-patch the server](feedback_never_sed_patch.md) — always commit to git first, sed patches get overwritten on deploy (cost us $2-3K)
- [Confluence breakout investigation](project_confluence_breakout_investigation.md) — FRVP bug fixed, bypass_fusion_threshold added, gate tightening TODO
- [Live evidence engine](live_evidence_engine_2026_07_15.md) — `bin/live_evidence.py` turns collected paper metadata into 4 decision sections: (1) per-archetype live-vs-holdout scorecard w/ Wilson CIs — LC ON TRACK (live PF 1.30 n=26 vs holdout 1.14), (2) counterfactual split by threshold_margin — LIVE confirmation of Lesson #54 (threshold-cleared PF 0.27 vs bypass-only 0.76: high fusion = worse), (3) automated LC time-cut watch-item ledger (1/15 cut events, accumulating), (4) execution costs = 29% of |net PnL|, maker-first would recover ~$3.3K/5mo. Run weekly.
- [Trader-knowledge audit](trader_knowledge_audit_2026_07_16.md) — full wiring×evidence cross-ref of all ~7 trader domains. Found+fixed CHoCH clobber (07-13 fix was overwritten to 0 by later stub — dead on every live bar until 07-16). Gaps: effort_result_ratio backtest-only (gates skip live), whipsaw/volume_fade_chop can NEVER fire (neutral direction), YAML exit ladders + regime_preferences dead config. FALSE ALARM: daily wyckoff (70% nonzero live, 11d median zero-streaks) and BOMS (0.7% base rate, 277d max gap) are healthy — don't "fix". V15 retest candidates: fvg_continuation/SMC-BOS, liquidity_vacuum, funding_div live record.
- [V15 structure verdicts](v15_structure_verdicts_2026_07_17.md) — V15 store (repaired BOS/CHoCH/BOMS/graded-wyckoff cols). SECOND repair found: structural checks used GLOBAL bar index on rolling window → _check_B/_check_C auto-rejected forever (fixed+deployed, wick_trap bit-identical). Verdicts: **order_block_retest VALIDATED as third edge** (holdout PF 2.08 n=31, CPCV 15/15 med 1.39 worst +$2.9K, OOS≥train, never tuned); fvg_continuation REJECTED honestly (holdout 0.71); liquidity_vacuum REJECTED (PF 0.40). Champions improved on V15: wick_trap holdout 1.71, LC 1.23.
- [Founding knowledge archaeology](founding_knowledge_archaeology_2026_07_17.md) — full inventory of v1.x trader knowledge (Bojan/Moneytaur/Wyckoff_Insider/ZeroIka/Crypto Chase) graded FULL/PROXY/DORMANT/EXTINCT vs deployed path. Both champions are lossy proxies of richer originals (wick_trap lost trap-reset spec; OBR lost HOB quality score). Ranked 13-item lost-fidelity list = re-implementation queue (trap-reset #1, HOB quality boost #2, reclaim-speed #3, equal-cluster magnets #4, PO3 sequencing #5). Discipline: original spec verbatim, pre-register, V15 battery + CPCV, one at a time.
