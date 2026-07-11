# Engine Integrity Audit — 2026-07-10

**Scope**: Is every piece of the engine actually firing, or silently broken?
**Method**: Every hard gate in `configs/champion/archetypes_v14rq/` + `configs/archetypes/` cross-checked against (a) the live feature log (533 bars, 2026-06-18 → 2026-07-10, `results/coinbase_paper/live_features/*.jsonl`), (b) `data/features_mtf/BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet` (73,829 bars, 2018-2026), and (c) `BTC_1H_FEATURES_V12_ENHANCED.parquet` (the store backtests actually replay — `BTC_1H_LATEST.parquet` symlinks to it). Plus: degenerate-feature sweep, exception-swallow sweep, config-key liveness, component sanity, and per-archetype signal liveness from `results/live_audit_2026_07_10/signal_log.json`.

Confirmed pipeline fact: gates see **exactly** the live-log columns — `isolated_archetype_engine.py:698` does `features = bar.to_dict()` with no enrichment, and `coinbase_runner.py:3130` logs the same Series it passes to `process_bar` at `:3141`.

---

## Severity-Ranked Findings

| # | Sev | Finding | Where |
|---|-----|---------|-------|
| 1 | CRITICAL | `bypass_threshold: true` is live NOW — sub-threshold signals trade ("BYPASSED for data collection"); same file's notes claim "PRODUCTION MODE. bypass_threshold=false … bypass is catastrophic" | `configs/champion_paper.json` (adaptive_fusion), signal_log entries 06-30→07-10 |
| 2 | CRITICAL | BOS/CHoCH emitters dead on the live path: `tf1h_bos_bullish/bearish`, `tf4h_bos_*`, `tf1h_choch_detected`, `tf4h_choch_flag` are 0 for all 533 live bars AND all 73,829 V14 rows — but fire 802 times in V12 (what backtests were calibrated on). `derived:any_bos_any_tf`/`any_bos_1h` are therefore always False | live log; V14; V12; `archetype_instance.py:47-51` |
| 3 | CRITICAL | **fvg_continuation permanently dead live**: `derived:any_bos_any_tf bool_true` with `nan_policy: fail` + `gate_mode: hard` can never pass (see #2). Zero signals ever in signal_log | `configs/champion/archetypes_v14rq/fvg_continuation.yaml` |
| 4 | CRITICAL | **liquidity_vacuum permanently dead everywhere**: gate `wick_exhaustion_last_3b min 1.4` is mathematically unreachable — the feature is a mean of per-bar wick fractions, bounded < 1.0 by construction (live max 0.93, V14 max 0.98, pass 0.0% both). `gate_mode: hard`. Zero signals ever | `liquidity_vacuum.yaml`; formula at `live_feature_computer.py:2524-2532` |
| 5 | CRITICAL | **liquidity_sweep gate unreachable**: `wick_lower_ratio min 1.3`, but the feature is bounded ≤ 1.0 (V14 max exactly 1.0, live max 0.91; pass 0.0% both, `nan_policy: fail`). `gate_mode: soft` → permanent ≥50% fusion penalty instead of block; its 2 signals since 06-30 were both rejected | `liquidity_sweep.yaml` |
| 6 | CRITICAL | `rsi_divergence` min-gates on **5 archetypes** (failed_continuation 0.1; fvg_continuation, order_block_retest, retest_cluster, volume_fade_chop 0.05) were **never active in any backtest** (feature absent from both V12 and V14 → nan_policy:skip) but ARE active live, where they block 96–98% of bars. A never-validated filter is the binding live constraint | YAMLs; live pass rates 2.1% / 4.3% |
| 7 | HIGH | wick_trap `instability` gate STILL broken (known issue, unfixed in BOTH config dirs): feature doesn't exist anywhere (live emits `instability_score`; V12/V14 have neither) → nan_policy:skip makes it inert. If fixed, it would block ~14.1% of live bars (instability_score > 0.45) | `wick_trap.yaml:29-33` both dirs |
| 8 | HIGH | `ls_ratio_extreme` scale mismatch live vs store: live = `(ls−1.1)/0.3` fixed approximation (observed 0.2…5.8, **never negative**); V14 = expanding z-score (±5, median 0). Result: funding_divergence gate `≤ −0.5` passes 0.0% live vs 26.3% in store (archetype ~dead live); long_squeeze gate `≥ 1.5` passes 62% live vs 11% in store (far looser live) | `live_feature_computer.py:2245-2249` vs `binance_futures_api.py:1035-1046` |
| 9 | HIGH | `effort_result_ratio` gates (failed_continuation max 1.4, volume_fade_chop max 1.5): real values in V12 (median 0.215), **all-NaN in V14, absent live** → filter exists in backtests, silently inert live (nan_policy:skip) | YAMLs; store column checks |
| 10 | HIGH | Gate-set asymmetry, other direction: `adx_14` and `wyckoff_bullish_score` exist only live (absent V12 + V14). volume_fade_chop's `adx_14 max 25` has nan_policy **default = fail** → auto-fails in any store replay (soft penalty); spring/order_block_retest `wyckoff_bullish_score` gates active only live | YAMLs; `archetype_instance.py:686` (default 'fail') |
| 11 | HIGH | Backtest dir still carries the dead `liquidity_threshold: 0.72` (max reachable = 0.675 in every dataset) for trap_within_trend + wick_trap, while champion dir has 0.43. trap_within_trend actually reads it (`logic_v2_adapter.py:3271`) → backtest and live run different filters. wick_trap's copy is a dead key (never read; only wick_lower/rsi/fusion thresholds consumed at `:3628-3693`) | `configs/archetypes/{trap_within_trend,wick_trap}.yaml` |
| 12 | HIGH | YAML `exit_logic:` sections still ~dead config: only `max_hold_hours` is consumed, and only for **phantom** exits (`v11_shadow_runner.py:1984`) + an assert-only ArchetypeSpec field (`isolated_archetype_engine.py:347`, `archetype_instance.py:175`). `scale_out_levels/pcts`, `runner_pct`, `trailing_*` in YAML never reach ExitLogic — real exits use hardcoded `create_default_exit_config()` defaults | `exit_logic.py:127-190,1192+` |
| 13 | MEDIUM | order_block_retest effectively dead (zero signals ever): soft gates where `any_bos_1h` always fails (#2) and `rsi_divergence ≥ 0.05` fails 96% → chronic ~50-75% fusion penalty; penalized fusion never clears threshold | `order_block_retest.yaml` |
| 14 | MEDIUM | whipsaw effectively dead (zero signals ever): `wyckoff_sow bool_true` — SOW fired 0/533 live bars, 0.09% in V14. Broader: the entire SOS/SOW/LPS/LPSY/UT/UTAD/spring_a/b event family is all-zero in the live window (~3 events expected from V14 base rates; P(0)≈5%) — watch, possibly under-firing live | `whipsaw.yaml`; degenerate sweep |
| 15 | MEDIUM | volume_fade_chop & failed_continuation: joint live gate pass = **0.0%** of 533 bars (rsi_divergence + chop_score ≤ 0.25 / rsi-extreme combos). Both had signals in backtests (gates differ there per #6/#9/#10). Zero signals since 06-30 | computed from live log |
| 16 | MEDIUM | `fusion_smc` low discrimination live: quantized {0.25,0.5,0.75,1.0}, sits at 1.0 for ≥75% of live bars (V14: 50%), changes on only 5% of bars | live log vs V14 |
| 17 | MEDIUM | Degenerate live inputs that vary in stores: `boms_strength`/`tf1d_boms_strength` constant 0 (confluence_breakout threshold key reads boms), `tpi_signal` constant 1.0, `tf1d_coil_breakout`/`tf1d_pti_*` constant 0 (`tf1d_pti_score` also dead in V14). `bars_since_pivot` is quasi-degenerate everywhere (12-19 unique values, saturates at 48) → trap_within_trend's `max 110` gate passes 100% (inert) and the feature itself looks broken | degenerate sweep |
| 18 | MEDIUM | Inert gates that never block (dead weight, or threshold wrong): retest_cluster `temporal_confluence_score min 0.45` (100% pass live, 99.6% V14); trap_within_trend `bars_since_pivot max 110` (100% both) | gate audit |
| 19 | MEDIUM | Regime plumbing split-brain: live_feature_computer emits labels {bull, bear, neutral, crisis} (`:997-1010`); RegimeService emits {risk_on, risk_off, neutral, crisis} (`regime_service.py:598-607`). YAML `regime_preferences` use risk_on-vocab and are consumed by **nothing** in the current mode (`regime_weight_mode` unset → legacy; `archetype_instance.py:282` documents they're not applied). Latent bug: if `threshold_adjustment` mode is enabled, `process_bar` passes the feature-vocab label ('bear') into risk_on-keyed weights → silent no-op. Also `macro_regime` is always 'neutral' (reads `regime_label` before it's computed, `live_feature_computer.py:2539`) | multiple |
| 20 | MEDIUM | Dead / phantom threshold keys: most YAML `thresholds:` entries are never read. E.g. liquidity_vacuum sets `liquidity_score_min, wyckoff_confidence_min, volume_spike_threshold, crisis_threshold, liq_max, vol_z` — none consumed — while code reads `volume_z_min, wick_lower_min, liquidity_max, volume_climax_3b_min` which the YAML doesn't set → hardcoded defaults silently in control. Consumed set is only: fusion_threshold, cooling_period_bars, and the ~30 keys grep-listed in logic_v2_adapter get_threshold calls | YAMLs vs `logic_v2_adapter.py` |
| 21 | MEDIUM | Dead top-level JSON config keys (no reader anywhere): `cooling_periods`, `use_isolated_fusion`, `runner_position_enabled`; `max_positions_by_regime` is now notes-only (replaced by stress-scaled limit) | both JSON configs |
| 22 | MEDIUM | V14 store data quality: `oi_change_4h`/`oi_change_24h` contain `+inf`; `taker_imbalance` live range (−0.88…0.53) wider than store (−0.28…0.73) — different formulas; `chop_score` live median 0.48 vs V14-recent 0.30 (failed_continuation's chop≤0.25 gate: 5% live vs 44% store) | V14 vs live |
| 23 | LOW | `parity_check.py` MONITOR list includes `instability` — a column that exists in neither the live log nor V14 → silently dead monitor row (same class as the wick_trap gate bug) | `scripts/rebuild/parity_check.py:73-76` |
| 24 | LOW | Silent-degradation except-blocks in signal/feature path (all log-and-continue at WARN or lower): wyckoff engine failure → EMA-alignment fallback silently changes `wyckoff_score` semantics (`live_feature_computer.py:1569,1786`); CMI failure → risk_temp=0.5 defaults, threshold silently changes (`:1134`); regime ML model load failure → "crisis_prob will be 0" (`:605`) / permanent-neutral fallback (`logistic_regime_model.py:132`) — note `models/` dir is absent from the repo (deploy risk); funding-rate fetch `except: pass` in run loop (`v11_shadow_runner.py:2532`); heartbeat wyckoff block `except: pass` (`coinbase_runner.py:2766`); oracle synthesis swallow (known, `coinbase_runner.py:2869`) | listed files |
| 25 | LOW | exhaustion_reversal: zero signals since 06-30 but NOT broken — joint gates pass 1.5% of live bars (8 bars); pattern rare in this regime. 4 signals earlier in the log | computed |

---

## Sweep 7: Signal-Path Liveness Since 2026-06-30 (v14rq unlock)

37 signals in the log window. Per-archetype:

| Archetype | Signals since 06-30 | All-time (log) | Verdict |
|---|---|---|---|
| confluence_breakout | 10 (9 allocated) | 51 | ALIVE (mostly via bypass; fusion 0.14-0.19 below threshold) |
| oi_divergence | 6 | 40 | ALIVE |
| liquidity_compression | 6 | 27 | ALIVE |
| trap_within_trend | 6 | 6 | ALIVE (new since unlock) |
| long_squeeze | 2 | 21 | ALIVE (note #8: gates far looser live than in store) |
| liquidity_sweep | 2 (both rejected) | 25 | CRIPPLED — permanent soft-gate penalty (#5) |
| retest_cluster | 2 | 8 | ALIVE |
| spring | 1 | 4 | ALIVE |
| wick_trap | 1 | 1 | ALIVE (instability gate inert, #7) |
| funding_divergence | 1 | 11 | CRIPPLED live (#8) |
| exhaustion_reversal | 0 | 4 | RARE, not broken (#25) |
| failed_continuation | 0 | 2 | DEAD live — joint gates 0.0% (#6, #15) |
| fvg_continuation | 0 | 0 ever | DEAD — impossible hard gate (#3) |
| liquidity_vacuum | 0 | 0 ever | DEAD — impossible hard gate (#4) |
| order_block_retest | 0 | 0 ever | DEAD — chronic soft penalties (#13) |
| whipsaw | 0 | 0 ever | DEAD — SOW never fires (#14) |
| volume_fade_chop | 0 | 0 ever | DEAD live — joint gates 0.0% (#15) |

**7 of 17 archetypes have produced zero signals since the unlock; 5 have never signaled in the log at all**, each traceable to a specific dead feature or unreachable threshold above — not to "pattern rarity" (except exhaustion_reversal).

## Sweep 6: Component Sanity Verdicts (user's specific fear: wyckoff + momentum)

- **wyckoff_score — FUNCTIONING live.** Varies across full 0–1 range (nuniq 31), changes on 6% of bars vs 7% in V14 (sticky by design: 24h rolling max of event confidence, `live_feature_computer.py:1555`). Live median 0 vs V14-2yr 0.558 is regime-attributable (3-week bear window). Note: `wyckoff_score == fusion_wyckoff == trend_strength_score` exactly — the latter is a mislabeled copy (`:2536`), so anything treating trend_strength_score as an independent trend measure is double-counting wyckoff. AR/AS/BC/SC/ST event confidences fire live; the SOS/SOW/UT/UTAD family does not (#14).
- **fusion_momentum — FUNCTIONING live.** Changes every bar (chg_rate 1.00), but compressed: live max 0.47 / median 0.21 vs V14-2yr max 0.88 / median 0.30. Plausibly regime; add to PSI monitoring.
- **liquidity_score — FUNCTIONING, best parity of all.** Live quartiles (0.125/0.125/0.325, max 0.675) match V14 to 3 decimals. Hard cap at 0.675 in every dataset confirms 0.72-class thresholds are structurally unreachable.
- **CMI — FUNCTIONING.** risk_temperature (0.09–0.45) and instability_score (0.10–0.64) vary continuously every bar; observed dynamic thresholds (0.42–0.56) are arithmetically consistent with base + (1−rt)·0.38 + inst·0.15. crisis_prob floor of 0.045 is CORRECT, not stuck: crisis_persistence=1.0 because 60d max drawdown was genuinely < −20% all window (−21%…−29%, real varying data).
- **fusion_smc — WEAK** (#16): pinned at 1.0 most of the time live.

## Verified Working (honest list)

- Gate evaluator itself (`archetype_instance.py:666-764`): min/max/bool/eq/in_range ops, NaN policy, and penalty math all correct; derived OHLC features (wick_anomaly, rsi_extreme_65, upper_wick_body_ratio) compute from columns that exist live.
- ExitLogic `_build_exit_rules` if→elif override bug: FIXED in place (`exit_logic.py:180-186`, comment documents it).
- `_open_position` insufficient-margin silence: FIXED — loud `[PAPER_MARGIN]` warning, never blocks (`v11_shadow_runner.py:1616-1627`).
- Live feature log parity substrate (`coinbase_runner.py:2640-2667`): logs exactly what gates see, NaN/inf-safe, append-only. (Caveat: no timestamp overlap with V14 yet — live starts 06-18, V14 ends 06-10 — so direct row-diff parity is currently impossible; extend V14 or backfill.)
- Live-vs-store agreement is genuinely good for: volume_zscore, rsi_14, atr_percentile, funding_Z, oi_change_4h/24h, fib_time_confluence, temporal_confluence_score, tf1h_pti_score, wick_lower_ratio, bb_width, liquidity_score.
- Signal dedup, cooling periods, adaptive threshold computation, downtrend_skip plumbing, phantom tracking: all wired and consuming their configs.
- Regime classifier config → RegimeService path is wired (`regime_classifier.enabled=true`, probabilistic mode); labels reach `detect()`.

## Top 5 Recommended Fixes (by expected impact)

1. Decide `bypass_threshold` explicitly and fix the contradictory notes (#1).
2. Fix or remove the two impossible wick gates: `wick_exhaustion_last_3b ≥ 1.4` and `wick_lower_ratio ≥ 1.3` — they were presumably meant for a differently-scaled feature (#4, #5).
3. Investigate why live-path SMC BOS/CHoCH flags never fire (V12 proves the pattern exists ~3% of bars) — this single emitter revives fvg_continuation and order_block_retest (#2).
4. Rename wick_trap gate to `instability_score` (and add the column to the store) — one-line YAML fix, 3rd audit that has flagged it (#7).
5. Unify `ls_ratio_extreme` (and `taker_imbalance`) formulas between `live_feature_computer` and `build_v14_store` (#8).

---
*Audit run 2026-07-10 on branch `feat/feature-store-rebuild`. Read-only; no code or config modified.*
