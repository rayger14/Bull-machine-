# Validated-Stack Grid Verdict + BE-Study RETRACTION (2026-07-08)

**Adjudicator:** quant-analyst (rails-on)
**Artifacts:** `results/champion_v14_stack/<cell>/<window>/{performance_stats.json,trade_log.csv}` + `grid_summary.json`
**Harness:** `scripts/champion/stack_validation.py` — REAL engine, production config keys (`downtrend_skip`, `exit_logic.wick_trap.breakeven_trigger_r`), NO monkey-patches
**Grid:** 9 cells, book {wt_only, core2, full16} x protection {none, skip200, skip200_be}
**Windows:** per-year 2018-2024, wfo_train 2018-22, holdout 2025-01..2026-06-10 (untouched)

> **STANDING ORDERS (verbatim):** NEVER turn off bypass_threshold — data collection mode is required for the foreseeable future. NEVER disable any archetype — all 16 stay enabled to collect maximum live signal data. NEVER make production config changes (bypass, disabled_archetypes, thresholds, archetype YAMLs) without explicit user approval. NEVER edit production code/configs directly — recommendations and diffs only.

---

## 1. Executive Summary

- **NO cell passes the pre-registered acceptance.** Every cell breaches the every-year >= -$2K floor; most also breach MaxDD <= 12% or holdout PF >= 1.3. The intuitive "edge + protection" stack does not exist in this grid.
- **RETRACTION: the 2026-06-29 breakeven-study gains were an implementation artifact, not a breakeven effect.** The monkey-patch mutated `pos.stop_loss` directly, which zeroed the R denominator from the next bar onward, silently disabling scale-outs and trailing for every post-BE trade. The "PF lift" was an accidental, un-registered *different exit policy* (ride-to-target with entry-floor stop), not stop protection. The honest production BE implementation changes **1 trade out of ~337 (+$15)**. Both the isolated BE study AND the "core2+BE deployable" full-system result are retracted as BE evidence.
- **skip200 is anti-correlated with wick_trap's edge (Rule 9 co-move FAIL).** On wt_only: train PF 1.37→1.71 UP, holdout PF 1.43→1.10 DOWN. The trades skip removes are overwhelmingly wick_trap WINNERS (2021: 6/6 winners, +$3,858; holdout: 26W/4L, +$5,646). wick_trap's identity is buying washouts below the mean — a below-200d-SMA filter deletes its best entries by construction. REJECT skip for wick_trap.
- **Honest best configs remain the UNSTACKED books:** `wt_only__none` (holdout PF 1.43, +$6.8K, n=70) then `core2__none` (holdout PF 1.30; liquidity_sweep adds +$62 over 35 holdout trades = nothing). Neither passes the acceptance bar; both bleed ~$8-9K in each sustained bear year (2018/2021H2/2022) by design (long-only washout buyer).
- **Per-archetype protection assignment ("skip for junk, none for wick_trap") is moot, not adopted.** skip-on-wick_trap is honestly rejected (Rule 9); skip-on-full16 only reduces bleed on a book that fails the floors regardless — there is no deployable consumer for the assignment.

---

## 2. Methodology

- Real `StandaloneBacktestEngine` runs (V14 full-live-path store, $100K, 2bps commission, 3bps slippage), scratch configs only (`configs/champion/stack_*.json`); production untouched.
- Train = 2018-2022, holdout = 2025-01..2026-06-10 (never tuned on), plus per-year decomposition 2018-2024.
- Pre-registered acceptance (in harness docstring, set before running): holdout PF >= 1.3 AND positive; train PF >= 1.3 (co-move); every year >= -$2K; MaxDD <= 12%; holdout n >= 30; wick_trap's own holdout PF >= 1.3.
- Verification of the BE noop and artifact claims done at trade-log level (`cmp`/diff of trade logs) and via `performance_stats.json` decomposition — not from aggregates alone.

## 3. Findings

### 3.1 Grid (key columns)

| cell | 2018 | 2021 | 2022 | holdPnL | holdPF | holdN | trainPF | fullPF | fullPnL | mdd |
|---|---|---|---|---|---|---|---|---|---|---|
| wt_only__none | -8058 | -7531 | -8667 | 6764 | 1.43 | 70 | 1.37 | 1.46 | 49073 | 14.3 |
| wt_only__skip200 | 1158 | -11389 | 0 | 1119 | 1.10 | 40 | 1.71 | 1.73 | 58976 | 11.1 |
| wt_only__skip200_be | 1158 | -11389 | 0 | 1134 | 1.10 | 40 | 1.71 | 1.73 | 58976 | 11.1 |
| core2__none | -8058 | -10279 | -8667 | 6826 | 1.30 | 105 | 1.48 | 1.41 | 63577 | 17.0 |
| core2__skip200 | 1158 | -14137 | 0 | 749 | 1.04 | 63 | 1.69 | 1.54 | 70790 | 13.7 |
| core2__skip200_be | 1158 | -14137 | 0 | 764 | 1.04 | 63 | 1.69 | 1.54 | 70790 | 13.7 |
| full16__none | -40389 | 14930 | -43156 | -14366 | 0.91 | 695 | 1.11 | 1.16 | 163279 | 51.2 |
| full16__skip200 | -7485 | 16640 | 0 | -8427 | 0.90 | 358 | 1.35 | 1.29 | 196967 | 16.4 |
| full16__skip200_be | -7485 | 16640 | 0 | -8427 | 0.90 | 358 | 1.35 | 1.29 | 196967 | 16.4 |

**ACCEPTANCE: all 9 cells FAIL** (year floor breached everywhere; wt_only__none also fails MaxDD 14.3% > 12%; skip cells fail holdout PF; full16 cells fail nearly everything, holdout negative).

### 3.2 Finding 1 VERIFIED (with one mechanism correction): production BE is a near-noop for wick_trap

- Trade logs `wt_only__skip200` vs `wt_only__skip200_be`: **bit-identical in every 2018-2024 window, wfo_train, and full.** The holdout differs by exactly ONE trade (entry 2025-07-05 23:00): same entry, same exit bar, same reason (stop_loss), exit price $108,145.81 → $108,198.11 = **+$15.75**. That is the entire BE effect (1119→1134, 749→764). Same for core2.
- **Mechanism (corrected):** the operative numbers are `atr_stop_mult: 4.9` and trailing `trailing_atr_mult: 2.0` with **`trailing_start_r: 1.0`** (v14rq YAML; not 0.5 as first stated — the "trailing crosses entry at +0.41R" arithmetic is correct but not operative, since trailing is inactive below 1.0R). What actually kills BE: the BE trigger (+1.0R) and trailing activation (+1.0R) coincide. On the first bar where close-based R >= 1.0, price is entry + 4.9 ATR, so the trailing stop is instantly close − 2 ATR ≈ entry + 2.9 ATR — **2.9 ATR better than breakeven the moment it exists**, then progressively tightened. BE can bind only when a same-bar scale-out return short-circuits the trailing update (exit chain step 3 returns before step 6) and the next bars gap down — which happened exactly once ($15).
- Conclusion unchanged: with wick_trap's current exit architecture there is **no room for BE@1R to do anything**. Bit-identicality is not a wiring failure — the override demonstrably reached ExitLogic (the one $15 trade proves the path fires).

### 3.3 Finding 1 corollary CONFIRMED — the June-29 BE study was an ARTIFACT (RETRACTION)

Mechanism, verified in code:

- Production path: `bin/backtest_v11_standalone.py` builds a **fresh `_PositionAdapter` each bar** with `adapter.stop_loss = tracked_pos.stop_loss`; ExitLogic stop mutations are synced back into `pos.trailing_stop` (lines ~1841-1842), never into `pos.stop_loss`. So the R denominator (`abs(entry - stop_loss)`) stays the true initial 4.9-ATR risk for the life of the trade. Production BE accounting is honest.
- Monkey-patch (`scripts/champion/breakeven_study.py::install_be`, identical code in `breakeven_fullsystem.py`): mutates **`pos.stop_loss` directly** to entry (buffer 0). From the next bar, the adapter's stop = entry → `_calculate_unrealized_r` stop_distance = 0 → **unrealized_r pinned at 0 for the rest of the trade** → scale-outs (need >= 0.5R) never fire again, trailing (needs >= 1.0R) never activates, R-gated distress/invalidation logic altered. Additionally the patch triggered on intrabar HIGH, not close-based R, firing far more often than production ever could.
- **Smoking gun in the artifacts** (`results/champion_v14_breakeven/wick_trap/{baseline,be10_b0}/full/performance_stats.json`): baseline 329 trades, WR 77.5%, avg_win $611, max_win $4,595, avg hold 60.5h → be10_b0 266 trades, WR **61.3%**, avg_win **$944**, max_win **$11,427 (2.5x)**, avg hold 70.6h. A pure stop-ratchet CANNOT increase the maximum winner 2.5x or cut win rate 16 points. Those are the signatures of scale-outs/trailing being disabled post-BE: full positions ride to time-exit (bigger, rarer wins; longer holds; fewer entries because slots stay occupied).
- The June-29 verdict's "clean loss-cut" decomposition (grossP −1.3%, grossL −13.1%) was a coincidence of aggregates masking a wholesale change in winner character (255→163 winners of a different species). Its own "different trade populations" caveat (section 2/7 of that doc) was in fact the entire result.

**RETRACTED as BE evidence (2026-07-08):**
1. `breakeven_study_verdict_2026_06_29.md` headline "PF gains are NOT an accounting artifact / wick_trap mechanistically clean, conditional candidate" — the accounting was internally consistent, but the *mechanism measured was not breakeven*. wick_trap train 1.37→1.56 and holdout 1.43→1.67 are not reproducible by any parameterization of the honest production BE (which produces +$15 total).
2. The un-linked MEMORY.md entry "Breakeven full-system + core2 validation 2026-06-29 … core2+BE holdout 1.30→1.42 (+$8K) … DEPLOYABLE config" — same `pos.stop_loss` mutation in `breakeven_fullsystem.py`, same artifact. There is no validated BE deployment candidate.
3. The production feature `_apply_breakeven_stop` (commit b769c5d) is harmless (off by default, honest ratchet) but its motivating validation is void. The MFE finding that motivated it (14% of losers were +1R before reversing) remains true — but the trailing stop already handles those cases at 1.0R with a 2.9-ATR-better stop; the residual leak is trades reversing between +0.41R and +1.0R, which BE@1R cannot touch either.

**What survives:** the accidental policy the patch actually tested — "after +1R intrabar touch: floor the stop at entry, no further scale-outs, no trailing, ride to time-exit/TP" — posted train 1.56 / holdout 1.67 / full 1.66 with co-move up. That is now an **un-registered, artifact-implemented, one-shot observation**, contaminated by corrupted R internals (distress/invalidation also ran with R=0). It may be re-registered as a NEW hypothesis ("wick_trap ride policy") and tested with a clean production implementation. It is NOT a validated result.

### 3.4 Finding 2 VERIFIED — skip200 deletes wick_trap's winners

- 2021, wt_only: skip removes exactly 6 trades (2021-07-29, 2021-08-06 clusters — the July-Aug washout below the 200d mean), **all 6 winners, +$3,858**; the surviving trade population is unchanged (−$11,389 both cells). 2021 PnL −7,531 → −11,389.
- Holdout, wt_only: skip removes 30 of 70 trades: **26 winners / 4 losers, net +$5,646 removed** (6,764 → 1,119; common trades identical).
- Rule 9 co-move: train PF 1.37→1.71 (up, driven by 2022 going to zero trades and 2018 truncation) vs holdout PF 1.43→1.10 (down). **Classic regime-fit — REJECT skip for wick_trap.** The 2022 "protection" is real but it is the premium paid for the archetype's core behavior; skip charges that premium against future washout winners.
- For full16, skip is a bleed/DD reducer (MaxDD 51→16.4, holdout −14.4K→−8.4K, PnL co-moves up, PF flat 0.91→0.90) — consistent with the 2026-07-02 downtrend study — but the holdout stays NEGATIVE. Skip stops bleeding on a book that shouldn't be funded; it creates no edge.

### 3.5 Finding 3 VERIFIED — unstacked books are the honest best; core2's second leg adds nothing OOS

Per-archetype holdout split (core2__none): wick_trap n=70 +$6,764; liquidity_sweep n=35 **+$62**. Under skip: liquidity_sweep −$370, wick_trap +$1,119. liquidity_sweep is a passenger (confirms risk_overlay_verdict_2026_06_27). The stack intuition — protections validated on the junk book transfer to the edge archetype — is refuted in both directions: skip (helps junk, starves wick_trap) and BE (noop everywhere in production).

## 4. Recommendation

1. **REJECT the stack.** Do not adopt skip200 or BE for wick_trap-centric books. No production change proposed.
2. **RETRACT the June-29 BE conclusions** (both docs' MEMORY entries corrected; retraction banner added to the 06-29 verdict). The `breakeven_trigger_r` engine feature stays as-is (off by default) — do not deploy it as an edge; it has none in this architecture.
3. **Per-archetype protection assignment: NOT ADOPTED (moot).** "None for wick_trap" is an honest Rule-9 rejection of skip, not post-hoc selection. "Skip for full16" has no deployable consumer (book fails floors with or without it). If junk archetypes are ever to be funded, re-test skip per-archetype on data collected after this date, pre-registered. Note: live full-book runs bypass_threshold=True for DATA COLLECTION under Standing Orders — applying skip to logged-but-unfunded signals would corrupt that collection; keep skip out of the live logger.
4. **Deployable ranking (none passes the pre-registered bar — user decision required):**
   - **#1 wt_only__none** — holdout PF 1.43, +$6.8K, n=70, holdout MaxDD 5.2%. Expected behavior: earns in washout-recovery/trend years (2019/20/23/holdout), **bleeds ~$8-9K per sustained bear year (2018/2021H2/2022)**, full-period MaxDD ~14.3% (breaches the 12% floor), full-period PnL $49K (below the $100K system floor). This is a single-archetype satellite, not a system.
   - **#2 core2__none** — only if the user wants liquidity_sweep funded for live-sample accumulation; costs PF dilution (1.30) and MaxDD 17% for +$62 OOS.
   - **Rejected:** every skip/BE cell (Rule 9 or noop); full16 in any dress (holdout negative).
5. **Worth validating next (in order):**
   - **CPCV on wt_only__none** before any funding decision, plus the capital-allocation-overlay wiring (16 logged / 1 funded) already recommended on 06-27.
   - **Re-register the "ride policy" as a new hypothesis:** clean production implementation (per-archetype `scale_out_enabled: false` + entry-floor stop after intrabar +1R + trailing deferred), frozen-entry A/B where feasible, pre-registered acceptance, per-archetype OOS split. One shot; if it fails, drop it permanently.
   - **trailing_start_r sweep for wick_trap (0.5 / 0.75 / 1.0)** — the honest "give-back protection" axis, since trailing dominates BE and the +0.41R..+1.0R window is currently unprotected. Cheap, uses existing production keys.
   - The level-anchored wick_trap redesign (industry study 2026-06-11) remains the highest-ceiling open item.

## 5. Sample Size & Honest Caveats

- Holdout n=70 (wt_only) supports directional conclusions; n=40 (skip cells) is thin. The skip rejection additionally rests on trade-level forensics (32W/4L removed across 2021+holdout), which is stronger than the aggregate PF.
- The BE noop is established at trade-log level (bit-identical files) — not a sampling question. The artifact diagnosis is established in code + stats signatures; the June-29 numbers were never wrong as *numbers*, only as *evidence for breakeven*.
- Holdout 2025-26 contains a −22.8% drawdown leg (mixed regime), but a 2022-magnitude sustained bear would cost wt_only__none roughly −$8.7K again — known, priced in, and the reason the every-year floor fails.
- "Would the ride policy survive a clean implementation?" Unknown — that is precisely why it must be re-registered rather than grandfathered.

## 6. What This Doesn't Test

- No CPCV on any cell (single WFO split + per-year decomposition only).
- No conditional/per-archetype skip cell was run (e.g., full16 with wick_trap exempted) — deliberately, to avoid growing the grid post-hoc.
- BE with triggers below 1.0R (inside the unprotected +0.41..+1.0R window) was not in the pre-registered grid; the trailing_start_r sweep covers the same leak more directly.
- Live parity of downtrend_skip and BE fills; dedup interactions of any future ride-policy in the full book.

## 7. Files

- **This report:** `docs/knowledge/stack_validation_verdict_2026_07_08.md`
- **Retraction banner added:** `docs/knowledge/breakeven_study_verdict_2026_06_29.md`
- **Index corrected:** `docs/knowledge/MEMORY.md` (two BE entries amended, this verdict indexed)
- **Read-only:** `results/champion_v14_stack/**`, `results/champion_v14_breakeven/**`, `scripts/champion/{stack_validation,breakeven_study,breakeven_fullsystem}.py`, `engine/archetypes/exit_logic.py`, `bin/backtest_v11_standalone.py`
- **Production code/configs: UNTOUCHED.**
