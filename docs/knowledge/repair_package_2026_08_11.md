# Live Data-Integrity Repair Package — 2026-08-11

**Branch**: `fix/live-data-integrity` (off `main`)
**Scope**: make the live engine's collected data clean and the config honest.
**Status**: PR-ready for user review. NOTHING merged, NOTHING deployed.
**Sibling PR**: `fix/boms-direction` (BOMS-direction fix, wyckoff_audit addenda 51-53) — separate branch, referenced not duplicated.

---

## 0. HEADLINE FINDING (read this first)

The authoritative audit (`engine_integrity_audit_2026_07_10.md`) was written on the **pre-repair** snapshot. **Most of its repair list was already implemented on `main` three days later** by commit **`7e1f481` "fix: audit-1 repair batch" (Jul 13)** plus `3d2a26e` (oi_divergence gate). This package was scoped from the 07-10 audit, so the bulk of items B, C, D, E were **already done before I started**.

I therefore did NOT blindly re-apply the audit list (that would duplicate merged work and, in one case, re-introduce a doctrine violation). Instead this branch delivers:

- **The genuine remaining deltas** (2 real changes): **A1** (config notes still lie) and **A2** (a gate `7e1f481` *activated* is a current-doctrine violation and must be removed).
- **Independent verification** that the already-merged fixes (B/C/D/E) actually do what they claim, with fresh evidence.
- **Honest recommendations** on the two items the audit wanted "removed" that `7e1f481` instead "rescaled" (C1/C2) — I recommend *keeping* the rescale, not removing.

**Net code/config change on this branch: 4 files, config-only, 0 Python.** `git diff origin/main --name-only`:
```
configs/champion_paper.json                        (A1)
configs/bull_machine_isolated_v11_fixed.json       (A1)
configs/champion/archetypes_v14rq/wick_trap.yaml   (A2)
configs/archetypes/wick_trap.yaml                  (A2)
```

---

## 1. Items that CHANGE nothing at runtime (safe to approve together)

### A1 — Config honesty: bypass_threshold notes now tell the truth  (commit `fix(A1)`)
**What changed**: In the **live** config `configs/champion_paper.json` (run by `deploy/coinbase-paper.service`) and the **canonical backtest** config `configs/bull_machine_isolated_v11_fixed.json`, three note sites each (`_threshold_note`, `_disabled_archetypes_note`, `portfolio_allocation.notes`) claimed *"bypass_threshold=false … bypass is catastrophic … 8 calibrated, 9 disabled"* — a superseded 2026-03-06 production experiment. But both files actually set `bypass_threshold: true` and enable all 16 archetypes. Rewrote all six notes to state the truth: bypass=true BY DESIGN (junk-book data collection, user decision 2026-07-23), dynamic threshold computed/logged but not gated on, fusion has negative predictive power (Lesson #54).
**Why**: audit #1. A config whose own notes contradict its live values is an integrity hazard.
**Expected live behavior delta**: **NONE.** Comments/notes only; no key that any reader consumes was touched.
**Revert line**: `git checkout origin/main -- configs/champion_paper.json configs/bull_machine_isolated_v11_fixed.json`

**Sweep result (all 37 configs carrying the stale note)** — only 3 files' notes contradict their own `bypass_threshold`:

| Config | bypass value | note says | verdict |
|---|---|---|---|
| `configs/champion_paper.json` | **true** | false | **LIVE — fixed** |
| `configs/bull_machine_isolated_v11_fixed.json` | **true** | false | **canonical backtest — fixed** |
| `configs/champion/replay_wt_bypass.json` | true | false | archived bypass-replay experiment — **left untouched** (not live), flagged here |
| 33× other `configs/champion/*.json` | false | false | note is **accurate** for them — left untouched |

---

## 2. Item that CHANGES live behavior (approve deliberately)

### A2 — Remove wick_trap `instability_score` gate: a live-only, never-validated filter  (commit `fix(A2)`)
**What changed**: Removed the `instability_score max 0.45` hard gate from wick_trap in **both** config dirs, replaced with a comment referencing audit #7.
**Why**: This is the one place where `7e1f481` went the **wrong** way per current doctrine. The 07-13 batch renamed the dead `instability` gate to `instability_score` and called it "activated." Measured impact:

| | live feature log (n=533) | backtest store V12 | backtest store V14 |
|---|---|---|---|
| `instability_score` present | **100% of bars** | **absent** | **absent** |
| bars with value > 0.45 (BLOCKED) | **14.1%** | n/a (gate inert) | n/a (gate inert) |

So the gate is **inert in every backtest that validated wick_trap**, but blocks **14.1% of live bars** — on the flagship archetype that carries every validated boost (seller_flow, bojan, breadth, exodus, wyckoff_phase). Running a never-backtested filter as a binding live constraint is exactly the divergence the audit exists to catch, and it violates the boosts-over-filters doctrine (filters 0/9 accepted this era).
**Expected live behavior delta**: wick_trap becomes eligible on ~14% more candidate bars (those it was silently dropping live only). This **restores** the behavior its backtest validated — it does **not** loosen the archetype beyond what was tested. NOTE: this corrects the audit's own text, which (from the pre-rename 07-10 snapshot) called this a no-op; given the 07-13 rename it is **not** a no-op live.
**Revert line**: `git checkout origin/main -- configs/champion/archetypes_v14rq/wick_trap.yaml configs/archetypes/wick_trap.yaml`

---

## 3. Items ALREADY FIXED on `main` (verified, no new change needed)

| Audit item | Fixed by | Independent verification on this branch |
|---|---|---|
| **B1** rsi_divergence min-gates ×5 (failed_continuation, fvg_continuation, order_block_retest, retest_cluster, volume_fade_chop) | `7e1f481` | All 5 YAMLs confirmed to no longer contain any `rsi_divergence` gate (both dirs). |
| **B2** adx_14 / wyckoff_bullish_score live-only gates | `7e1f481` | volume_fade_chop `adx_14 max 25` now has `nan_policy: skip` (store/live symmetric); `wyckoff_bullish_score` removed from spring + order_block_retest. |
| **C3** liquidity_threshold 0.72→0.43 dir sync | `7e1f481` | `configs/archetypes/trap_within_trend.yaml`, `configs/archetypes/wick_trap.yaml`, and `archetypes_v14rq/trap_within_trend.yaml` all = **0.43** (was 0.72 in backtest dir). |
| **D** live BOS/CHoCH emitter dead (0% live) | `7e1f481` | **Replayed the fixed emitter over store OHLCV** (600 bars, rolling 800-bar causal buffer, 2020 window): `tf1h_bos_bullish` **2.50%**, `bearish` **0.67%** — matches the store's ~1–4% magnitude (V12: 1.31%/1.17%); was **0%** before. Direction-sensible; derived CHoCH 1.50%. |
| **E** ls_ratio_extreme scale mismatch | `7e1f481` | Code now maintains `_ls_history` (168h) and emits a true rolling z-score. Store target confirmed: V14 `ls_ratio_extreme` is symmetric z (−5.0/0.0/+5.0), funding_divergence `≤−0.5` store pass **26.3%**, long_squeeze `≥1.5` store pass **10.9%** — exactly the audit-#8 targets the live formula now aims at. |

### D — causality / no-repaint check (3 points)
1. **Causal by construction**: the replay buffer is `ohlcv.iloc[:i+1]` — the emitter never sees a future bar.
2. **Current-bar recency filter**: `_recent_breaks` accepts a break only when `0 ≤ (last_bar_ts − break_ts) < 1 bar`, so a stale break elsewhere in the 1000-bar buffer cannot fire the flag (this is *why* the rate is a sparse ~2–3% and not saturated).
3. **Observed**: sample emitted flag had `break_ts == bar_ts` exactly (`2020-04-15 12:00 == 2020-04-15 12:00`) — the flag aligns to the bar it represents.

---

## 4. Items the audit wanted REMOVED, but that `7e1f481` RESCALED — recommendation: KEEP the rescale

### C1 — liquidity_vacuum `wick_exhaustion_last_3b` (audit #4)
- Audit state: `min 1.4` (mathematically unreachable, feature bounded <1.0) → archetype dead everywhere.
- `7e1f481` rescaled to `min 0.47` (intent = 1.4/3, the feature is a mean of 3 wick fractions).
- **Verified now**: old 1.4 = **0.0%** pass (store+live, confirms it was dead); new 0.47 = **76.7%** store / **76.2%** live pass. The archetype's **joint hard-gate pass is now 21.0% store / 23.3% live** — the wick gate is no longer the blocker.
- **Recommendation: do NOT remove.** The audit's "impossible gate" concern is resolved by the rescale. liquidity_vacuum is still signal-dead live (commit note "still dead"), but with 23% hard-gate pass under `bypass=true` the residual cause is **downstream** (fusion/identity logic in `logic_v2_adapter`, or dedup), not this gate. Removing it would be an unvalidated loosening that wouldn't revive the archetype. **→ Separate open item: root-cause liquidity_vacuum signal deadness in the identity/fusion path (out of this config-honesty package).**

### C2 — liquidity_sweep `wick_lower_ratio` (audit #5)
- Audit state: `min 1.3` (unreachable, feature bounded ≤1.0), `gate_mode: soft` → permanent ≥50% fusion penalty.
- `7e1f481` rescaled to `min 0.5` (intent "wick > body" ≈ 0.5 of range).
- **Verified now**: old 1.3 = **0.0%** pass; new 0.5 = **18.9%** store / **18.2%** live pass — the soft gate now discriminates instead of always-penalizing.
- **Honest downstream finding** (from `7e1f481`'s re-baseline): with the gate repaired, liquidity_sweep's holdout **PF fell to 0.61 (−$8.3K)** — its prior "edge" was an artifact of the broken-gate penalty. So the archetype is genuinely weak now that the gate works.
- **Recommendation: do NOT remove the gate.** The impossible-gate concern is resolved; removing it is an unvalidated change the re-baseline suggests won't help. Keep liquidity_sweep in the full-size data-collection book as-is (per junk-book decision), and treat its weak PF as data, not a bug.

---

## 5. What this package does NOT touch (per instructions)
- **`enforce_gates_under_bypass: false` on confluence_breakout** — deliberate (yaml comment: gates filter out winners). **Left as an open user decision.**
- No archetype disables, no sizing changes, no new filters/boosts, no threshold tuning, no exit-logic changes.
- **audit #12 (YAML `exit_logic:` deadness)** — real but out of scope; noted as a known follow-up.
- Archived experiment configs under `configs/champion/*.json` — left untouched (their notes are accurate; `replay_wt_bypass.json` is the lone archived exception, flagged in §1).

## 6. Things I could NOT fully verify (honest caveats)
- **D live fire-rate cannot be verified from existing live logs** — the live feature logs end 2026-07-10, the emitter fix landed 2026-07-13 and has not been deployed since, so every logged live bar still shows BOS=0. My verification is a **replay of the fixed code over store OHLCV** (proves the code fires ~2–3% and is causal), NOT a live-capture confirmation. A true live check requires a post-deploy log sample.
- **Derived CHoCH is an addition, not a parity match**: the store's `tf4h_choch_flag` is 0.00% even in V12 and `tf1h_choch_detected` is an absent column — the store never populated CHoCH. The live fix *derives* CHoCH from `new_trend != previous_trend`; there is no store ground-truth to match it against.
- **liquidity_vacuum residual deadness** (C1) is diagnosed to the downstream path but not root-caused here — flagged as a separate item.

---

## 7. Verification log (commands + results)
- **JSON valid**: `python3 -m json.tool` passes on both edited JSONs.
- **YAML valid**: `yaml.safe_load` passes on both edited wick_trap YAMLs; hard_gates now `[derived:wick_anomaly, volume_zscore]`.
- **Dry-run live load**: loaded `configs/champion_paper.json` the way `coinbase_runner` does, then `IsolatedArchetypeEngine(config, archetype_config_dir)` — **17 archetypes loaded, no crash**, bypass_threshold=True. (Pre-existing `*_example.yaml` "missing name" warnings are unrelated template files.)
- **Tests**: my diff is **config-only (0 Python files)**.
  - All tests that actually consume the changed configs **pass**: `test_seller_flow_boost.py` + `test_maker_shadow.py` + `test_downtrend_skip.py` = **26 passed, 0 failed**.
  - The pre-existing suite failures (FusionEngine ImportError; `test_integration_fixes.py` import-time `sys.exit(1)`; wyckoff-event, temporal, archetype-validation asserts; `test_config_legacy_v121` loading a nonexistent `config_v1_2_1.json`) are on `main` and **provably independent** of this branch — those tests build configs inline or load unrelated files; none read any of the 4 changed files. **Zero new failures.**

## 8. Standing orders honored
- NEVER turned off bypass_threshold (kept true, corrected the note).
- NEVER disabled any archetype (all 16 stay enabled; A2 *un-blocks* wick_trap bars).
- NEVER made a production change without approval — this is a review branch; nothing merged, nothing deployed.
