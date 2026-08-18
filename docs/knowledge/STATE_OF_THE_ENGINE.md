# STATE OF THE ENGINE — one page of truth (2026-08-11)

*Read-only reconciliation of live server + configs + campaign ledgers. Every claim is sourced (file/log path in parentheses). "UNVERIFIED" means I could not confirm it — not that it's false.*

> **The one-sentence truth.** The live engine is deliberately running an unvalidated 17-archetype "junk book" at full size, on purpose, to harvest forward data — so it is *supposed* to bleed modestly. Underneath that noise sits exactly **one proven money-maker (wick_trap) plus two thin real edges, six validated sizing boosts, and repaired sensors.** Everything else is either a data-collection instrument or an offline research pile.

---

## 1. WHAT IS LIVE RIGHT NOW

- **Service**: `coinbase-paper`, active, up since **2026-08-06 21:36 UTC** (~5 days), no crashes in 3 days (`systemctl status coinbase-paper`; journal). Deploys via **rsync, not git** — server git is stale, so ignore the server's git commit (server audit §4).
- **Config**: `configs/champion_paper.json`, archetype dir `configs/champion/archetypes_v14rq/` (verified live values, not the notes).
- **Capital**: wallet **$2,000,000** — a *measurement* wallet, not real money. Sized 20× on 2026-08-06 so margin never crowds out signals; per-trade **dollars are unchanged**: ~$2,000 risk, $35K margin cap, $52.5K notional at 1.5× leverage (`champion_paper.json` `initial_cash` + `_initial_cash_note`; `position_sizing`). *Note: the service still passes a stale `--initial-cash 100000` flag that the config overrides — harmless, worth cleaning (`deploy/coinbase-paper.service`).*
- **Bypass mode = TRUE, by design.** Sub-threshold signals trade for data collection; the dynamic threshold (~0.34 live) is computed and logged but **not gated on** (`champion_paper.json` `adaptive_fusion.bypass_threshold: true`). This is a Standing Order — do not turn off.
- **Archetypes enabled: all 17** (`disabled_archetypes: []`). Dedup = 1 long + 1 short per bar (`signal_dedup: best_per_direction`).
- **The 6 boosts (all live, all 1.25× sizing on wick_trap/long context):** seller-flow, bojan-wick, breadth/LOCAL-flush, exodus-refusal, wyckoff-phase (C_accum), rotation-calm — see §2a for evidence (`champion_paper.json` boost blocks).
- **Exits**: Smart Exits V2 hardcoded defaults (168h hold, scale-outs [0.5R/1.0R/2.0R], chop-aware trailing, 4/5 composite invalidation + distress). YAML exit ladders are **dead config** — only `max_hold_hours` is read (`engine/archetypes/exit_logic.py`; audit #12). Breakeven-1R is **NOT** deployed (retracted twice).
- **Live PnL since inception (2026-03)**: realized **−$16,384, PF 0.80, WR 67.7%, MaxDD −32.7%**, 5 open longs (`results/coinbase_paper/performance_summary.json`, 2026-08-11; cross-check `trade_outcomes.csv` −$15,855 / PF 0.77). **The owner's "−$24.5K / PF 0.72" is a *backtest* figure** for the 2025-26 holdout window (wyckoff_audit add.45), not the live number — live is milder (~−$16K).
- **WHY that's expected**: the junk book at full size **is** the drawdown. A validated-only counterfactual (wick_trap+LC+order_block_retest trades only) = **+$2,355, MaxDD −4.9%** vs the actual −$16K / −31% (`live_emergent_mining_2026_07_21.md`). The book's job right now is *forward data*, not paper PnL (user decision 2026-07-23, `project_junk_book_full_size_decision`).

---

## 2. THE THREE THINGS THE ENGINE CURRENTLY IS

### (a) VALIDATED STACK — the real edge (what actually makes money)
| Component | Status | Validation evidence (one line) |
|---|---|---|
| **wick_trap** (buy the liquidity flush) | LIVE champion | Holdout PF **1.43** (n=70), V15 holdout **1.71**, CPCV **15/15** positive, worst alternate history +$2,421 (`wick_trap_cpcv_2026_07_09.md`) |
| **liquidity_compression (LC)** | LIVE, thin edge | Holdout PF **1.14/1.14** perfect co-move, live PF **1.20–1.30** on track (`lc_battery_2026_07_14.md`, `live_evidence_engine`) |
| **order_block_retest** | LIVE since 07-17 | V15 holdout PF **2.08** (n=31), CPCV 15/15 median 1.39, OOS≥train, never tuned (`v15_structure_verdicts_2026_07_17.md`) |
| Boost 1 — seller-flow | LIVE | 07-28: wick_trap 1.25× on taker_imbalance≤0 flush; train 1.28→1.34, holdout 1.16→1.18 |
| Boost 2 — bojan-wick | LIVE | 07-28 incremental: train +$5.6K, holdout +$831 |
| Boost 3 — breadth/LOCAL-flush | LIVE | 08-01: 1.25×; train +$11.3K, holdout +$3.1K |
| Boost 4 — exodus refusal | LIVE | 08-02: refuse GLOBAL-flush + stables-rotation; train 1.36→1.56, holdout 1.26→1.51, DD better both |
| Boost 5 — wyckoff-phase (C_accum) | LIVE | 08-04: 1.25×, shadow phase, zero fusion leak; train +$8.6K, hold +$769 |
| Boost 6 — rotation-calm | LIVE | 08-06: 1.25×; 3/3 on non-binding wallet, cohort PF 1.46/1.94/1.52 |
| **RTZ filter** (retest-zone hold) | validated OFFLINE, not in live config | Real keeper: train PF 2.02→2.38, OOS-A flips to pass (`wyckoff_audit` add.45) — a study win, **not yet deployed** |
| **Time layer** (crowd-out governor) | validated, live status UNVERIFIED | "deployed net-positive 3/3" per commit `237731f`, but no distinct key in `champion_paper.json` — treat as UNVERIFIED live |

*(The "wick_trap + exhaustion_reversal champion pair" from June is superseded: the pair **failed** the 2025-26 pristine holdout; the deployed champion is wick_trap **standalone** — `champion_strategy_pair_2026_06_10.md`.)*

### (b) DATA-COLLECTION INSTRUMENT — the junk book
The other ~14 archetypes run at **full $52.5K size** with **bypass ON**, expected to bleed, on purpose. They generate the forward signal/feature/outcome data every study needs (`project_junk_book_full_size_decision`, user 2026-07-23). Do **not** re-propose sizing cuts. Their losses are the cost of the data, not a bug.

### (c) RESEARCH PILE — offline only
**~160 branches, 37 configs.** Dozens of `study/*` and `quant/*` branches hold rejected or parked experiments (`git branch`). None are live. The two that matter for a decision are the pending fixes in §5; the rest are archive (see §6 rejected-closed list).

---

## 3. PER-ARCHETYPE TRUST TABLE (all 17)

*Signals column = live signal log since the 06-30 unlock (`engine_integrity_audit_2026_07_10.md` Sweep 7) unless noted. Verdict: TRUSTED = validated edge; INSTRUMENT = kept full-size for data, no proven edge; DEAD = cannot fire / never fires; BROKEN = wired wrong.*

| Archetype | Live signals | Validated? (test) | Verdict |
|---|---|---|---|
| wick_trap | firing (1 in log window; carries all 6 boosts) | YES — holdout 1.43, CPCV 15/15 | **TRUSTED** |
| liquidity_compression | 6 (+ live PF 1.20–1.30) | YES — holdout 1.14 co-move | **TRUSTED** |
| order_block_retest | live-capable since 07-17 fix | YES — V15 holdout 2.08, CPCV 15/15 | **TRUSTED** |
| trap_within_trend | 6 (3 open now) | No — V12-era artifact edge | INSTRUMENT |
| retest_cluster | 2 (1 open now) | No standalone edge | INSTRUMENT |
| exhaustion_reversal | 0 recent (rare, not broken) | No — holdout 0.85; pair failed holdout | INSTRUMENT |
| confluence_breakout | 10 (mostly via bypass, fusion 0.14–0.19) | No — no breakout test; no-chase failed OOS | INSTRUMENT |
| long_squeeze | 2 (gates looser live than store) | No — short archetype, PF<0.7 on bull data | INSTRUMENT |
| funding_divergence | 1 (crippled live pre-07-13) | Weak-real, starved — PF 1.53 n=6 | INSTRUMENT |
| spring | 1 | REJECTED — dip-buyer, accepts UTAD tops | INSTRUMENT |
| liquidity_sweep | 2 (both rejected by gate) | REJECTED post-repair — holdout 0.61 (old "edge" was a broken-gate artifact) | INSTRUMENT |
| oi_divergence | 6 | REJECTED live — PF 0.23, −$6.5K; OI gate anti-selects | DEAD (leak) |
| fvg_continuation | 1 open now (post-V15) | REJECTED — V15 holdout 0.71; was BOS-dead | DEAD-ish |
| failed_continuation | 0 — joint gates 0% live | effort_result_ratio backtest-only, skipped live | BROKEN |
| liquidity_vacuum | 0 ever | Gate rescaled 07-13, still signal-dead (downstream identity/fusion) | BROKEN |
| whipsaw | 0 ever | `direction:neutral` → `detect()` returns None; SOW never fires | BROKEN |
| volume_fade_chop | 0 ever | `direction:neutral` → returns None; joint gates 0% | BROKEN |

*Live open positions right now (`state.json`, 08-11): 3× trap_within_trend, 1× retest_cluster, 1× fvg_continuation — all long. Three concurrent trap_within_trend is exactly the concurrency-pileup risk flagged in `live_emergent_mining_2026_07_21.md` (2+ open → PF 0.42).*

---

## 4. SENSOR / PLUMBING STATUS

| Sensor | Status |
|---|---|
| **Structural range** | VALIDATED, working (`engine_integrity_audit` Sweep 6; liquidity_score best parity of all) |
| **spring_b causal fix (V20)** | DEPLOYED 2026-08-05 (PR #61) — look-ahead bug fixed, causal, 4/4 causality tests pass (`wyckoff_audit` §V20) |
| **BOS/CHoCH emitter** | **FIXED (main 07-13, clobber re-fixed 07-16) AND CONFIRMED LIVE** — fires non-zero in Aug live logs: tf1h_bos 2.4%/1.2%, choch 0.8%; tf4h fired in July (rare ~0.7%), quiet in Aug (server audit §3, `results/coinbase_paper/live_features/2026-08.jsonl`). This resolves the audit's #2/#3 CRITICAL and the repair-package's "UNVERIFIED live" caveat. |
| **BOMS direction** | FIXED on branch `fix/boms-direction`, **not merged, not deployed** (`wyckoff_audit` add.51). Was dead (direction discarded on displacement-only breaks). First fair HTF-state test shows a promising n=35 pulse beating a plain trend filter (add.52) — unproven. |
| **Regime columns** (regime_label/risk_on/off/crisis) | EMPTY (all-NaN) in offline study store — CMI RegimeService runs live but its output was never materialized to the store (`wyckoff_audit` add.50) |
| **tf1h_ob / fvg store columns** | all-NaN / frozen historically; SMC family only revived on the V15 store rebuild (`v15_structure_verdicts`) |
| **Known-dead features** | `tf4h_fvg_present`, `fusion_smc` (pinned ~1.0 live), `tpi_signal` (1.0), `macro_regime` (pinned 'neutral'), boms_strength (~0.7%), tf1d_pti block, wyckoff_bullish_score (froze 07-07) (`engine_integrity_audit` #17, `live_emergent_mining` #7) |

**Punchline**: the sensors traders care about most (BOS/CHoCH, spring, structural range) are now genuinely alive after months of being dead — but *fixed sensors ≠ edge*. The plain BOS-retest still loses (order_block_retest via plain trigger PF 0.75); the fix enables fair tests, it doesn't create money (`wyckoff_audit` add.51–52).

---

## 5. PENDING DECISIONS FOR THE OWNER

| Decision | Recommendation |
|---|---|
| Merge **`fix/boms-direction`**? | **YES, low-risk merge.** Pure detection fix; only consumer is knowledge_hooks (still gated on unchanged `boms_detected`); no archetype reads the revived feature, so it can't change live behavior — it just un-deads an HTF signal for future study (add.51). Store rebuild still needed before it's usable offline. |
| Merge **`fix/live-data-integrity`** (A1 config-honesty + A2 remove wick_trap `instability_score` gate)? | **YES on A1 (zero runtime change — just stops the notes from lying).** **A2 is a real live change**: it un-blocks ~14% of live bars on the flagship archetype, *restoring* the behavior wick_trap's backtest validated (a never-backtested filter is currently binding live). A2 is doctrine-correct (filters 0/9 accepted) — recommend YES, but approve it deliberately as a behavior change (`repair_package_2026_08_11.md` A1/A2). |
| **Deploy** after merging? | Deploy is a separate explicit go (standing rule 2026-08-01). If A2 merges, a deploy is what makes it live. Recommend: merge both, then one deploy — and log a post-deploy live sample to confirm A2's effect and re-confirm BOS/CHoCH. |
| `enforce_gates_under_bypass` on **confluence_breakout** (currently `false`)? | **Leave FALSE (keep open).** CB is gate-immune (soft + bypass_fusion coexist); turning it on filters winners and mode-flips are 0-for-1 empirically. Not worth touching under the data-collection mandate (`repair_package` §5; Rule 10). |

*All four honor the Standing Orders: bypass stays TRUE, no archetype disabled, nothing merged/deployed without your explicit go.*

---

## 6. WHAT IS PROVEN vs WAITING

**PROVEN (deployed, cleared every honest test):** wick_trap champion; LC + order_block_retest (thin second/third edges); the 6 sizing boosts; the repaired sensors (BOS/CHoCH live, spring_b causal, structural range). Fusion is a *confirmed anti-signal* (Lesson #54, r=−0.08, replicated 4× incl. live) — the edge is structural detection + boosts, never fusion score.

**CROSS-ASSET-VALIDATED, AWAITING FORWARD DATA — the trend-continuation door:** buy the pullback *after* a confirmed up-break (needs no Wyckoff store, self-regime-filters). Standalone on 4 assets, identical BTC-tuned params: BTC PF **2.56** / SPX **3.53** / NDX **2.82** / Gold **3.51** (uncorrelated) (`wyckoff_audit` add.48). **But**: too rare to prove on history (~1–4 trades/yr/asset) and it has a known **bear-market-rally breakout leak** (worst on Gold) that no cheap offline regime overlay robustly plugs (add.49–50). Verdict = **forward-collection only, do not deploy**; the campaign's best surviving candidate.

**REJECTED-CLOSED (do not reopen):**
- **Fusion as a quality filter** — negative predictive power, 4 replications incl. live (Lesson #54).
- **Dip-buying / springs at discount** — intrinsically markdown-fragile on *every* asset (SPX/Gold/BTC bears all 0% WR); spring accepts distribution tops (`wyckoff_audit` add.47).
- **PO3** — killed on all 7 angles; bottleneck is Wyckoff *detection* quality, not the concept (`project_po3_fully_rejected`).
- **Filters / hard gates generally** — 0-for-9 accepted this era; dedup reshuffles them into false wins (Rules 7–10).
- **Fine Wyckoff phase / distribution detection** — un-buyable, un-trainable (14-label scarcity), un-LLM-able, and genuinely absent at crypto tops; desks use coarse regime gates instead (`wyckoff_audit` add.49).
- **HTF-state + LTF-trigger expansion** — the built version was rejected; a *new* n=35 pulse post-BOMS-fix (add.52) is promising but **unproven** — a future re-test, not a reopening.
- **Short mirror** — no short edge; being below-200d does not predict downside (fwd returns ~coin-flip); the only working bear tool is a defensive *skip*, not a short (`wyckoff_audit` add.47, `downtrend_study_2026_07_02`).

---

### STANDING ORDERS (reproduced verbatim)
- **NEVER turn off bypass_threshold** — data collection mode is required for the foreseeable future.
- **NEVER disable any archetype** — all 17 stay enabled to collect maximum live signal data.
- **NEVER make production config/code changes** (bypass, disables, thresholds, YAMLs) without explicit user approval — recommendations and diffs only.
- **Always ask before deploy** — prior approvals do not carry forward.

*Sources reconciled: live server (`165.1.79.19`, read-only 2026-08-11), `configs/champion_paper.json`, `configs/champion/archetypes_v14rq/`, `docs/knowledge/{wyckoff_audit, MEMORY, engine_integrity_audit_2026_07_10, repair_package_2026_08_11, champion_strategy_pair_2026_06_10, v15_structure_verdicts_2026_07_17, trader_knowledge_audit_2026_07_16, live_emergent_mining_2026_07_21}.md`. Config notes were **not** trusted where they contradicted live values (they lied on bypass mode and archetype count). Nothing on the server was modified.*
