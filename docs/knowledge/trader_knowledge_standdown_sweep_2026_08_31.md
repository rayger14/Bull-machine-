# Trader-Knowledge Stand-Down Sweep — 2026-08-31

**Question (user):** does any recorded trader wisdom help with the regime problem
(knowing when NOT to trade)? Full corpus sweep (WI, Mancini, Bojan, Moneytaur,
ZeroIka/Chase) + immediate tests of the three actionable findings.

## Corpus verdict
Every FORECAST-shaped stand-down teaching (ours or theirs) is dead: WI sizing
tiers invert across eras, event-rhythm fails 3/3, dominance reader rejected,
day-type sizing inert, ZeroIka regime reads reduce to killed DXY/CVD families.
What survives from the masters is exactly what survives from our 18 funerals:
**structural refusals + outcome reactions.** No prediction anywhere.

## Tests run today (all pre-registered)
1. **Defect found & fixed:** V23 (and its 2018-2026 extension) NEVER had
   `alt_basket_ret_4h`/`stables_rot_rising` — the exodus-refusal gate (the only
   entry-refusal ever to pass, LIVE in champion_paper since Aug 2) was silently
   disabled in every silo study ever run; silo configs also lacked
   `wt_no_exodus_K`. Columns spliced from V22_CTX (97.3% coverage, tail
   2026-06-10→08-30 NaN=gate-pass, pending DefiLlama recompute).
2. **Exodus gate re-measured (wick_trap, both eras):** 2020 −$5.7K, 2021
   −$10.5K, 2022 −$0.9K, **2023 +$4.4K, 2024 +$3.7K, fresh 2025-26 +$4.5K**
   (PF 0.79→0.89). Positive 3 consecutive eras since stables became structurally
   meaningful. **Live config validated — keep ON.** Corrected fresh book:
   −$28.2K (not −$32.7K).
3. **Mancini "one and done" (win-triggered daily stop): BURIED both eras**
   (−$56.8K discovery / −$13.7K fresh). Our wins cluster in impulses (Aug
   audit: 83% WR at 3+); his rule guards against post-win overtrading into
   chop — wrong organ for this organism.
4. **Boosts 1+2 OFF-ablation (fresh battery, patch verified):** book delta
   +$761 = noise. Not the bleed driver. Un-flagged status + broken
   tf4h_wyckoff_bearish_score (fix unmerged on fix/wyckoff-regime-score)
   remain hygiene items, not urgencies.

## Remaining from the sweep (untested, queued)
- Mancini event blackout as VOLATILITY de-size (30x outsized-bar odds on
  FOMC/CPI) — sizing-shaped only; calendar exists unused in regime_detector.
- WI setup-expiry clock ("window armed, no model forms → stand down") —
  forward-only, n<10 offline; belongs in the trade scorecard build.
- 2+ concurrent positions throttle (live PF 0.42 evidence) — queued since July.

## Standing conclusions (unchanged, sharpened)
Outcome-feedback family: 2-for-2 (K=2/48h stand-down). Prediction family:
0-for-everything. 2025 was a WHIPSAW (42% vol, 27% below-200d) invisible to
bear detectors calibrated on 2018/2022 — which is WHY only outcome rules work.
