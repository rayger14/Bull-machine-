# long_squeeze Forensic — May 18, 2026

**Branch**: `chore/quant-lessons-v2-codify`
**Status**: Final
**Verdict**: **Boost-based rule proposed (NOT applied) — gate-immune architecture confirmed; existing hard_gates are inert in live**

---

## TL;DR

Live `long_squeeze` is bleeding (PnL −$617, PF 0.50, n=4 unique entries). Forensic on the 4 trades reveals:

1. **Gate violations in live**: 2 of 3 losing trades had `rsi_14 < 60` despite the YAML requiring `rsi_14 >= 60`. With `gate_mode: soft` + global `bypass_threshold: true`, the gate only penalizes fusion, and fusion gets bypassed → gate is effectively unenforced. This is Rule 10 (gate-immune architecture) in production.

2. **The 1 winner's bar-level signature is striking**: it had extreme RSI (73.9) AND extreme volume (vol_z 2.81) — well above the lone-loser threshold. The other 3 entries (losses) had mid-range RSI (29.5–64) and mid-range vol_z (−0.59 to 1.36).

3. **Filter doesn't help here (Rule 8 says: try boost instead)**. The empirical pattern: long_squeeze fires too freely because gates are inert. The right structural fix per the new rails is a sizing **boost** for the high-confidence sub-condition (RSI >= 70 AND vol_z >= 2.0), not another filter attempt.

Sample n=4 is well below the n=30 floor — diagnosis is directional only. No production change recommended yet. Need 30+ live trades or a backtest with the same trigger logic to validate.

---

## Per-trade detail (live, Apr 23 – May 7, 2026)

| Date | PnL | Direction | RSI | fund_Z | wyck_4h_bear | vol_z | Gate violations |
|------|----:|-----------|----:|-------:|-------------:|------:|-----------------|
| Apr 23 | −$459.58 | short | **48.5** | 1.98 | 0.928 | 0.80 | RSI 48.5 < 60 (gate min) |
| May 1 | −$346.08 | short | 64.0 | 1.51 | 0.928 | **−0.59** | all required gates passed; but vol_z negative (opposite of squeeze pattern — buyers fading) |
| May 4 | **+$616.54** | short | **73.9** | 2.09 | 0.928 | **2.81** | — WIN |
| May 7 | −$428.85 | short | **29.5** | 1.81 | 0.928 | 1.36 | RSI 29.5 < 60 (gate min) |

The winner had:
- **RSI 73.9** (the only entry above 70 = true overheating)
- **vol_z 2.81** (more than 2× any other entry — climactic volume)
- fund_Z 2.09 (highest, but all 4 had funding >1)

The losers shared:
- Mid-range or weak RSI (3 of 4 ≤ 64; 2 violated the gate min)
- Vol_z below 1.5 for 2 of 3 (no climax)
- wyck_4h_bear = 0.928 across ALL 4 (the persistent regime score is not the differentiator — it's flat across wins and losses)

---

## Diagnosis: gate-immune architecture in live

The configured hard_gates on `long_squeeze.yaml`:
- `funding_Z >= 0.5`
- `rsi_14 >= 60`
- `ls_ratio_extreme >= 1.5`
- `funding_oi_divergence == -1`
- `vol_shock <= 0.10`
- `derived:accumulation_at_support == false`

With `gate_mode: soft` (`configs/archetypes/long_squeeze.yaml:50`), a failed gate just penalizes fusion (`engine/archetypes/archetype_instance.py:733-746`). With global `bypass_threshold: true`, fusion threshold check is skipped (`bin/live/v11_shadow_runner.py:1009 onward`). Combined: hard_gate violations don't actually block trades in live.

This matches the Apr 23 and May 7 losses — both fired with RSI well below 60. The yaml's intent ("require RSI >= 60 for entry") is silently violated.

This is **Rule 10** (gate-immune architecture) in real-world action. We documented the pattern for CB earlier; long_squeeze has the same architecture (soft mode + bypass_threshold globally on).

---

## What the May 17 regime-gate experiments missed

The May 17 follow-up study (`docs/knowledge/three_fix_followups_2026_05_17.md`) tested 3 regime-gate variants:
- `ema_slope_50 <= 0`
- `tf4h_wyckoff_bullish_score < 0.5`
- `funding_Z >= 1.5`

All 3 made things WORSE in OOS. Why? Because regime signals don't differentiate the 4 live trades — all 4 had identical wyck_4h_bear (0.928) and similar fund_Z (1.51–2.09). **The differentiator is bar-level, not regime-level.** The May 4 winner stood out on `rsi_14` and `vol_z` — neither is a regime signal.

This validates the May 17 conclusion ("forensic on the winners is the next step, not more regime gates") and Rule 8 (filter→boost reframe — filters keep failing because the loser-vs-winner discrimination is wrong-scale).

---

## Recommendation (boost-based; NOT applied)

Use the `scripts/dist_exhaustion_boost/run_variant_3of3.py` pattern as the wiring template. Define the long_squeeze high-confidence sub-condition:

```
when (signal.archetype_id == 'long_squeeze'
      AND signal.direction == 'short'
      AND rsi_14 >= 70.0
      AND volume_zscore >= 2.0
      AND funding_Z >= 1.5):
    intent.allocated_size_pct *= X
```

Test X ∈ {1.0, 1.25, 1.5, 1.75, 2.0} via WFO. **Both train AND OOS PnL must improve, and long_squeeze archetype PnL specifically must improve (not just system PnL via dedup-shuffling) — Rules 7 + 9.**

**Critical caveat**: this is a BOOST not a FILTER. It does NOT block low-conviction long_squeeze trades. Those will continue to bleed if the gate-immune architecture remains. Two separate paths:
- **Boost path**: amplify the high-conviction subset (this recommendation)
- **Architectural path**: fix the gate-immune issue (Rule 10 — flip soft to hard, OR add a new evaluation mode, OR convert to a derived feature). Both options were rejected for CB on May 17 (hard mode over-blocks; tightening is no-op). Same pattern likely applies here.

Recommend testing the boost first.

---

## Why no production change today

- **n=4 unique live trades** is far below the n=30 floor (Rule from Methodology section). The pattern is suggestive but not statistically separable from noise.
- **No backtest validation yet** — the bar-level features that differentiate the winner (RSI 73, vol_z 2.8) may not have been the criteria in the 25 historical long_squeeze trades cited in `MEMORY.md:42` (PF 0.13).
- **Standing Order**: no production config changes without explicit user approval.

Next step is a backtest variant testing the proposed boost on 2018-2022 train + 2023-2024 OOS, with explicit per-archetype PnL reporting (Rule 7) and train+OOS co-movement check (Rule 9).

---

## Files

- This report: `docs/knowledge/long_squeeze_forensic_2026_05_18.md`
- Live trade source: `/tmp/trade_outcomes_live.csv` (refreshed May 18 from server)
- Long_squeeze YAML: `configs/archetypes/long_squeeze.yaml`
- Identity gate: `engine/archetypes/logic.py::_check_S5` (lines 976-998)
- Boost wiring template: `scripts/dist_exhaustion_boost/run_variant_3of3.py`

## Constraints Honored

- ✓ READ-ONLY (no YAML changes)
- ✓ Standing Orders intact
- ✓ Real live data (n=4 — explicitly flagged as below n=30 floor)
- ✓ Rules 7 + 8 + 9 + 10 applied in the analysis
- ✓ Boost reframe per Rule 8
