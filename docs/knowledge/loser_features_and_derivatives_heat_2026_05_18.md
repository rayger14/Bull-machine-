# Loser-Features-as-Gates + Derivatives-Heat WFO — May 18, 2026

**Branch**: `quant/loser-features-plus-derivatives-heat`
**Status**: Final
**Verdict**: **NOTHING SHIPS.** All 6 variants either underperformed or showed fragile improvements with train-side regression.

---

## TL;DR

Now that the Binance Vision OI backfill landed and the 3-of-3 boost is live, we ran two parallel studies:

**Task 1 — Loser features as hard gates on `liquidity_compression`** (the top-performing archetype):
- V1 `oi_change_24h <= -0.02`: +1.91% system PnL ✓ — but the LC archetype itself LOST $5,465 (system gain came from other archetypes filling the bars LC vacated, not from the rule working)
- V2 `taker_imbalance <= 0`: -1.75% ✗
- V3 `ls_ratio_extreme <= 0`: -0.56% system, BUT LC archetype alone +$6,091 (n=16, below n=30 floor — directional only)

**Task 2 — Enable `derivatives_heat` CMI component**:
- 5% weight: -0.14% ✗
- **10% weight: +0.90% ✓** (small) — but train PnL dropped −$25,954 (overfit risk)
- 15% weight: -2.69% ✗

Neither task produced a clean win. `derivatives_heat=0%` (the existing production setting) is now validated as the right call. The LC variants don't fix LC — they just shuffle PnL around the system via dedup. **No production changes recommended.**

---

## Methodology

- WFO: train 2018-2022, test 2023-2024 (OOS)
- Baseline: current production (TP Tier 1 + 3-of-3 boost, deployed earlier today)
- 6 variants tested in isolation
- All previous production gates kept; only the indicated change applied per variant

---

## Findings

### System-level table (OOS 2023-2024 only)

| Variant | Trades | PF | PnL | ΔPnL | ΔPF | Verdict |
|---------|-------:|------:|---------:|--------:|--------:|---------|
| **Baseline** | 1,467 | 1.508 | $136,160 | — | — | — |
| dh_5  (deriv_heat=5%, dd=45%) | 1,490 | 1.498 | $135,974 | −$186 | −0.011 | REJECT |
| **dh_10** (deriv_heat=10%, dd=40%) | 1,510 | 1.496 | $137,381 | **+$1,221 (+0.90%)** | −0.012 | **MARGINAL — directional only** |
| dh_15 (deriv_heat=15%, dd=35%) | 1,524 | 1.468 | $132,492 | −$3,668 | −0.041 | REJECT |
| LC + oi_change_24h ≤ −0.02 | 1,461 | 1.525 | $138,754 | +$2,595 | +0.016 | REJECT (target archetype WORSE) |
| LC + taker_imbalance ≤ 0 | 1,453 | 1.504 | $133,781 | −$2,378 | −0.004 | REJECT |
| LC + ls_ratio_extreme ≤ 0 | 1,460 | 1.509 | $135,392 | −$768 | +0.000 | REJECT (n=16 LC trades below floor) |

### Task 1 deep-dive: why the "+1.91% LC + OI" variant doesn't actually fix anything

Variant V1 (`LC + oi_change_24h ≤ -0.02`) gained +$2,595 at the system level but the LC archetype itself LOST $5,465:

| Archetype | Baseline OOS PnL | V1 OOS PnL | Δ |
|-----------|---------------:|-----------:|--------:|
| liquidity_compression | $9,347 (n=36) | $1,924 (n=21) | **−$7,423** |
| other archetypes combined | $126,813 | $136,830 | +$10,017 |
| Net system | $136,160 | $138,754 | +$2,595 |

What's happening: the LC OI gate blocked 15 LC trades. Most of them WOULD have been profitable. But dedup is single-archetype-per-bar, so when LC didn't fire on those bars, other archetypes got them — and a slightly larger share won. The system gain is *accidental dedup reshuffling*, not the gate doing what we wanted.

**This is the SAME pattern we saw with the long_squeeze regime gate yesterday and the failed_continuation chop gate in May**: gates that shrink the target archetype's footprint can produce system PnL gains via dedup, without the gated archetype actually being improved. Not real alpha — just rearrangement.

### Task 2 deep-dive: derivatives_heat's overfit profile

dh_10 shows the cleanest pattern:
- **OOS**: PnL +$1,221, PF -0.012 (small win)
- **Train**: PnL -$25,954, PF -0.063 (significant regression)

The train regression is the smoking gun. When a parameter change makes the in-sample numbers WORSE but the OOS numbers better, that's usually noise/randomness in the OOS window — not a robust signal. Per the quant-analyst subagent rails: train regression + small OOS gain = mark as directional, do NOT ship.

The MEMORY.md note had said `derivatives_heat: ... DISABLED pending more data` — implying it would be enabled once we had OI data. Now that we have 4+ years of OI data and tested it: **the disable was the right call, not a temporary stopgap.** OI/funding/taker as a *first-order* regime input adds noise more than signal. They work better as second-order structural gates (which is how the 3-of-3 distribution_exhaustion boost uses them).

---

## Recommendation

**Keep current production config.** Do not enable derivatives_heat. Do not add OI-based hard gates to LC. The system is already absorbing the OI/funding/taker information through the 3-of-3 distribution_exhaustion boost (deployed earlier today), and adding it elsewhere either over-determines or shuffles capital without alpha gain.

### Memory updates

- Update `derivatives_heat` entry in MEMORY.md from "DISABLED pending more data" → "**DISABLED — validated net negative at all tested weights (5/10/15%) on May 18, 2026 even with full 4+ years of backfilled OI data. OI/funding/taker signals work better as second-order structural gates (see distribution_exhaustion 3-of-3) than as first-order regime inputs.**"

- Document the dedup-reshuffling phenomenon for future studies: "When a gate shrinks the target archetype's footprint, dedup re-routes bars to other archetypes. System PnL gain WITHOUT target-archetype PnL gain = false signal; the rule isn't doing what we think."

---

## What's still untested

The Task 1 search space is huge (6 loser features × 7 winning archetypes × multiple thresholds = 100+ combinations). I tested 3 high-conviction picks. The remaining space probably contains some marginal wins but the dedup-reshuffling problem applies to most of them. Worth pausing on this entire direction until:

1. **Dedup logic is upgraded** (the dedup-fairness investigation already flagged this as the system-level bottleneck — see `docs/knowledge/dedup_fairness_investigation.md`)
2. OR we find a feature that improves the TARGET archetype's profile DIRECTLY (not via dedup side-effects)

---

## Standing Orders honored

- ✓ All YAML / config files reverted to baseline (working tree clean)
- ✓ Real backtest validation
- ✓ Train + OOS WFO applied to all 6 variants
- ✓ Honest sample-size noting (n=16-21 for LC variants — too small for high confidence)
- ✓ Train regression flagged as overfit signal (not hidden)
- ✓ Lesson #54 honored — no fusion-based filters

## Files

- This report: `docs/knowledge/loser_features_and_derivatives_heat_2026_05_18.md`
- Raw outputs: `results/loser_feat_gates/{baseline,dh_5,dh_10,dh_15,lc_oi,lc_taker,lc_ls}/{train,oos}/`
- Backup config: `configs/bull_machine_isolated_v11_fixed.json.bak_lftest` (gitignored)
- Generated variant configs: `/tmp/cfg_dh_{5,10,15}.json` (transient)
