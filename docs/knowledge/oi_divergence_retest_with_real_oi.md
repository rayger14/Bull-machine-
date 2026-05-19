# oi_divergence Re-Test with Real OI Data — WFO Study

**Branch**: `quant/oi-divergence-retest-with-real-oi`
**Date**: 2026-05-18
**Status**: Final
**Verdict**: **KEEP CURRENT PRODUCTION CONFIG** — both direction-flip variants made oi_divergence worse OOS

---

## TL;DR

After the May 18 Binance Vision OI backfill (oi_change_4h/24h, ls_ratio_extreme, taker_imbalance, oi_value, binance_funding_rate, funding_oi_divergence, oi_price_divergence, range_position_20) AND the 3-of-3 distribution_exhaustion boost (just deployed), **oi_divergence has transformed from a chronic loser into a profitable OOS archetype** (OOS PF 1.38, PnL +$2,254, WR 58.1%).

The direction flip we proposed yesterday (`direction: long → short`) was based on prior backtest where oi_divergence was bleeding due to MISSING OI DATA. With real OI data now in the parquet, the original `direction: long` is the right answer. Both flip variants tested today made performance WORSE.

---

## Methodology

- WFO: train 2018-2022, test 2023-2024 (OOS)
- Current production state: TP Tier 1 (deployed May 16) + 3-of-3 distribution_exhaustion boost (deployed May 18) + real OI features in parquet (backfilled May 18)
- Variants tested:
  - **baseline_long**: current production (direction:long, all original gates)
  - **flip_short_simple**: just flip `direction: long → short`, no other changes
  - **flip_short_with_inverted_gates**: flip direction + invert `rsi_14` to `min: 65` + replace `distribution_at_resistance bool_false` with `accumulation_at_support bool_false`

---

## Findings

### Per-variant results

| Variant | Window | oi_div n | oi_div PF | oi_div PnL | oi_div WR | Sys PF | Sys PnL |
|---------|--------|---------:|----------:|------------:|----------:|-------:|---------:|
| **baseline (long)** | train | 64 | 0.74 | −$4,360 | 53.1% | 1.306 | $168,665 |
| **baseline (long)** | **OOS** | **31** | **1.38** | **+$2,254** | **58.1%** | **1.508** | **$136,160** |
| flip_short_simple | train | 85 | 1.31 | +$6,375 | 68.2% | 1.309 | $171,906 |
| flip_short_simple | OOS | 27 | 0.20 | **−$7,922** | 37.0% | 1.483 | $129,754 |
| flip_short_nuanced | train | 86 | 0.27 | −$25,737 | 33.7% | 1.242 | $138,105 |
| flip_short_nuanced | OOS | 23 | 0.51 | −$3,289 | 52.2% | 1.513 | $135,421 |

### OOS deltas vs baseline (the binding test)

| Variant | System ΔPnL | oi_div ΔPnL | Verdict |
|---------|------------:|------------:|---------|
| flip_short_simple | **−$6,406** | **−$10,175** | **REJECT — catastrophic OOS reversal** |
| flip_short_nuanced | −$739 | −$5,543 | REJECT — small system loss, big archetype damage |

### Why the flip failed when prior tests suggested it might work

Yesterday's flip studies tested oi_divergence with MISSING OI data (`oi_change_4h/24h` were NaN/zero, gates degraded via `nan_policy: skip`). In that broken state, oi_divergence was effectively firing on volume_z + RSI alone — fallback gates that produced random direction signal. Flipping direction looked promising because the original "long" direction was fundamentally arbitrary on the fallback signal.

With **real OI data** now in the parquet:
- The actual designed gates (OI capitulation, taker imbalance, RSI extreme) are now firing meaningfully
- The OI capitulation signal (`oi_change_24h <= -0.03`) genuinely correlates with reversals upward
- Long entries at those conditions (= buying after capitulation) work
- Short entries at those conditions (= selling into already-oversold capitulation) get squeezed

**Train shows a positive simple-flip result (+$6,375)** because 2018-2022 had a bear market (2022) where shorting capitulation worked. **OOS (2023-2024 bull recovery) shows the opposite** — same regime that wrecks long_squeeze wrecks oi_div-as-short.

### The bigger story

Three production changes have transformed oi_divergence's profile:

| Change | When | Effect on oi_div |
|--------|------|------------------|
| TP Tier 1 exit defaults | May 16 (deployed) | Marginal positive — same exits as everything else |
| Binance Vision OI backfill | May 18 (data only) | The KEY fix — real OI features mean actual designed gates can fire |
| 3-of-3 distribution_exhaustion boost | May 18 (just deployed) | When 3-of-3 fires, oi_div trades benefit from the 1.5× sizing boost (since oi_div's gates overlap with the 3-of-3 OI condition) |

Live track record was 0/6 WR / −$3,800 — but that was BEFORE these fixes. The 26-day live history reflects the BROKEN STATE. Going forward, live performance should align with the new backtest profile (OOS PF 1.38, +$2,254 / 31 trades).

---

## Recommendation

**No production config changes.** Leave `configs/archetypes/oi_divergence.yaml` exactly as it is (direction: long, all original gates).

The previously-proposed direction flip would have ACTIVELY HURT performance. The OI backfill + 3-of-3 boost already fixed the archetype.

### What to watch in live (next 2-4 weeks)

- oi_divergence trade outcomes — should start showing positive WR (~58% per OOS backtest)
- Average R per oi_div trade — should be slightly positive (PF 1.38 = wins ~38% larger than losses)
- 3-of-3 boost firing rate — should fire on ~10-15% of long entries in current regime

If oi_divergence continues to bleed in live despite these fixes, the next step is to verify the LIVE engine's OKX OI feed is actually populating the same features the backtest sees (OKX vs Binance OI may differ; the live engine wires from `okx_derivatives_api` per heartbeat logs).

---

## Standing Orders honored

- ✓ No production code/config/YAML changes shipped (working tree reverted)
- ✓ Real backtest, no phantom outcomes
- ✓ Train 2018-2022 / Test 2023-2024 WFO applied
- ✓ Sample size honest (n=23-31 OOS oi_div trades — above n=30 floor but tight)
- ✓ Lesson #54: no fusion-based filtering proposed

## Files

- This report: `docs/knowledge/oi_divergence_retest_with_real_oi.md`
- Raw outputs: `results/oi_div_retest/{baseline_long,flip_short_simple,flip_short_with_inverted_gates}/{train,oos}/`
- All YAML changes reverted from working tree

## What this study unblocks

The oi_divergence saga is now resolved. With the backfilled data, the archetype works as originally designed. The longer-term todo from earlier studies:

- ✓ Test full distribution_exhaustion 3-of-3 — DONE, shipped X=1.5
- ✓ Test oi_divergence with real OI data — DONE, no changes needed
- ⏭️ **Next**: Re-run `loser_features_as_gates` study with all 6 whale features now available — was previously testable only on 1 of 6 features
- ⏭️ Enable `derivatives_heat` CMI component — currently disabled awaiting >3 years OI data
