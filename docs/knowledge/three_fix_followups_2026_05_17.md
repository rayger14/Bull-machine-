# Three-Fix Follow-Up Studies — May 17, 2026

**Branch**: `quant/three-fix-followups` (no production changes — all 6 variants rejected)
**Baseline**: post-TP-Tier-1 merge (`feat/tp-tier1-defaults-validated` shipped May 16)
**Verdict**: 0 of 6 variants accepted. Three structural findings emerge from the rejection pattern.

---

## TL;DR

We tested 6 variants across the 3 rejected fixes from yesterday:
- **Fix 1 (oi_divergence full direction rework)**: ditto-rejected
- **Fix 2B/C (long_squeeze regime + selectivity)**: rejected
- **Fix 3A/B/C (CB hard-gate tightening)**: 2 no-ops + 1 mild regression

None ship. But the *rejection pattern* reveals structural problems that demand different approaches.

---

## Verdict Table (OOS 2023-2024 vs new baseline $131,004)

| Fix | System ΔPnL | Target archetype outcome | Verdict |
|-----|------------:|--------------------------|--------|
| 1 oi_div FULL flip | +$2,135 (+1.63%) | oi_div OOS PnL went DEEPER negative (−$662 → −$2,780) | **REJECT** — system gain is incidental, archetype itself is worse |
| 2B long_sq + Wyckoff regime gate | −$2,981 (−2.28%) | long_sq WR collapsed 42% → 19%, PnL −$2,722 → −$5,703 | **REJECT** — same failure mode as ema_slope variant: gate filtered wins |
| 2C long_sq funding_Z 0.5 → 1.5 | **+$0 (no-op)** | long_sq metrics identical | **REJECT** — gate had no effect (no live trades hit funding_Z that high in backtest) |
| 3A CB volume_z 0.5 → 1.0 | **+$0 (no-op)** | CB metrics identical | **REJECT** — see "CB gate impasse" below |
| 3B CB atr_pctile 0.40 → 0.30 | **+$0 (no-op)** | CB metrics identical | **REJECT** — same root cause |
| 3C CB add ema_slope_50 > 0 | −$989 (−0.76%) | CB PnL slightly worse | **REJECT** — same root cause + extra noise |

---

## Three structural findings

### Finding 1: oi_divergence loses regardless of direction

Three attempts now, all lose money for the archetype itself:
- Original (direction: long): live OOS −$1,511 / 42 trades
- Simple flip (direction: short, RSI gate unchanged): −$3,173 / 55 trades
- **Full rework (direction: short + RSI min:65 + accumulation_at_support gate)**: **−$2,780 / 72 trades**

The full rework is better than the simple flip (less catastrophic loss, more trades), but still −$2,780. The system PnL benefits when oi_div is wired as short because it reduces dedup competition with profitable long archetypes — but that's incidental, not a fix.

**Implication**: oi_divergence's underlying *detection* is suspect. It may be picking up noise rather than real smart-money flow. The historical parquet is missing the actual OI columns (`oi_change_4h/24h` are all zero/NaN), so the archetype is firing on the FALLBACK gates (volume_zscore + RSI), which aren't what it was designed to detect.

**Real fix path**: requires the Binance OI backfill to even test the archetype's actual hypothesis. Until then, oi_divergence should arguably be **demoted to shadow mode** (Standing Order forbids disabling, but logging-only mode might be the intended state).

### Finding 2: long_squeeze regime/selectivity gates fail because of selection asymmetry

Three regime/selectivity attempts now, all failed in the same direction:
- ema_slope_50 ≤ 0 (yesterday): WR 50% → 8%
- tf4h_wyckoff_bullish < 0.5 (today): WR 42% → 19%
- funding_Z ≥ 1.5 (today): no effect — gate didn't actually filter

The pattern: **whenever we filter long_squeeze trades by a regime signal, we filter out the WINS, not the losses.** This means the winning long_squeeze trades occur *during* the same regimes we'd intuitively want to block (bull trends with rising EMA slope and bullish 4H Wyckoff). They're contrarian-correct trades that go against the trend AND work.

**Implication**: regime gates are the wrong tool. The signal that distinguishes winning long_squeeze trades from losing ones isn't a regime variable — it's likely something at the bar-level (e.g., the wick structure, the volume profile at the squeeze, the specific liquidations cascade).

**Real fix path**: forensic analysis on the winning trades — pull each one, look at the *bar-level features* that differed, build a gate from data. This is hours of work, not minutes.

### Finding 3: CB's soft+bypass arrangement is gate-immune

Fix 3A and 3B (tightening volume_zscore and atr_percentile) produced **zero changes** — same trade count, same PnL, same WR. This wasn't a coincidence.

Root cause: CB has `gate_mode: soft` AND `bypass_fusion_threshold: true`. In soft mode, failing hard_gates only penalize fusion. Since CB bypasses the fusion threshold check entirely, the penalty has no effect. So **any hard_gate tightening on CB is structurally a no-op while these two flags coexist.**

Yesterday's Fix 3 attempted `gate_mode: soft → hard` (which DID block) and over-blocked (-$17,885). So the impasse is:
- Soft mode + bypass: gates do nothing
- Hard mode: gates work, but block 78% of profitable trades

**Real fix path**: needs a third option that doesn't exist yet. Options:
- A new gate evaluation mode: "hard if X structural condition met, soft otherwise" (e.g., hard only when fusion < 0.10)
- A specific CB-quality derived feature (e.g., "compression-then-real-expansion" detector) used as a single high-precision hard gate
- Remove `bypass_fusion_threshold: true` from CB and add a much lower fusion_threshold (e.g., 0.08) instead — would re-introduce the fusion filter that was originally bypassed for being too restrictive

None of these are 1-line fixes. Each is a focused 1-2 hour study.

---

## What's still worth doing

1. **Binance OI backfill** (2-4 hour script) — unblocks oi_divergence's actual designed behavior + enables several other studies. Per yesterday's report, this is the highest-leverage data investment.
2. **Forensic on the winning long_squeeze trades** — pull each winner from the live trade log, look at bar-level differentiators, build a gate from observed patterns. Fast (~1 hour).
3. **CB design rethink** — pick one of the three options above, scope a focused study. Medium effort (~3-4 hours).

What's NOT worth more attempts:
- More regime/selectivity gates on long_squeeze (3 failed = pattern, not coincidence)
- More direction/gate variants on oi_divergence WITHOUT the OI backfill first
- More tightening attempts on CB while soft+bypass coexist

---

## Standing Orders honored

- ✓ No archetypes disabled
- ✓ `bypass_threshold` untouched
- ✓ Real backtest validation (no phantom outcomes)
- ✓ Train/test WFO applied to all 6 variants
- ✓ Lesson #54: zero fusion-based filtering proposed
- ✓ All YAML changes reverted from working tree (config files unchanged)

## Files

- This report: `docs/knowledge/three_fix_followups_2026_05_17.md`
- Raw outputs: `results/followups/{baseline_postTP,fix1_oi_full,fix2b_wyck,fix2c_funding,fix3a_vol,fix3b_atr,fix3c_ema}/{train,oos}/`
- No production code/config modified
