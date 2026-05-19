# CB Re-Audit — May 18, 2026

**Branch**: `chore/quant-lessons-v2-codify`
**Status**: Final
**Verdict**: **CLOSE the CB-rethink todo. The May 4 "Q0 quartile" framing is OBSOLETE — old quartile pattern has reversed.**

---

## TL;DR

May 4 audit framed CB's problem as "Q0 (low-fusion) trades systematically lose, Q3 (high-fusion) trades systematically win — add a fusion-score floor." That recommendation was correctly rejected because it violates Lesson #54 (fusion has negative predictive power).

**Re-running the same quartile analysis today on live data shows the pattern has effectively reversed.** Q0 is now profitable (+$448, PF 1.53). The current weakness is in Q1/Q2 (the MIDDLE of the fusion distribution). Q3 remains the goldmine.

This is exactly what Lesson #54 predicted: **fusion-score correlations with trade outcomes are spurious and don't persist out-of-sample.** The May 4 Q0-vs-Q3 split happened to fit the data window we had then. With more data, it dissolves.

CB is currently profitable overall: 22 live trades, 68% WR, +$1,038 PnL. No design rethink needed. The CB hard-gate immune architecture (`gate_mode: soft + bypass_fusion_threshold: true`) is intentional and validated — it lets the structural identity gate (`atr_percentile < 0.30 AND poc_distance < 0.05`) decide alone, which works in the current regime.

Close the "CB hard-gate design rethink" todo from the master findings.

---

## The fusion-quartile re-audit

Same methodology as May 4: split live CB trades into 4 fusion-score quartiles and compute per-quartile metrics.

### May 4 (10 trades available then)

| Quartile | Avg fusion | n | PnL | WR | PF |
|----------|-----------:|--:|----:|---:|----:|
| Q0 | 0.179 | 3 | **−$1,712** | 33% | **0.13** ← framed as the problem |
| Q1 | 0.194 | 2 | +$60 | 50% | 1.57 |
| Q2 | 0.214 | 2 | +$282 | 100% | ∞ |
| Q3 | 0.331 | 3 | +$2,480 | 100% | ∞ ← framed as the goldmine |

### May 18 (22 trades available now)

| Quartile | Avg fusion | n | PnL | WR | PF |
|----------|-----------:|--:|----:|---:|----:|
| Q0 | 0.125 | 6 | **+$448** | **83%** | **1.53** ← was the "problem", now profitable |
| Q1 | 0.172 | 5 | **−$1,270** | 40% | **0.51** ← new weakness |
| Q2 | 0.202 | 5 | −$400 | 60% | 0.53 ← also weak |
| Q3 | 0.329 | 6 | **+$2,259** | **83%** | **4.01** ← still the goldmine |

### What changed

The quartile boundaries themselves shifted as more trades accumulated (Q0 average dropped from 0.179 → 0.125 because we now have lower-fusion trades). But more importantly:

- **Q0 transformed**: PnL went from −$1,712 (PF 0.13) to +$448 (PF 1.53). The "lowest fusion = lose" pattern reversed.
- **Q1/Q2 emerged as weak**: collectively −$1,670 across 10 trades (PF ~0.5).
- **Q3 confirmed**: still PF 4.01, 83% WR. The high-fusion cluster is genuinely productive.

The "Q0 systematic losers" pattern was a small-sample artifact of which specific 3 trades happened to be in that bucket on May 4. With 22 trades, the quartile boundaries reshape and the pattern dissolves. **Classic Lesson #54 behavior.**

---

## What's NOT a problem with CB anymore

The May 4 framing motivated three rejected fixes:
- Adding a fusion floor (`fusion_threshold: 0.20`) → would have violated Lesson #54
- Flipping `gate_mode: soft → hard` → tested May 17, REJECTED (over-blocked 78%, cost $17,885)
- Adding `ema_slope_50 > 0` hard gate → tested May 17, REJECTED (no measurable improvement)

All three rejected for the right reasons. The actual CB performance has improved on its own through:
- TP Tier 1 deploy (May 16) — better trailing/scale-out for CB's compression-breakout setups
- 3-of-3 distribution_exhaustion boost (May 18) — boosts CB entries when 4H bearish + OI capitulating + price at support align
- PR #30 fixes earlier (FRVP `distance_to_poc` bug, `bypass_fusion_threshold` flag)

**Verdict**: CB doesn't need architectural rethink. Its `soft + bypass_fusion_threshold` arrangement is correct for the archetype (compression patterns score near-zero on SMC/BOMS, so fusion isn't a quality signal — Lesson #54 applies). The identity gate carries the weight.

---

## What MIGHT be a future angle (NOT a recommendation, just exploration)

If you wanted to chase CB's Q1/Q2 weakness, the structurally correct move per Rule 8 is to find the bar-level discriminator between Q3 winners and Q1/Q2 losers. Likely candidates from the existing 3-of-3 boost framework:

- `tf4h_wyckoff_bullish_score` at Q3 entries vs Q1/Q2 (does Wyckoff regime align with the win?)
- `volume_zscore` at Q3 vs Q1/Q2 entries (we know vol_z is a strong winner-discriminator)
- `ema_slope_50` at Q3 vs Q1/Q2

If a clean discriminator exists, propose a **boost** for CB entries matching the Q3 signature, not a filter for Q1/Q2. Estimated payoff is small (Q1/Q2 combined is only −$1,670, recoverable amount ~$500-1000 from boost) — likely not worth a focused study.

---

## Recommendation

**Close the CB design rethink todo.** Mark CB as "no current action — `soft + bypass_fusion_threshold` validated as correct architecture for compression-pattern archetype" in MEMORY.md "Validated Decisions to Hold".

---

## Files

- This report: `docs/knowledge/cb_reaudit_2026_05_18.md`
- Live data source: `/tmp/trade_outcomes_live.csv` (refreshed May 18)
- CB YAML: `configs/archetypes/confluence_breakout.yaml`
- May 4 audit (now superseded): `docs/knowledge/four_fixes_validation_2026_05_16.md` (Fix 3 section)

## Constraints Honored

- ✓ READ-ONLY (no config changes)
- ✓ Rule 7 / 8 / 9 / 10 applied
- ✓ Honest re-evaluation of prior framing (May 4 quartile pattern was Lesson #54 spurious correlation)
