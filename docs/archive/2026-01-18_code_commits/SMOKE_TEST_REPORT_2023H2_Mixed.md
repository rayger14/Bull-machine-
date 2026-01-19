# SMOKE TEST REPORT
================================================================================

**Test Period**: 2023-08-01 to 2023-12-31 (3,648 bars)
**Total Execution Time**: 54.2s
**Timestamp**: 2025-12-17 14:42:35

## ARCHETYPE SUMMARY
--------------------------------------------------------------------------------

| Arch | Name | Signals | Unique% | Conf Min | Conf Max | Conf Mean | Dom Boost Avg | Dom Boost % | Direction |
|------|------|---------|---------|----------|----------|-----------|---------------|-------------|-----------|
| A    | Spring               |     194 |  100.0% |     0.38 |     3.74 |      0.93 |          2.29x |      100.0% | No direction info    |
| B    | Order Block Retest   |     125 |  100.0% |     0.37 |     2.81 |      0.68 |          1.98x |      100.0% | No direction info    |
| C    | Wick Trap            |    1461 |  100.0% |     0.41 |     3.58 |      0.65 |          1.64x |      100.0% | No direction info    |
| D    | Failed Continuation  |      16 |  100.0% |     0.41 |     2.35 |      1.00 |          2.38x |      100.0% | No direction info    |
| E    | Volume Exhaustion    |     199 |  100.0% |     0.46 |     2.31 |      0.90 |          1.99x |      100.0% | No direction info    |
| F    | Exhaustion Reversal  |     187 |  100.0% |     0.41 |     3.23 |      0.83 |          1.97x |      100.0% | No direction info    |
| G    | Liquidity Sweep      |     198 |  100.0% |     0.48 |     4.16 |      1.03 |          1.72x |      100.0% | No direction info    |
| H    | Momentum Continuation |     567 |  100.0% |     0.35 |     5.00 |      0.98 |          2.37x |      100.0% | No direction info    |
| K    | Trap Within Trend    |      15 |  100.0% |     0.42 |     3.40 |      1.43 |          2.00x |      100.0% | No direction info    |
| L    | Retest Cluster       |     451 |  100.0% |     0.40 |     3.13 |      0.80 |          1.66x |      100.0% | No direction info    |
| M    | Confluence Breakout  |      52 |  100.0% |     0.47 |     3.04 |      0.90 |          1.76x |      100.0% | No direction info    |
| S1   | Liquidity Vacuum     |     219 |  100.0% |     0.30 |     0.65 |      0.43 |          1.15x |       49.8% | No direction info    |
| S4   | Funding Divergence   |      22 |  100.0% |     0.40 |     3.25 |      0.94 |          2.51x |      100.0% | No direction info    |
| S5   | Long Squeeze         |      30 |  100.0% |     0.70 |     5.00 |      2.13 |          3.83x |      100.0% | No direction info    |
| S3   | Whipsaw              |      16 |  100.0% |     0.47 |     1.15 |      0.63 |          1.41x |      100.0% | No direction info    |
| S8   | Volume Fade Chop     |     925 |  100.0% |     0.43 |     2.15 |      0.62 |          1.48x |      100.0% | No direction info    |

## DIVERSITY ANALYSIS
--------------------------------------------------------------------------------

**Total unique timestamps with signals**: 2,541
**Average signal overlap**: 50.0% 
❌ HIGH - significant overlap, check for redundancy

**Archetype pairs with high overlap (>50%)**:
  - C & G: 100.0% overlap (198 signals)
  - C & L: 97.3% overlap (439 signals)
  - B & C: 93.6% overlap (117 signals)
  - C & M: 82.7% overlap (43 signals)
  - S4 & H: 81.8% overlap (18 signals)
  - S5 & H: 80.0% overlap (24 signals)
  - C & F: 77.0% overlap (144 signals)
  - S4 & C: 72.7% overlap (16 signals)
  - A & C: 69.6% overlap (135 signals)
  - S5 & C: 66.7% overlap (20 signals)

## REALISM CHECKS
--------------------------------------------------------------------------------

✅ All confidence scores in valid range
✅ Domain boosts detected in majority of signals
✅ Direction alignment correct for all archetypes
✅ No critical issues detected

## PERFORMANCE
--------------------------------------------------------------------------------

**Total execution time**: 54.2s
**Average per archetype**: 3.39s

**Slowest archetypes**:
  - S1 (Liquidity Vacuum): 27.11s
  - S4 (Funding Divergence): 4.92s
  - A (Spring): 3.98s
  - H (Momentum Continuation): 3.42s
  - S5 (Long Squeeze): 3.35s

## RECOMMENDATIONS
--------------------------------------------------------------------------------

✅ All archetypes producing reasonable signal counts

## SUCCESS CRITERIA
--------------------------------------------------------------------------------

✅ PASS: All archetypes produce >0 signals
❌ FAIL: Average overlap <20% (diverse)
✅ PASS: All confidence scores in [0.0-5.0]
✅ PASS: Domain boosts present in >50% of signals

**Overall**: 3/4 criteria passed

⚠️ **1 criteria failed - review needed**