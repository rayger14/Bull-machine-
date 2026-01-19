# SMOKE TEST REPORT
================================================================================

**Test Period**: 2023-01-01 to 2023-04-01 (2,157 bars)
**Total Execution Time**: 49.3s
**Timestamp**: 2025-12-19 15:25:29

## ARCHETYPE SUMMARY
--------------------------------------------------------------------------------

| Arch | Name | Signals | Unique% | Conf Min | Conf Max | Conf Mean | Dom Boost Avg | Dom Boost % | Direction |
|------|------|---------|---------|----------|----------|-----------|---------------|-------------|-----------|
| A    | Spring               |     102 |  100.0% |     0.35 |     5.00 |      1.07 |          2.64x |      100.0% | 100% LONG / 0% SHORT |
| B    | Order Block Retest   |      58 |  100.0% |     0.38 |     3.67 |      0.89 |          2.12x |      100.0% | 100% LONG / 0% SHORT |
| C    | Wick Trap            |     874 |  100.0% |     0.40 |     5.00 |      0.68 |          1.71x |      100.0% | 100% LONG / 0% SHORT |
| D    | Failed Continuation  |      13 |  100.0% |     0.42 |     1.82 |      0.91 |          2.18x |      100.0% | 100% LONG / 0% SHORT |
| E    | Volume Exhaustion    |     124 |  100.0% |     0.46 |     5.00 |      0.97 |          2.16x |      100.0% | EITHER (bidirectional) |
| F    | Exhaustion Reversal  |      75 |  100.0% |     0.41 |     3.28 |      0.90 |          2.14x |      100.0% | 100% LONG / 0% SHORT |
| G    | Liquidity Sweep      |      92 |  100.0% |     0.48 |     3.87 |      0.96 |          1.84x |      100.0% | 100% LONG / 0% SHORT |
| H    | Momentum Continuation |     565 |  100.0% |     0.35 |     5.00 |      0.87 |          2.13x |      100.0% | 100% LONG / 0% SHORT |
| K    | Trap Within Trend    |      15 |  100.0% |     0.42 |     5.00 |      1.55 |          2.31x |      100.0% | 100% LONG / 0% SHORT |
| L    | Retest Cluster       |     399 |  100.0% |     0.47 |     5.00 |      0.83 |          1.74x |      100.0% | 100% LONG / 0% SHORT |
| M    | Confluence Breakout  |      27 |  100.0% |     0.60 |     1.08 |      0.81 |          1.53x |      100.0% | 100% LONG / 0% SHORT |
| S1   | Liquidity Vacuum     |     202 |  100.0% |     0.30 |     0.62 |      0.39 |          1.16x |       56.4% | 100% LONG / 0% SHORT |
| S4   | Funding Divergence   |      14 |  100.0% |     0.40 |     1.34 |      0.61 |          1.64x |      100.0% | 0% LONG / 100% SHORT |
| S5   | Long Squeeze         |      34 |  100.0% |     0.89 |     5.00 |      2.18 |          3.45x |       91.2% | 100% LONG / 0% SHORT |
| S3   | Whipsaw              |       1 |  100.0% |     1.36 |     1.36 |      1.36 |          3.01x |      100.0% | EITHER (bidirectional) |
| S8   | Volume Fade Chop     |     317 |  100.0% |     0.43 |     2.15 |      0.63 |          1.51x |      100.0% | EITHER (bidirectional) |

## DIVERSITY ANALYSIS
--------------------------------------------------------------------------------

**Total unique timestamps with signals**: 1,507
**Average signal overlap**: 56.5% 
❌ HIGH - significant overlap, check for redundancy

**Archetype pairs with high overlap (>50%)**:
  - S5 & H: 100.0% overlap (34 signals)
  - C & G: 100.0% overlap (92 signals)
  - C & M: 100.0% overlap (27 signals)
  - E & S3: 100.0% overlap (1 signals)
  - C & L: 97.7% overlap (390 signals)
  - S4 & H: 85.7% overlap (12 signals)
  - B & C: 84.5% overlap (49 signals)
  - C & F: 77.3% overlap (58 signals)
  - S5 & C: 76.5% overlap (26 signals)
  - S1 & H: 71.8% overlap (145 signals)

## REALISM CHECKS
--------------------------------------------------------------------------------

⚠️ 1 issues detected:

  ⚠️ S3: Very low signal count (1) - check thresholds

## PERFORMANCE
--------------------------------------------------------------------------------

**Total execution time**: 49.3s
**Average per archetype**: 3.08s

**Slowest archetypes**:
  - H (Momentum Continuation): 8.38s
  - S8 (Volume Fade Chop): 6.32s
  - C (Wick Trap): 6.26s
  - S4 (Funding Divergence): 4.63s
  - S1 (Liquidity Vacuum): 3.28s

## RECOMMENDATIONS
--------------------------------------------------------------------------------


⚠️ **WARNING**: 1 archetype(s) produced <5 signals:
  - S3 (Whipsaw): 1 signals - may need threshold tuning

## SUCCESS CRITERIA
--------------------------------------------------------------------------------

✅ PASS: All archetypes produce >0 signals
❌ FAIL: Average overlap <20% (diverse)
✅ PASS: All confidence scores in [0.0-5.0]
✅ PASS: Domain boosts present in >50% of signals

**Overall**: 3/4 criteria passed

⚠️ **1 criteria failed - review needed**