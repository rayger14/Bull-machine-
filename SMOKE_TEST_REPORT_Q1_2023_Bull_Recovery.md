# SMOKE TEST REPORT
================================================================================

**Test Period**: 2023-01-01 to 2023-04-01 (2,157 bars)
**Total Execution Time**: 46.2s
**Timestamp**: 2025-12-16 14:19:43

## ARCHETYPE SUMMARY
--------------------------------------------------------------------------------

| Arch | Name | Signals | Unique% | Conf Min | Conf Max | Conf Mean | Dom Boost Avg | Dom Boost % | Direction |
|------|------|---------|---------|----------|----------|-----------|---------------|-------------|-----------|
| A    | Spring               |     102 |  100.0% |     0.35 |     5.00 |      1.07 |          2.64x |      100.0% | No direction info    |
| B    | Order Block Retest   |      58 |  100.0% |     0.38 |     3.67 |      0.89 |          2.12x |      100.0% | No direction info    |
| C    | Wick Trap            |     874 |  100.0% |     0.40 |     5.00 |      0.68 |          1.00x |        0.0% | No direction info    |
| D    | Failed Continuation  |      13 |  100.0% |     0.42 |     1.82 |      0.91 |          1.00x |        0.0% | No direction info    |
| E    | Volume Exhaustion    |     124 |  100.0% |     0.46 |     5.00 |      0.97 |          1.00x |        0.0% | No direction info    |
| F    | Exhaustion Reversal  |      75 |  100.0% |     0.41 |     3.28 |      0.90 |          1.00x |        0.0% | No direction info    |
| G    | Liquidity Sweep      |      97 |  100.0% |     0.44 |     3.87 |      1.07 |          1.00x |        0.0% | No direction info    |
| H    | Momentum Continuation |     565 |  100.0% |     0.35 |     5.00 |      0.87 |          2.13x |      100.0% | No direction info    |
| K    | Trap Within Trend    |      15 |  100.0% |     0.42 |     5.00 |      1.55 |          1.00x |        0.0% | No direction info    |
| L    | Retest Cluster       |     399 |  100.0% |     0.47 |     5.00 |      0.83 |          1.00x |        0.0% | No direction info    |
| M    | Confluence Breakout  |      27 |  100.0% |     0.60 |     1.08 |      0.81 |          1.00x |        0.0% | No direction info    |
| S1   | Liquidity Vacuum     |     202 |  100.0% |     0.30 |     0.62 |      0.39 |          1.16x |       56.4% | No direction info    |
| S4   | Funding Divergence   |      14 |  100.0% |     0.40 |     1.34 |      0.61 |          1.64x |      100.0% | No direction info    |
| S5   | Long Squeeze         |      34 |  100.0% |     0.89 |     6.50 |      2.22 |          3.45x |       91.2% | No direction info    |
| S3   | Whipsaw              |       1 |  100.0% |     1.36 |     1.36 |      1.36 |          1.00x |        0.0% | No direction info    |
| S8   | Volume Fade Chop     |     317 |  100.0% |     0.43 |     2.15 |      0.63 |          1.00x |        0.0% | No direction info    |

## DIVERSITY ANALYSIS
--------------------------------------------------------------------------------

**Total unique timestamps with signals**: 1,507
**Average signal overlap**: 56.7% 
❌ HIGH - significant overlap, check for redundancy

**Archetype pairs with high overlap (>50%)**:
  - S5 & H: 100.0% overlap (34 signals)
  - C & G: 100.0% overlap (97 signals)
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

⚠️ 12 issues detected:

  ❌ S5: Confidence scores out of valid range [0.0-5.0]: [0.89, 6.50]
  ⚠️ C: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ D: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ E: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ F: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ G: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ K: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ L: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ M: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ S3: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ S3: Very low signal count (1) - check thresholds
  ⚠️ S8: Low domain boost detection (0.0%) - boosts may not be working

## PERFORMANCE
--------------------------------------------------------------------------------

**Total execution time**: 46.2s
**Average per archetype**: 2.89s

**Slowest archetypes**:
  - S1 (Liquidity Vacuum): 15.52s
  - A (Spring): 7.04s
  - S4 (Funding Divergence): 7.01s
  - S5 (Long Squeeze): 5.60s
  - B (Order Block Retest): 5.12s

## RECOMMENDATIONS
--------------------------------------------------------------------------------


⚠️ **WARNING**: 1 archetype(s) produced <5 signals:
  - S3 (Whipsaw): 1 signals - may need threshold tuning

## SUCCESS CRITERIA
--------------------------------------------------------------------------------

✅ PASS: All archetypes produce >0 signals
❌ FAIL: Average overlap <20% (diverse)
❌ FAIL: All confidence scores in [0.0-5.0]
❌ FAIL: Domain boosts present in >50% of signals

**Overall**: 1/4 criteria passed

⚠️ **3 criteria failed - review needed**