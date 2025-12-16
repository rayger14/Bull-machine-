# SMOKE TEST REPORT
================================================================================

**Test Period**: 2022-06-01 to 2022-12-31 (5,112 bars)
**Total Execution Time**: 90.6s
**Timestamp**: 2025-12-16 13:47:14

## ARCHETYPE SUMMARY
--------------------------------------------------------------------------------

| Arch | Name | Signals | Unique% | Conf Min | Conf Max | Conf Mean | Dom Boost Avg | Dom Boost % | Direction |
|------|------|---------|---------|----------|----------|-----------|---------------|-------------|-----------|
| A    | Spring               |     224 |  100.0% |     0.36 |     5.00 |      1.00 |          2.46x |      100.0% | No direction info    |
| B    | Order Block Retest   |       0 |  100.0% |     0.00 |     0.00 |      0.00 |          0.00x |        0.0% | N/A                  |
| C    | Wick Trap            |    1628 |  100.0% |     0.40 |     5.00 |      0.68 |          1.00x |        0.0% | No direction info    |
| D    | Failed Continuation  |      56 |  100.0% |     0.42 |     1.82 |      0.85 |          1.00x |        0.0% | No direction info    |
| E    | Volume Exhaustion    |     326 |  100.0% |     0.46 |     3.29 |      0.94 |          1.00x |        0.0% | No direction info    |
| F    | Exhaustion Reversal  |     236 |  100.0% |     0.41 |     2.95 |      0.88 |          1.00x |        0.0% | No direction info    |
| G    | Liquidity Sweep      |     186 |  100.0% |     0.48 |     5.00 |      1.18 |          1.00x |        0.0% | No direction info    |
| H    | Momentum Continuation |     879 |  100.0% |     0.35 |     5.00 |      0.86 |          2.08x |      100.0% | No direction info    |
| K    | Trap Within Trend    |      27 |  100.0% |     0.40 |     3.32 |      1.03 |          1.00x |        0.0% | No direction info    |
| L    | Retest Cluster       |     672 |  100.0% |     0.40 |     3.75 |      0.83 |          1.00x |        0.0% | No direction info    |
| M    | Confluence Breakout  |      24 |  100.0% |     0.49 |     1.69 |      0.74 |          1.00x |        0.0% | No direction info    |
| S1   | Liquidity Vacuum     |     408 |  100.0% |     0.30 |     4.96 |      0.52 |          1.19x |       68.1% | No direction info    |
| S4   | Funding Divergence   |      27 |  100.0% |     0.43 |     1.59 |      0.88 |          1.98x |      100.0% | No direction info    |
| S5   | Long Squeeze         |       1 |  100.0% |     1.20 |     1.20 |      1.20 |          3.20x |      100.0% | No direction info    |
| S3   | Whipsaw              |       2 |  100.0% |     0.79 |     1.36 |      1.07 |          1.00x |        0.0% | No direction info    |
| S8   | Volume Fade Chop     |     629 |  100.0% |     0.43 |     2.15 |      0.65 |          1.00x |        0.0% | No direction info    |

## DIVERSITY ANALYSIS
--------------------------------------------------------------------------------

**Total unique timestamps with signals**: 3,073
**Average signal overlap**: 48.9% 
❌ HIGH - significant overlap, check for redundancy

**Archetype pairs with high overlap (>50%)**:
  - S5 & H: 100.0% overlap (1 signals)
  - C & G: 100.0% overlap (186 signals)
  - C & M: 95.8% overlap (23 signals)
  - C & L: 93.9% overlap (631 signals)
  - S4 & H: 74.1% overlap (20 signals)
  - C & F: 61.9% overlap (146 signals)
  - A & C: 59.8% overlap (134 signals)
  - S1 & H: 59.1% overlap (241 signals)
  - C & K: 55.6% overlap (15 signals)
  - G & K: 55.6% overlap (15 signals)

## REALISM CHECKS
--------------------------------------------------------------------------------

⚠️ 14 issues detected:

  ⚠️ S5: Very low signal count (1) - check thresholds
  ⚠️ B: Low domain boost detection (0.0%) - boosts may not be working
  ❌ B: ZERO signals detected - archetype may be broken or thresholds too strict
  ⚠️ C: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ D: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ E: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ F: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ G: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ K: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ L: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ M: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ S3: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ S3: Very low signal count (2) - check thresholds
  ⚠️ S8: Low domain boost detection (0.0%) - boosts may not be working

## PERFORMANCE
--------------------------------------------------------------------------------

**Total execution time**: 90.6s
**Average per archetype**: 5.66s

**Slowest archetypes**:
  - S1 (Liquidity Vacuum): 22.54s
  - S4 (Funding Divergence): 14.68s
  - S5 (Long Squeeze): 12.30s
  - A (Spring): 11.80s
  - C (Wick Trap): 8.07s

## RECOMMENDATIONS
--------------------------------------------------------------------------------

❌ **CRITICAL**: 1 archetype(s) produced ZERO signals:
  - B (Order Block Retest): Check method implementation or relax thresholds

⚠️ **WARNING**: 2 archetype(s) produced <5 signals:
  - S5 (Long Squeeze): 1 signals - may need threshold tuning
  - S3 (Whipsaw): 2 signals - may need threshold tuning

## SUCCESS CRITERIA
--------------------------------------------------------------------------------

❌ FAIL: All archetypes produce >0 signals
❌ FAIL: Average overlap <20% (diverse)
✅ PASS: All confidence scores in [0.0-5.0]
❌ FAIL: Domain boosts present in >50% of signals

**Overall**: 1/4 criteria passed

⚠️ **3 criteria failed - review needed**