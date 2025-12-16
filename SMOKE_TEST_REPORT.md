# SMOKE TEST REPORT
================================================================================

**Test Period**: 2023-01-01 to 2023-04-01 (2,157 bars)
**Total Execution Time**: 8.9s
**Timestamp**: 2025-12-15 17:27:24

## ARCHETYPE SUMMARY
--------------------------------------------------------------------------------

| Arch | Name | Signals | Unique% | Conf Min | Conf Max | Conf Mean | Dom Boost Avg | Dom Boost % | Direction |
|------|------|---------|---------|----------|----------|-----------|---------------|-------------|-----------|
| A    | Spring               |       0 |  100.0% |     0.00 |     0.00 |      0.00 |          0.00x |        0.0% | N/A                  |
| B    | Order Block Retest   |      46 |  100.0% |     0.38 |     3.67 |      0.95 |          2.09x |      100.0% | No direction info    |
| C    | Wick Trap            |       0 |  100.0% |     0.00 |     0.00 |      0.00 |          0.00x |        0.0% | N/A                  |
| D    | Failed Continuation  |      13 |  100.0% |     0.42 |     1.82 |      0.91 |          1.00x |        0.0% | No direction info    |
| E    | Volume Exhaustion    |     124 |  100.0% |     0.46 |     5.00 |      0.97 |          1.00x |        0.0% | No direction info    |
| F    | Exhaustion Reversal  |      75 |  100.0% |     0.41 |     3.28 |      0.90 |          1.00x |        0.0% | No direction info    |
| G    | Liquidity Sweep      |      97 |  100.0% |     0.44 |     3.87 |      1.07 |          1.00x |        0.0% | No direction info    |
| H    | Momentum Continuation |     565 |  100.0% |     0.35 |     5.52 |      0.87 |          2.13x |      100.0% | No direction info    |
| K    | Trap Within Trend    |      15 |  100.0% |     0.42 |     5.00 |      1.55 |          1.00x |        0.0% | No direction info    |
| L    | Retest Cluster       |       0 |  100.0% |     0.00 |     0.00 |      0.00 |          0.00x |        0.0% | N/A                  |
| M    | Confluence Breakout  |       0 |  100.0% |     0.00 |     0.00 |      0.00 |          0.00x |        0.0% | N/A                  |
| S1   | Liquidity Vacuum     |       0 |  100.0% |     0.00 |     0.00 |      0.00 |          0.00x |        0.0% | N/A                  |
| S4   | Funding Divergence   |      14 |  100.0% |     0.40 |     1.34 |      0.61 |          1.64x |      100.0% | No direction info    |
| S5   | Long Squeeze         |       0 |  100.0% |     0.00 |     0.00 |      0.00 |          0.00x |        0.0% | N/A                  |
| S3   | Whipsaw              |       1 |  100.0% |     1.36 |     1.36 |      1.36 |          1.00x |        0.0% | No direction info    |
| S8   | Volume Fade Chop     |       0 |  100.0% |     0.00 |     0.00 |      0.00 |          0.00x |        0.0% | N/A                  |

## DIVERSITY ANALYSIS
--------------------------------------------------------------------------------

**Total unique timestamps with signals**: 833
**Average signal overlap**: 12.8% 
✅ GOOD - archetypes are diverse

**Archetype pairs with high overlap (>50%)**:
  - E & S3: 100.0% overlap (1 signals)
  - S4 & H: 85.7% overlap (12 signals)

## REALISM CHECKS
--------------------------------------------------------------------------------

⚠️ 22 issues detected:

  ⚠️ S1: Low domain boost detection (0.0%) - boosts may not be working
  ❌ S1: ZERO signals detected - archetype may be broken or thresholds too strict
  ⚠️ S5: Low domain boost detection (0.0%) - boosts may not be working
  ❌ S5: ZERO signals detected - archetype may be broken or thresholds too strict
  ⚠️ A: Low domain boost detection (0.0%) - boosts may not be working
  ❌ A: ZERO signals detected - archetype may be broken or thresholds too strict
  ⚠️ C: Low domain boost detection (0.0%) - boosts may not be working
  ❌ C: ZERO signals detected - archetype may be broken or thresholds too strict
  ⚠️ D: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ E: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ F: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ G: Low domain boost detection (0.0%) - boosts may not be working
  ❌ H: Confidence scores out of valid range [0.0-5.0]: [0.35, 5.52]
  ⚠️ K: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ L: Low domain boost detection (0.0%) - boosts may not be working
  ❌ L: ZERO signals detected - archetype may be broken or thresholds too strict
  ⚠️ M: Low domain boost detection (0.0%) - boosts may not be working
  ❌ M: ZERO signals detected - archetype may be broken or thresholds too strict
  ⚠️ S3: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ S3: Very low signal count (1) - check thresholds
  ⚠️ S8: Low domain boost detection (0.0%) - boosts may not be working
  ❌ S8: ZERO signals detected - archetype may be broken or thresholds too strict

## PERFORMANCE
--------------------------------------------------------------------------------

**Total execution time**: 8.9s
**Average per archetype**: 0.56s

**Slowest archetypes**:
  - S1 (Liquidity Vacuum): 3.35s
  - H (Momentum Continuation): 0.74s
  - S5 (Long Squeeze): 0.65s
  - S4 (Funding Divergence): 0.60s
  - B (Order Block Retest): 0.58s

## RECOMMENDATIONS
--------------------------------------------------------------------------------

❌ **CRITICAL**: 7 archetype(s) produced ZERO signals:
  - S1 (Liquidity Vacuum): Check method implementation or relax thresholds
  - S5 (Long Squeeze): Check method implementation or relax thresholds
  - A (Spring): Check method implementation or relax thresholds
  - C (Wick Trap): Check method implementation or relax thresholds
  - L (Retest Cluster): Check method implementation or relax thresholds
  - M (Confluence Breakout): Check method implementation or relax thresholds
  - S8 (Volume Fade Chop): Check method implementation or relax thresholds

⚠️ **WARNING**: 1 archetype(s) produced <5 signals:
  - S3 (Whipsaw): 1 signals - may need threshold tuning

## SUCCESS CRITERIA
--------------------------------------------------------------------------------

❌ FAIL: All archetypes produce >0 signals
✅ PASS: Average overlap <20% (diverse)
❌ FAIL: All confidence scores in [0.0-5.0]
❌ FAIL: Domain boosts present in >50% of signals

**Overall**: 1/4 criteria passed

⚠️ **3 criteria failed - review needed**