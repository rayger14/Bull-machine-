# SMOKE TEST REPORT
================================================================================

**Test Period**: 2023-08-01 to 2023-12-31 (3,648 bars)
**Total Execution Time**: 43.9s
**Timestamp**: 2025-12-16 13:47:58

## ARCHETYPE SUMMARY
--------------------------------------------------------------------------------

| Arch | Name | Signals | Unique% | Conf Min | Conf Max | Conf Mean | Dom Boost Avg | Dom Boost % | Direction |
|------|------|---------|---------|----------|----------|-----------|---------------|-------------|-----------|
| A    | Spring               |     194 |  100.0% |     0.38 |     3.74 |      0.93 |          2.29x |      100.0% | No direction info    |
| B    | Order Block Retest   |      44 |  100.0% |     0.40 |     2.90 |      0.81 |          1.88x |      100.0% | No direction info    |
| C    | Wick Trap            |    1461 |  100.0% |     0.41 |     3.58 |      0.65 |          1.00x |        0.0% | No direction info    |
| D    | Failed Continuation  |      16 |  100.0% |     0.41 |     2.35 |      1.00 |          1.00x |        0.0% | No direction info    |
| E    | Volume Exhaustion    |     199 |  100.0% |     0.46 |     2.31 |      0.90 |          1.00x |        0.0% | No direction info    |
| F    | Exhaustion Reversal  |     187 |  100.0% |     0.41 |     3.23 |      0.83 |          1.00x |        0.0% | No direction info    |
| G    | Liquidity Sweep      |     198 |  100.0% |     0.48 |     4.16 |      1.03 |          1.00x |        0.0% | No direction info    |
| H    | Momentum Continuation |     567 |  100.0% |     0.35 |     5.00 |      0.98 |          2.37x |      100.0% | No direction info    |
| K    | Trap Within Trend    |      15 |  100.0% |     0.42 |     3.40 |      1.43 |          1.00x |        0.0% | No direction info    |
| L    | Retest Cluster       |     451 |  100.0% |     0.40 |     3.13 |      0.80 |          1.00x |        0.0% | No direction info    |
| M    | Confluence Breakout  |      52 |  100.0% |     0.47 |     3.04 |      0.90 |          1.00x |        0.0% | No direction info    |
| S1   | Liquidity Vacuum     |     219 |  100.0% |     0.30 |     0.65 |      0.43 |          1.15x |       49.8% | No direction info    |
| S4   | Funding Divergence   |      22 |  100.0% |     0.40 |     3.25 |      0.94 |          2.51x |      100.0% | No direction info    |
| S5   | Long Squeeze         |       0 |  100.0% |     0.00 |     0.00 |      0.00 |          0.00x |        0.0% | N/A                  |
| S3   | Whipsaw              |      16 |  100.0% |     0.47 |     1.15 |      0.63 |          1.00x |        0.0% | No direction info    |
| S8   | Volume Fade Chop     |     925 |  100.0% |     0.43 |     2.15 |      0.62 |          1.00x |        0.0% | No direction info    |

## DIVERSITY ANALYSIS
--------------------------------------------------------------------------------

**Total unique timestamps with signals**: 2,536
**Average signal overlap**: 49.4% 
❌ HIGH - significant overlap, check for redundancy

**Archetype pairs with high overlap (>50%)**:
  - C & G: 100.0% overlap (198 signals)
  - C & L: 97.3% overlap (439 signals)
  - B & C: 84.1% overlap (37 signals)
  - C & M: 82.7% overlap (43 signals)
  - S4 & H: 81.8% overlap (18 signals)
  - C & F: 77.0% overlap (144 signals)
  - S4 & C: 72.7% overlap (16 signals)
  - A & C: 69.6% overlap (135 signals)
  - C & K: 66.7% overlap (10 signals)
  - G & K: 66.7% overlap (10 signals)

## REALISM CHECKS
--------------------------------------------------------------------------------

⚠️ 12 issues detected:

  ⚠️ S5: Low domain boost detection (0.0%) - boosts may not be working
  ❌ S5: ZERO signals detected - archetype may be broken or thresholds too strict
  ⚠️ C: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ D: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ E: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ F: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ G: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ K: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ L: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ M: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ S3: Low domain boost detection (0.0%) - boosts may not be working
  ⚠️ S8: Low domain boost detection (0.0%) - boosts may not be working

## PERFORMANCE
--------------------------------------------------------------------------------

**Total execution time**: 43.9s
**Average per archetype**: 2.75s

**Slowest archetypes**:
  - A (Spring): 8.38s
  - L (Retest Cluster): 5.27s
  - H (Momentum Continuation): 5.09s
  - C (Wick Trap): 4.43s
  - B (Order Block Retest): 3.81s

## RECOMMENDATIONS
--------------------------------------------------------------------------------

❌ **CRITICAL**: 1 archetype(s) produced ZERO signals:
  - S5 (Long Squeeze): Check method implementation or relax thresholds

## SUCCESS CRITERIA
--------------------------------------------------------------------------------

❌ FAIL: All archetypes produce >0 signals
❌ FAIL: Average overlap <20% (diverse)
✅ PASS: All confidence scores in [0.0-5.0]
❌ FAIL: Domain boosts present in >50% of signals

**Overall**: 1/4 criteria passed

⚠️ **3 criteria failed - review needed**