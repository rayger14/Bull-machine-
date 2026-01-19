# SMOKE TEST REPORT
================================================================================

**Test Period**: 2022-06-01 to 2022-12-31 (5,112 bars)
**Total Execution Time**: 46.1s
**Timestamp**: 2025-12-17 14:41:41

## ARCHETYPE SUMMARY
--------------------------------------------------------------------------------

| Arch | Name | Signals | Unique% | Conf Min | Conf Max | Conf Mean | Dom Boost Avg | Dom Boost % | Direction |
|------|------|---------|---------|----------|----------|-----------|---------------|-------------|-----------|
| A    | Spring               |     224 |  100.0% |     0.36 |     5.00 |      1.00 |          2.46x |      100.0% | No direction info    |
| B    | Order Block Retest   |      48 |  100.0% |     0.38 |     1.87 |      0.67 |          2.25x |      100.0% | No direction info    |
| C    | Wick Trap            |    1628 |  100.0% |     0.40 |     5.00 |      0.68 |          1.70x |      100.0% | No direction info    |
| D    | Failed Continuation  |      56 |  100.0% |     0.42 |     1.82 |      0.85 |          2.03x |      100.0% | No direction info    |
| E    | Volume Exhaustion    |     326 |  100.0% |     0.46 |     3.29 |      0.94 |          2.10x |      100.0% | No direction info    |
| F    | Exhaustion Reversal  |     236 |  100.0% |     0.41 |     2.95 |      0.88 |          2.09x |      100.0% | No direction info    |
| G    | Liquidity Sweep      |     186 |  100.0% |     0.48 |     5.00 |      1.18 |          1.99x |      100.0% | No direction info    |
| H    | Momentum Continuation |     879 |  100.0% |     0.35 |     5.00 |      0.86 |          2.08x |      100.0% | No direction info    |
| K    | Trap Within Trend    |      27 |  100.0% |     0.40 |     3.32 |      1.03 |          1.48x |      100.0% | No direction info    |
| L    | Retest Cluster       |     672 |  100.0% |     0.40 |     3.75 |      0.83 |          1.72x |      100.0% | No direction info    |
| M    | Confluence Breakout  |      24 |  100.0% |     0.49 |     1.69 |      0.74 |          1.51x |      100.0% | No direction info    |
| S1   | Liquidity Vacuum     |     286 |  100.0% |     0.30 |     0.76 |      0.43 |          1.17x |       54.5% | No direction info    |
| S4   | Funding Divergence   |      27 |  100.0% |     0.43 |     1.59 |      0.88 |          1.98x |      100.0% | No direction info    |
| S5   | Long Squeeze         |      35 |  100.0% |     0.72 |     4.68 |      1.57 |          3.33x |      100.0% | No direction info    |
| S3   | Whipsaw              |       2 |  100.0% |     0.79 |     1.36 |      1.07 |          2.39x |      100.0% | No direction info    |
| S8   | Volume Fade Chop     |     629 |  100.0% |     0.43 |     2.15 |      0.65 |          1.54x |      100.0% | No direction info    |

## DIVERSITY ANALYSIS
--------------------------------------------------------------------------------

**Total unique timestamps with signals**: 3,035
**Average signal overlap**: 49.0% 
❌ HIGH - significant overlap, check for redundancy

**Archetype pairs with high overlap (>50%)**:
  - C & G: 100.0% overlap (186 signals)
  - S5 & H: 97.1% overlap (34 signals)
  - C & M: 95.8% overlap (23 signals)
  - C & L: 93.9% overlap (631 signals)
  - B & C: 81.2% overlap (39 signals)
  - S4 & H: 74.1% overlap (20 signals)
  - S1 & H: 71.7% overlap (205 signals)
  - C & F: 61.9% overlap (146 signals)
  - S5 & C: 60.0% overlap (21 signals)
  - A & C: 59.8% overlap (134 signals)

## REALISM CHECKS
--------------------------------------------------------------------------------

⚠️ 1 issues detected:

  ⚠️ S3: Very low signal count (2) - check thresholds

## PERFORMANCE
--------------------------------------------------------------------------------

**Total execution time**: 46.1s
**Average per archetype**: 2.88s

**Slowest archetypes**:
  - S4 (Funding Divergence): 14.98s
  - S1 (Liquidity Vacuum): 11.42s
  - S5 (Long Squeeze): 5.01s
  - A (Spring): 2.41s
  - H (Momentum Continuation): 2.36s

## RECOMMENDATIONS
--------------------------------------------------------------------------------


⚠️ **WARNING**: 1 archetype(s) produced <5 signals:
  - S3 (Whipsaw): 2 signals - may need threshold tuning

## SUCCESS CRITERIA
--------------------------------------------------------------------------------

✅ PASS: All archetypes produce >0 signals
❌ FAIL: Average overlap <20% (diverse)
✅ PASS: All confidence scores in [0.0-5.0]
✅ PASS: Domain boosts present in >50% of signals

**Overall**: 3/4 criteria passed

⚠️ **1 criteria failed - review needed**