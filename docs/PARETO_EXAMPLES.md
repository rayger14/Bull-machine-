# Pareto Visualization Examples

This document shows example outputs and interpretations from the Pareto frontier visualization.

## Example 1: Balanced Frontier

### Input Study
- 500 trials
- Objectives: PF, Trade Count, Max DD
- Search space: 8 parameters

### Console Output
```
================================================================================
PARETO FRONTIER VISUALIZATION
================================================================================

✓ Output directory: results/phase3_frontier

✓ Loaded study 'bear_phase2_tuning' with 500 trials
✓ Extracted 487 completed trials
✓ Identified 23 Pareto-optimal trials

Generating visualizations...
✓ Saved 3D interactive plot: results/phase3_frontier/pareto_3d.html
✓ Saved 2D projections: results/phase3_frontier/pareto_2d_projections.png
✓ Saved parameter sensitivity: results/phase3_frontier/parameter_sensitivity.png
✓ Saved distribution plots: results/phase3_frontier/pareto_distributions.png
✓ Exported 23 Pareto trials to: results/phase3_frontier/pareto_trials.csv

================================================================================
PARETO-OPTIMAL TRIALS SUMMARY
================================================================================

Top 5 by Profit Factor:
   trial_number  profit_factor  trade_count  max_drawdown
0           342          1.652         38.2       -0.2341
1           289          1.598         35.7       -0.2156
2           411          1.547         29.4       -0.1987
3           156          1.521         31.8       -0.2089
4           478          1.498         27.3       -0.1845

================================================================================

✓ Saved summary report: results/phase3_frontier/pareto_analysis_summary.txt

================================================================================
VISUALIZATION COMPLETE
================================================================================

Outputs saved to: results/phase3_frontier
  - pareto_3d.html              : Interactive 3D plot
  - pareto_2d_projections.png   : 2D projection plots
  - parameter_sensitivity.png   : Parameter correlation heatmap
  - pareto_distributions.png    : Objective distributions
  - pareto_trials.csv           : Pareto-optimal trial data
  - pareto_analysis_summary.txt : Text summary report
================================================================================
```

### Summary Report Excerpt
```
================================================================================
PARETO FRONTIER ANALYSIS SUMMARY
================================================================================

Total Trials: 487
Pareto-Optimal Trials: 23 (4.7%)

--------------------------------------------------------------------------------
OBJECTIVE STATISTICS
--------------------------------------------------------------------------------

Profit Factor:
  All Trials:     Mean=1.187, Std=0.234, Min=0.543, Max=1.652
  Pareto Optimal: Mean=1.421, Std=0.098, Min=1.298, Max=1.652

Annual Trade Count:
  All Trials:     Mean=31.4, Std=12.8, Min=8.2, Max=67.3
  Pareto Optimal: Mean=32.8, Std=5.6, Min=24.1, Max=42.9

Max Drawdown:
  All Trials:     Mean=-0.267, Std=0.089, Min=-0.512, Max=-0.134
  Pareto Optimal: Mean=-0.204, Std=0.034, Min=-0.254, Max=-0.158

--------------------------------------------------------------------------------
TOP 10 PARETO-OPTIMAL TRIALS (by Profit Factor)
--------------------------------------------------------------------------------

Trial    PF       Trades     DD
----------------------------------------
342      1.652    38.2       -23.41%
289      1.598    35.7       -21.56%
411      1.547    29.4       -19.87%
156      1.521    31.8       -20.89%
478      1.498    27.3       -18.45%
234      1.467    33.1       -22.34%
389      1.445    28.9       -17.92%
201      1.432    36.4       -23.67%
445      1.418    30.2       -19.23%
167      1.401    34.8       -21.45%

================================================================================
TARGET ZONE ANALYSIS
================================================================================

Trials meeting targets (PF > 1.3, Trades 25-40): 18

Trials in target zone:
Trial    PF       Trades     DD
----------------------------------------
342      1.652    38.2       -23.41%
289      1.598    35.7       -21.56%
411      1.547    29.4       -19.87%
156      1.521    31.8       -20.89%
478      1.498    27.3       -18.45%
234      1.467    33.1       -22.34%
389      1.445    28.9       -17.92%
201      1.432    36.4       -23.67%
445      1.418    30.2       -19.23%
167      1.401    34.8       -21.45%
...

================================================================================
```

### CSV Export Sample
```csv
trial_number,profit_factor,trade_count,max_drawdown,confidence_threshold,risk_per_trade,min_rrr,atr_multiplier,...
342,1.652,38.2,-0.2341,0.782,0.0234,2.45,1.87,...
289,1.598,35.7,-0.2156,0.756,0.0198,2.67,2.12,...
411,1.547,29.4,-0.1987,0.823,0.0187,2.34,1.65,...
156,1.521,31.8,-0.2089,0.798,0.0212,2.56,1.93,...
478,1.498,27.3,-0.1845,0.845,0.0165,2.12,1.54,...
```

## Example 2: Parameter Sensitivity Insights

### Correlation Matrix
```
Parameter            | PF     | Trade Count | Max DD
--------------------------------------------------
confidence_threshold | +0.684 | -0.512     | +0.234
risk_per_trade      | -0.123 | +0.089     | -0.567
min_rrr             | +0.456 | -0.234     | +0.312
atr_multiplier      | +0.234 | +0.156     | -0.123
stop_loss_buffer    | -0.098 | -0.045     | +0.678
take_profit_ratio   | +0.345 | -0.089     | +0.189
volume_threshold    | +0.123 | +0.678     | -0.234
confluence_weight   | +0.567 | -0.345     | +0.189
```

### Interpretation
**High Impact Parameters** (|corr| > 0.5):
1. `confidence_threshold`: Strong positive correlation with PF (+0.684)
   - Higher confidence → Higher profit factor
   - But reduces trade count (-0.512)

2. `stop_loss_buffer`: Strong positive correlation with DD (+0.678)
   - Wider stops → Larger drawdowns
   - Critical for risk management

3. `volume_threshold`: Strong positive correlation with trade count (+0.678)
   - Lower threshold → More trades
   - Minimal impact on PF and DD

**Phase 3 Action Items**:
- Fine-tune `confidence_threshold` in range [0.75, 0.85]
- Tighten `stop_loss_buffer` to reduce DD
- Leave `volume_threshold` at discovered optimum

## Example 3: Trade-off Analysis

### Scenario: Aggressive vs Conservative

#### Aggressive Strategy (Trial 342)
```
Profit Factor:    1.652  ⭐⭐⭐⭐⭐
Trade Count:      38.2   ⭐⭐⭐⭐
Max Drawdown:     -23.4% ⭐⭐⭐

Profile: High returns with moderate risk
Use case: Bull markets, risk-tolerant scenarios
```

#### Conservative Strategy (Trial 389)
```
Profit Factor:    1.445  ⭐⭐⭐⭐
Trade Count:      28.9   ⭐⭐⭐
Max Drawdown:     -17.9% ⭐⭐⭐⭐⭐

Profile: Stable returns with low risk
Use case: Bear markets, risk-averse scenarios
```

#### Balanced Strategy (Trial 156)
```
Profit Factor:    1.521  ⭐⭐⭐⭐⭐
Trade Count:      31.8   ⭐⭐⭐⭐
Max Drawdown:     -20.9% ⭐⭐⭐⭐

Profile: Optimal balance across all objectives
Use case: General purpose, all market conditions
```

### Ensemble Recommendation
Combine all three for robustness:
- 40% Balanced (Trial 156)
- 30% Aggressive (Trial 342)
- 30% Conservative (Trial 389)

Expected ensemble performance:
- Profit Factor: ~1.53 (weighted average)
- Trade Count: ~33.5 (diversified activity)
- Max Drawdown: ~-20.7% (risk-managed)

## Example 4: Visual Interpretation

### 3D Plot Reading

```
        Max Drawdown (↓ better)
              ↑
              |
        -15%  |    ⬥ B (Conservative)
              |
        -20%  |  ⬥ A (Balanced)
              |
        -25%  | ⬥ C (Aggressive)
              |
              •─────────────────→ Trade Count
             /
            /
     Profit Factor (↑ better)
```

**Point A (Balanced)**:
- Best overall position
- Moderate on all dimensions
- Recommended for base strategy

**Point B (Conservative)**:
- Lowest drawdown
- Lower profit factor
- Good for risk mitigation

**Point C (Aggressive)**:
- Highest profit factor
- Higher drawdown
- Good for growth scenarios

### 2D Projection Insights

#### PF vs Trade Count
```
PF
↑
1.6 |         ⬥ (342)
    |       ⬥ (289)
1.5 |     ⬥   ⬥
    |   ⬥   ⬥
1.4 | ⬥   ⬥     ← Target zone (green)
    |   ⬥
1.3 +─────────────────→ TC
    0   20  40  60
        └─────┘
        Target: 25-40
```

**Observation**: Most Pareto trials cluster in target zone
**Action**: Focus Phase 3 on this region

#### PF vs Max DD
```
PF
↑
1.6 |         ⬥
    |       ⬥ ⬥
1.5 |     ⬥   ⬥
    |   ⬥   ⬥
1.4 | ⬥   ⬥
    |   ⬥
1.3 +─────────────────→ DD
   -30% -20% -10%  0%
    └─────┘
    Target: -15% to -30%
```

**Observation**: Trade-off is linear (higher PF → higher DD)
**Action**: Set maximum acceptable DD, then maximize PF

## Example 5: Failure Cases

### Case 1: No Pareto Trials
```
✗ No valid trials with all objectives
```
**Cause**: Optimization failed or incomplete
**Solution**: Check trial states, ensure objectives are logged

### Case 2: All Trials Pareto-Optimal
```
✓ Identified 487 Pareto-optimal trials (100%)
```
**Cause**: Objectives not competitive or search space too large
**Solution**: Narrow search space or adjust objective weights

### Case 3: Clustering
```
✓ Identified 3 Pareto-optimal trials (0.6%)
```
**Cause**: Very similar trials, low diversity
**Solution**: Increase search space or run more trials

## Interpretation Tips

### Reading Correlations

**Strong Positive (+0.7 to +1.0)**:
- Parameter increases → Objective increases
- Direct relationship
- Example: Higher confidence → Higher PF

**Strong Negative (-0.7 to -1.0)**:
- Parameter increases → Objective decreases
- Inverse relationship
- Example: Tighter stops → Lower DD

**Weak (< ±0.3)**:
- Little or no relationship
- Parameter may be redundant
- Candidate for removal

### Target Zone Analysis

**Many trials in zone**: Optimization successful
**Few trials in zone**: May need more exploration
**No trials in zone**: Targets may be unrealistic

### Distribution Comparison

**Pareto mean >> All mean**: Strong optimization
**Pareto std < All std**: Consistent performance
**Distributions overlap**: Weak frontier

## Next Steps

1. **Review visualizations**: Understand trade-offs
2. **Examine top trials**: Identify patterns
3. **Check parameters**: Find important factors
4. **Select ensemble**: Choose diverse strategies
5. **Plan Phase 3**: Focus on high-impact parameters

## Common Questions

### Q: How many Pareto trials is good?
**A**: 5-20 is typical for 500 trials (1-4%)

### Q: Should I pick the trial with highest PF?
**A**: Not always - consider DD and trade count too

### Q: What if no trials meet targets?
**A**: Adjust targets or run Phase 3 to explore further

### Q: Can I use non-Pareto trials?
**A**: Yes, but they're strictly worse than some Pareto trial

### Q: How do I choose ensemble members?
**A**: Select diverse trials across the frontier

---

**For More Examples**: Run `python bin/test_pareto_visualization.py`
