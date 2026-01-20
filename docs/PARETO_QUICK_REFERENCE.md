# Pareto Visualization Quick Reference

## Command Cheat Sheet

```bash
# Basic usage
python bin/visualize_pareto_frontier.py \
    --study-name bear_phase2_tuning \
    --db-path optuna_studies.db

# Custom output location
python bin/visualize_pareto_frontier.py \
    --study-name bear_phase2_tuning \
    --db-path optuna_studies.db \
    --output-dir results/custom_analysis

# Test with synthetic data
python bin/test_pareto_visualization.py
```

## Output Files at a Glance

| File | Type | Purpose | Use When |
|------|------|---------|----------|
| `pareto_3d.html` | Interactive 3D | Explore multi-objective space | Initial analysis, presentations |
| `pareto_2d_projections.png` | Static 2D plots | Understand trade-offs | Reports, slides |
| `parameter_sensitivity.png` | Heatmap + bars | Identify key parameters | Phase 3 planning |
| `pareto_distributions.png` | Histograms | Compare Pareto vs all | Performance validation |
| `pareto_trials.csv` | Data export | Further analysis | Excel, custom tools |
| `pareto_analysis_summary.txt` | Text report | Quick overview | Command-line review |

## Visual Guide

### 3D Interactive Plot (`pareto_3d.html`)

```
        Max Drawdown (Z)
              ↑
              |    ⬥ ← Pareto optimal (red)
              |   ⬥
              |  ⬥ • ← All trials (gray)
              | ⬥ •
              |⬥ • •
              •─────────────→ Trade Count (Y)
             /
            /
     Profit Factor (X)
```

**Key:**
- Red diamonds: Pareto-optimal trials
- Gray dots: Dominated trials
- Hover for details
- Rotate, zoom, pan with mouse

### 2D Projections (`pareto_2d_projections.png`)

```
┌─────────────────────┬─────────────────────┐
│ PF vs Trade Count   │ PF vs Max Drawdown  │
│                     │                     │
│ [Green line: PF>1.3]│ [Target zones shown]│
│ [Blue zone: 25-40]  │                     │
├─────────────────────┼─────────────────────┤
│ TC vs Max Drawdown  │ Pareto Ranked by PF │
│                     │                     │
│ [Target zones]      │ [Bar chart]         │
└─────────────────────┴─────────────────────┘
```

### Parameter Sensitivity (`parameter_sensitivity.png`)

```
Heatmap (left)           Importance (right)
─────────────            ─────────────────
         PF  TC  DD      param_x ████████ 0.65
param_a [██][  ][  ]     param_y ██████   0.48
param_b [  ][██][  ]     param_z ████     0.32
param_c [  ][  ][██]     param_w ██       0.15
```

**Colors:**
- Red: Strong positive correlation
- Blue: Strong negative correlation
- White: No correlation

## Interpretation Guide

### What is Pareto Optimal?

A trial is Pareto optimal if **no other trial beats it in ALL objectives**.

**Example:**
```
Trial A: PF=1.5, TC=30, DD=-20%  ← Pareto optimal (balanced)
Trial B: PF=1.6, TC=35, DD=-25%  ← Pareto optimal (high PF)
Trial C: PF=1.4, TC=32, DD=-18%  ← Pareto optimal (low DD)
Trial D: PF=1.3, TC=28, DD=-22%  ← Dominated by A
```

### Reading the Frontier

```
High PF
   ↑
   |     Region A: Aggressive
   |     (High PF, High DD)
   |         ⬥
   |
   |    Region B: Balanced  ⬥
   |           ⬥
   |
   |  Region C: Conservative
   |  (Lower PF, Low DD)  ⬥
   |
   └────────────────────────→ Low DD
```

### Target Zone Analysis

```
✓ Meets all targets:
  - PF > 1.3
  - 25 ≤ Trades ≤ 40
  - -30% ≤ DD ≤ -15%

⚠ Close to targets:
  - 1 or 2 criteria met

✗ Outside targets:
  - 0 criteria met
```

## Decision Making

### Choosing Best Trial

1. **Risk-Averse**: Choose trial with lowest DD among Pareto set
2. **Performance-Focused**: Choose trial with highest PF
3. **Balanced**: Choose trial closest to target zone center
4. **Ensemble**: Select diverse trials across frontier

### Parameter Tuning Priority

```
High Priority    → |█████████| > 0.5 correlation
Medium Priority  → |████     | 0.3-0.5
Low Priority     → |█        | < 0.3
```

Focus Phase 3 tuning on high-priority parameters.

## Common Patterns

### Pattern 1: PF-DD Trade-off
```
High PF ←→ High DD
```
**Action:** Find acceptable DD level, maximize PF within that constraint

### Pattern 2: Trade Count Sweet Spot
```
Too few trades (< 25)   → Underutilized
Target range (25-40)     → Optimal
Too many trades (> 40)   → Overtrading
```
**Action:** Constrain to target range

### Pattern 3: Parameter Clustering
```
High-performing trials cluster around:
  - confidence_threshold: 0.7-0.8
  - risk_per_trade: 0.02-0.03
```
**Action:** Narrow search space for Phase 3

## Troubleshooting

### Problem: All trials look similar in 3D plot

**Cause:** Objectives have different scales

**Solution:** Normalize objectives or use 2D projections

### Problem: No clear Pareto frontier

**Cause:** Optimization hasn't converged

**Solution:** Run more trials or adjust objective weights

### Problem: Too many Pareto-optimal trials

**Cause:** Low diversity in trials or flat objective space

**Solution:** Increase search space or add objective weighting

## Workflow Integration

```
Phase 2: Multi-Objective Tuning
    ↓
Generate Pareto Visualizations
    ↓
Analyze Trade-offs
    ├─→ Identify target zone trials
    ├─→ Find high-impact parameters
    └─→ Detect parameter patterns
    ↓
Phase 3: Ensemble Optimization
    └─→ Use top Pareto trials as ensemble members
```

## Quick Wins

1. **Fast Analysis**: Check `pareto_analysis_summary.txt` first
2. **Target ID**: Look for green checkmarks in summary report
3. **Parameter Focus**: Use sensitivity bars to prioritize tuning
4. **Ensemble Selection**: Pick 5-10 diverse Pareto trials from CSV

## Advanced Tips

### Custom Filtering

Edit script to filter by win rate:
```python
# Only high win rate trials
filtered = pareto_trials[pareto_trials['attr_win_rate'] > 0.5]
```

### Objective Weighting

Adjust trade count scoring:
```python
# Prefer fewer trades
valid['trade_count_score'] = -np.abs(valid['trade_count'] - 27.5)
```

### Export Selection

Select specific trials:
```python
# Top 10 by PF, meeting DD constraint
top10 = pareto_trials[
    (pareto_trials['max_drawdown'] >= -0.25) &
    (pareto_trials['profit_factor'] > 1.3)
].nlargest(10, 'profit_factor')
```

## Performance Metrics

| Study Size | Runtime | Memory | Output Size |
|-----------|---------|---------|-------------|
| 100 trials | 2 sec | 50 MB | 5 MB |
| 1000 trials | 8 sec | 80 MB | 12 MB |
| 5000 trials | 30 sec | 200 MB | 30 MB |

## Next Steps After Visualization

1. Review `pareto_analysis_summary.txt`
2. Open `pareto_3d.html` in browser
3. Identify trials in target zone
4. Check `parameter_sensitivity.png` for Phase 3 focus
5. Export top trials to `configs/phase3_candidates.csv`
6. Design ensemble combining diverse strategies

---

**Quick Help:**
```bash
python bin/visualize_pareto_frontier.py --help
```

**Full Guide:** See `docs/PARETO_VISUALIZATION_GUIDE.md`
