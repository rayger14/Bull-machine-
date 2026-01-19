# Pareto Frontier Visualization Guide

## Overview

The Pareto frontier visualization script analyzes multi-objective optimization results from Optuna studies and generates publication-quality visualizations for understanding trade-offs between competing objectives.

## Objectives Analyzed

1. **Profit Factor** (maximize): Risk-adjusted profitability
2. **Annual Trade Count** (optimize to range): Target 25-40 trades/year
3. **Max Drawdown** (minimize): Risk exposure

## Installation

```bash
pip install optuna plotly matplotlib seaborn numpy pandas scipy
```

## Usage

### Basic Usage

```bash
python bin/visualize_pareto_frontier.py \
    --study-name bear_phase2_tuning \
    --db-path optuna_studies.db
```

### Custom Output Directory

```bash
python bin/visualize_pareto_frontier.py \
    --study-name bear_phase2_tuning \
    --db-path optuna_studies.db \
    --output-dir results/custom_analysis
```

## Output Files

### 1. Interactive 3D Plot (`pareto_3d.html`)

**Features:**
- 3D scatter plot with Profit Factor, Trade Count, and Max Drawdown axes
- Pareto-optimal trials highlighted in red (diamond markers)
- All other trials shown in gray
- Interactive hover tooltips showing trial details
- Rotatable, zoomable view

**Usage:**
- Open in web browser
- Rotate with mouse drag
- Zoom with scroll wheel
- Hover over points for details

### 2. 2D Projections (`pareto_2d_projections.png`)

Four projection plots:

#### Plot 1: Profit Factor vs Trade Count
- Shows relationship between profitability and activity
- Target zones: PF > 1.3 (green line), Trades 25-40 (blue shaded)

#### Plot 2: Profit Factor vs Max Drawdown
- Reveals risk-reward trade-offs
- Target zones: PF > 1.3, DD -15% to -30%

#### Plot 3: Trade Count vs Max Drawdown
- Shows activity vs risk relationship
- Target zones overlaid

#### Plot 4: Pareto Trials Ranked
- Horizontal bar chart of Pareto trials by PF
- Color-coded from low to high performance

### 3. Parameter Sensitivity (`parameter_sensitivity.png`)

Two panels:

#### Panel 1: Correlation Heatmap
- Shows correlation between each parameter and each objective
- Red = positive correlation
- Blue = negative correlation
- Values range from -1 to +1

#### Panel 2: Parameter Importance
- Bar chart of average absolute correlation
- Identifies most influential parameters
- Sorted by importance

### 4. Distribution Comparison (`pareto_distributions.png`)

Three histograms comparing Pareto-optimal trials vs all trials:
- Profit Factor distribution
- Trade Count distribution
- Max Drawdown distribution

Shows mean values for each group to quantify improvement.

### 5. CSV Export (`pareto_trials.csv`)

Structured data of Pareto-optimal trials:
- Trial number
- Objective values (PF, trades, DD)
- All parameter values
- Sorted by Profit Factor

**Use cases:**
- Import into Excel/Google Sheets
- Further statistical analysis
- Configuration file generation

### 6. Summary Report (`pareto_analysis_summary.txt`)

Text report containing:
- Trial counts and percentages
- Objective statistics (mean, std, min, max)
- Top 10 Pareto-optimal trials
- Target zone analysis
- Trials meeting all targets

## Understanding Pareto Optimality

### What is Pareto Optimal?

A trial is **Pareto optimal** if no other trial is strictly better in all objectives. For a trial to be dominated, another trial must be:
- Better or equal in ALL objectives
- Strictly better in AT LEAST ONE objective

### Example

Consider three trials:

| Trial | Profit Factor | Trades | Max DD |
|-------|--------------|--------|---------|
| A     | 1.5          | 30     | -20%    |
| B     | 1.6          | 35     | -25%    |
| C     | 1.4          | 32     | -18%    |

**Analysis:**
- Trial A: Pareto optimal (balanced)
- Trial B: Pareto optimal (higher PF, acceptable DD)
- Trial C: Pareto optimal (lowest DD)

None dominates the others because each has at least one objective where it's best.

### Trade-off Interpretation

The Pareto frontier reveals:

1. **High PF, High DD**: Aggressive strategies with better returns but higher risk
2. **Moderate PF, Low DD**: Conservative strategies with stable but lower returns
3. **Optimal Trade Count**: Strategies hitting the 25-40 target range

## Target Zone Analysis

### Defined Targets

| Objective        | Target        | Rationale                           |
|------------------|---------------|-------------------------------------|
| Profit Factor    | > 1.3         | Sustainable profitability           |
| Trade Count      | 25-40/year    | Sufficient activity without overtrading |
| Max Drawdown     | -15% to -30%  | Manageable risk exposure            |

### Identifying Best Trials

Look for trials in the target zone (all conditions met):
- Pareto optimal
- PF > 1.3
- 25 ≤ Trades ≤ 40
- -30% ≤ DD ≤ -15%

These represent the **sweet spot** balancing all objectives.

## Parameter Sensitivity Insights

### High Positive Correlation
- Parameter increases → Objective increases
- Example: Higher confidence threshold → Higher PF

### High Negative Correlation
- Parameter increases → Objective decreases
- Example: Tighter stop loss → Lower DD

### Low Correlation (near 0)
- Parameter has minimal effect on objective
- Candidate for simplification or removal

### Action Items

1. **High importance parameters**: Fine-tune carefully in Phase 3
2. **Low importance parameters**: Consider fixing or removing
3. **Correlated parameters**: Check for redundancy

## Workflow Integration

### Phase 2 (Current): Multi-Objective Tuning
```bash
# Run optimization
python bin/phase2_multi_objective_tuning.py --config configs/bear_archetypes_phase2.json

# Visualize results
python bin/visualize_pareto_frontier.py --study-name bear_phase2_tuning --db-path optuna_studies.db
```

### Phase 3: Ensemble Optimization

Use insights from Pareto analysis:
1. Select top 5-10 Pareto-optimal trials
2. Create ensemble combining diverse strategies
3. Focus tuning on high-impact parameters

## Troubleshooting

### Issue: No Pareto trials found

**Causes:**
- Too few trials completed
- All trials have NaN values

**Solution:**
```bash
# Check study status
python -c "import optuna; study = optuna.load_study('bear_phase2_tuning', 'sqlite:///optuna_studies.db'); print(f'Completed: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)}')"
```

### Issue: Empty parameter sensitivity

**Cause:** No parameter columns in trial data

**Solution:** Ensure optimization is logging parameters correctly

### Issue: HTML plot not displaying

**Cause:** Missing Plotly dependency

**Solution:**
```bash
pip install plotly kaleido
```

## Advanced Usage

### Filtering Trials

Edit script to add custom filters:

```python
# Only analyze recent trials
recent_trials = all_trials[all_trials['trial_number'] >= 100]
pareto_trials, non_pareto = identify_pareto_front(recent_trials)
```

### Custom Objectives

Modify trade count scoring for different targets:

```python
# Target 15-25 trades instead of 25-40
valid['trade_count_score'] = -np.abs(valid['trade_count'] - 20.0)
```

### Export Format

Change output format:

```python
# Export as JSON instead of CSV
export_df.to_json(output_path, orient='records', indent=2)
```

## Visualization Best Practices

### For Presentations

1. Use HTML 3D plot for interactive exploration
2. Use 2D projections for slides (high-res PNG)
3. Highlight specific trials of interest

### For Reports

1. Include distribution comparison to show improvement
2. Add parameter sensitivity to justify tuning focus
3. Reference summary statistics from text report

### For Decision Making

1. Start with 2D projections to identify trade-offs
2. Use 3D plot to understand multi-objective space
3. Export CSV for detailed trial-by-trial comparison

## Example Analysis Workflow

```bash
# 1. Generate all visualizations
python bin/visualize_pareto_frontier.py \
    --study-name bear_phase2_tuning \
    --db-path optuna_studies.db

# 2. Review summary report
cat results/phase3_frontier/pareto_analysis_summary.txt

# 3. Open interactive 3D plot
open results/phase3_frontier/pareto_3d.html

# 4. Examine 2D projections
open results/phase3_frontier/pareto_2d_projections.png

# 5. Identify high-impact parameters
open results/phase3_frontier/parameter_sensitivity.png

# 6. Export top trials for Phase 3
head -n 11 results/phase3_frontier/pareto_trials.csv > configs/phase3_top_trials.csv
```

## Performance Notes

- **Fast**: < 10 seconds for 1000 trials
- **Memory**: ~100 MB for typical study
- **Scalability**: Tested up to 10,000 trials

## Next Steps

1. **Phase 3 Preparation**: Use Pareto trials as ensemble candidates
2. **Parameter Pruning**: Remove low-importance parameters
3. **Target Refinement**: Adjust target zones based on results
4. **Ensemble Design**: Combine diverse Pareto-optimal strategies

## References

- [Pareto Efficiency](https://en.wikipedia.org/wiki/Pareto_efficiency)
- [Multi-Objective Optimization](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/007_multi_objective.html)
- [Plotly Documentation](https://plotly.com/python/)

---

**Generated by:** Bull Machine V2 Optimization Pipeline
**Version:** 2.0.0
**Last Updated:** 2025-11-19
