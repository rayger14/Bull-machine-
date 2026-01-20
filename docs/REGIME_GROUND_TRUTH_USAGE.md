# Regime Ground Truth Labels - Usage Guide

## Overview

The regime ground truth file (`data/regime_ground_truth_2020_2024.json`) provides manually labeled market regimes for the 2020-2024 period, based on historical BTC price action, volatility, and known market events.

## File Structure

```json
{
  "metadata": {
    "version": "1.0",
    "created": "2024-11-14",
    "description": "Ground truth regime labels for BTC/crypto markets 2020-2024",
    "methodology": "...",
    "regimes": { ... }
  },
  "yearly": { "2020": "risk_on", ... },
  "quarterly": { "2020-Q1": "crisis", ... },
  "monthly": { "2020-01": "risk_on", ... },
  "key_events": { ... },
  "validation_notes": { ... }
}
```

## Regime Definitions

| Regime | Description | Characteristics |
|--------|-------------|-----------------|
| **risk_on** | Bull market | Risk appetite, uptrends, low-moderate volatility |
| **neutral** | Choppy/sideways | Transitional periods, mixed signals, range-bound |
| **risk_off** | Bear market | Fear, crashes, high volatility, deleveraging |
| **crisis** | Extreme panic | Black swan events, severe deleveraging |

## Label Distribution (2020-2024)

```
Total: 60 months
  risk_on:  28 months (46.7%)
  neutral:  17 months (28.3%)
  risk_off: 11 months (18.3%)
  crisis:    4 months (6.7%)
```

### Crisis Events (4 months)
- **2020-03**: COVID-19 crash (BTC: $9K → $3.8K)
- **2022-05**: LUNA/UST collapse ($40B wipeout)
- **2022-06**: Continued crash aftermath (Celsius, 3AC)
- **2022-11**: FTX collapse ($8B hole)

## Usage Examples

### 1. Load Ground Truth in Python

```python
import json

with open('data/regime_ground_truth_2020_2024.json', 'r') as f:
    gt = json.load(f)

# Get yearly label
regime_2022 = gt['yearly']['2022']  # 'risk_off'

# Get monthly label
regime_may_2022 = gt['monthly']['2022-05']  # 'crisis'

# Get all monthly labels
monthly_labels = gt['monthly']
```

### 2. Validate Regime Classifier

```bash
# Validate against monthly labels (default)
python bin/validate_regime_classifier.py

# Validate against quarterly labels
python bin/validate_regime_classifier.py --granularity quarterly

# Use custom model
python bin/validate_regime_classifier.py --model models/my_classifier.pkl
```

The validation script will output:
- Overall accuracy
- Per-regime accuracy
- Confusion matrix
- Precision/recall/F1 scores
- List of misclassified periods

### 3. Backtest with Regime Override

Use ground truth to force specific regimes during backtesting:

```python
from engine.context.regime_classifier import RegimeClassifier

# Create regime override dict
regime_override = {
    "2020": "risk_on",
    "2021": "risk_on",
    "2022": "risk_off",
    "2023": "neutral",
    "2024": "risk_on"
}

# Load classifier with override
classifier = RegimeClassifier.load(
    model_path="models/regime_classifier_gmm.pkl",
    feature_order=feature_order,
    regime_override=regime_override
)

# Predictions will be overridden by year
result = classifier.classify(macro_features, timestamp)
# For 2022 timestamps, will return 'risk_off' regardless of features
```

### 4. Filter Backtest by Regime

```python
import json
import pandas as pd

# Load ground truth
with open('data/regime_ground_truth_2020_2024.json', 'r') as f:
    gt = json.load(f)

# Create timestamp → regime mapping
regime_map = {}
for month_str, regime in gt['monthly'].items():
    # Convert "2022-05" to datetime range
    start = pd.to_datetime(month_str + "-01")
    end = (start + pd.DateOffset(months=1)) - pd.Timedelta(seconds=1)
    regime_map[(start, end)] = regime

# Filter backtest results
def get_regime(timestamp):
    for (start, end), regime in regime_map.items():
        if start <= timestamp <= end:
            return regime
    return None

backtest_df['regime'] = backtest_df.index.map(get_regime)

# Analyze performance by regime
for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
    regime_df = backtest_df[backtest_df['regime'] == regime]
    print(f"{regime}: {len(regime_df)} trades, "
          f"Sharpe={regime_df['returns'].mean() / regime_df['returns'].std():.2f}")
```

## Key Historical Periods

### 2020: Recovery Year (risk_on)
- **Q1**: COVID crash (March = crisis)
- **Q2-Q4**: Strong recovery on Fed stimulus

### 2021: Peak Bull (risk_on)
- **Q1-Q2**: Parabolic rise to $64K
- **Q3**: Summer consolidation (neutral)
- **Q4**: Final push to $69K ATH

### 2022: Bear Market (risk_off)
- **Q1**: Fed tightening begins, macro deteriorates
- **Q2**: Luna crash (May) → bottom $17.6K (June)
- **Q3-Q4**: Slow grind lower, FTX collapse (Nov)

### 2023: Choppy Recovery (neutral)
- **Q1-Q3**: Range-bound $20K-$30K
- **Q4**: ETF optimism breakout (risk_on)

### 2024: New Bull (risk_on)
- **Q1**: ETF approval, rally to $73K ATH (risk_on)
- **Q2-Q3**: Consolidation phase (neutral)
- **Q4**: Post-election rally to $100K+ (risk_on)

## Methodology Notes

### Data Sources
1. **BTC Price Data**: Daily OHLCV from Coinbase (2020-2024)
2. **VIX Data**: Available Oct 2023 onwards only
3. **Historical Events**: Public records of Luna, FTX, ETF approval

### Labeling Criteria
- **Monthly returns**: Large positive = risk_on, large negative = risk_off
- **Volatility**: Extreme vol (>35%) often indicates crisis
- **Trend**: Sustained uptrend = risk_on, downtrend = risk_off, sideways = neutral
- **Events**: Known crashes labeled as crisis

### Crisis vs Risk-off
- **Crisis**: Reserved for extreme events (COVID, Luna, FTX)
- **Risk-off**: General bear market conditions

### Neutral vs Risk-on
- **Neutral**: Choppy, range-bound, unclear direction
- **Risk-on**: Clear uptrend with conviction

## Validation Best Practices

### 1. Use Monthly Granularity
Monthly labels are most reliable. Quarterly/yearly are summaries.

### 2. Handle Edge Cases
Transitions between regimes happen gradually. Consider using:
- Weighted labels (e.g., 70% risk_off, 30% neutral)
- Lag effects (regime change takes time to manifest)

### 3. Crisis is Rare
Only 4/60 months (6.7%) are labeled crisis. Don't expect high accuracy on this class without special handling.

### 4. 2024 is Mixed
2024 shows both bull (Q1, Q4) and consolidation (Q2-Q3). Annual label is "risk_on" but monthly is more nuanced.

## Future Enhancements

### Potential Additions
1. **Intra-month labels**: Higher granularity for volatile months
2. **Confidence scores**: Low/medium/high confidence per label
3. **Transition periods**: Explicit labels for regime changes
4. **Multi-asset**: Extend to ETH, SOL, etc.
5. **2019 data**: Pre-COVID baseline period

### Request Changes
To suggest changes or report labeling errors:
1. Create an issue with specific date and rationale
2. Reference BTC price data and events
3. Propose alternative label with justification

## Quick Reference

```bash
# View ground truth summary
python -c "
import json
with open('data/regime_ground_truth_2020_2024.json') as f:
    gt = json.load(f)
    for year in ['2020', '2021', '2022', '2023', '2024']:
        print(f'{year}: {gt[\"yearly\"][year]}')
"

# Validate classifier
python bin/validate_regime_classifier.py

# Check specific period
python -c "
import json
with open('data/regime_ground_truth_2020_2024.json') as f:
    gt = json.load(f)
    print('May 2022 (Luna crash):', gt['monthly']['2022-05'])
    print('Event:', gt['key_events']['2022-05'])
"
```

## Contact

For questions or issues with ground truth labels, contact the ML team or create an issue in the repo.
