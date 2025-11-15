# Knowledge v2.0 Testing Workflow

## Quick Start: Precompute-Once, Test-Cheaply Pattern

The key insight: **compute features once, reuse for every A/B/C test**. This prevents the 40+ minute stalls from on-the-fly feature computation.

## Step 1: Build Feature Store v2 (One-Time)

Build the complete feature store with all 104 columns (Week 1-4 features):

```bash
# ETH Q3 2024 (takes ~2-3 minutes for 2,166 bars)
python3 bin/build_feature_store_v2.py \
  --asset ETH \
  --start 2024-07-01 \
  --end 2024-09-30 \
  --include-week1-4

# Output: data/features_v2/ETH_1H_2024-07-01_to_2024-09-30.parquet
```

**Quick integrity check:**
```bash
python3 - <<'PY'
import pandas as pd
df = pd.read_parquet('data/features_v2/ETH_1H_2024-07-01_to_2024-09-30.parquet')
print(f"✓ Rows: {len(df)}")
print(f"✓ Columns: {len(df.columns)}")
required = ['boms_detected', 'pti_score', 'frvp_poc', 'macro_regime']
missing = [c for c in required if c not in df.columns]
print(f"✓ Required features missing: {missing if missing else 'None'}")
PY
```

## Step 2: Smoke Test (10 Days)

Before running full Q3 2024, validate on a tiny dataset:

```bash
# Build 10-day feature store (completes in seconds)
python3 bin/build_feature_store_v2.py \
  --asset ETH \
  --start 2024-09-01 \
  --end 2024-09-10 \
  --include-week1-4

# Verify it created: data/features_v2/ETH_1H_2024-09-01_to_2024-09-10.parquet
ls -lh data/features_v2/ETH_1H_2024-09-01_to_2024-09-10.parquet
```

## Step 3: A/B/C Testing

### Test Configurations

Three configs are ready in `configs/knowledge_v2/`:

1. **Baseline** (`ETH_baseline.json`): `knowledge_v2.enabled = false`
2. **Shadow** (`ETH_shadow_mode.json`): `knowledge_v2.enabled = true, shadow_mode = true`
3. **Active** (`ETH_v2_active.json`): `knowledge_v2.enabled = true, shadow_mode = false`

### Current Status

**Note:** The hybrid_runner currently computes features on-the-fly. To use pre-built feature stores, you'll need to either:

1. Update `bin/live/hybrid_runner.py` to accept `--features` parameter
2. Create a new `bin/compare_baseline_vs_ml.py` script (recommended)
3. Manually patch configs to load from parquet instead of computing

### Recommended Approach: Create Comparison Tool

```python
# bin/compare_baseline_vs_ml.py (pseudocode)
# Load pre-built feature store
features = pd.read_parquet(args.features_path)

# Run 3 tests:
for config_name in ['baseline', 'shadow', 'active']:
    config = load_config(f'configs/knowledge_v2/ETH_{config_name}.json')

    # Use features directly instead of recomputing
    results = run_backtest(features, config)

    # Save metrics
    save_metrics(results, f'reports/v2_ab_test/ETH_Q3_{config_name}.json')
```

## Step 4: Verify Hook Firing

After shadow mode test, confirm hooks are actually firing:

```bash
# Count hook firings in shadow logs
python3 - <<'PY'
import json, pathlib, collections
p = pathlib.Path('logs/knowledge_v2/shadow_hooks.jsonl')  # adjust path
if p.exists():
    ctr = collections.Counter()
    for line in p.open():
        j = json.loads(line)
        for r in j.get("reasons", []):
            if r.startswith("HOOK:"):
                ctr[r.split(" ")[0]] += 1
    print("Hook firing counts:", ctr)
else:
    print("No shadow log file found - hooks may not be integrated yet")
PY
```

Expected output:
```
Hook firing counts: Counter({'HOOK:BOMS+': 23, 'HOOK:FakeoutPenalty': 18, ...})
```

## Step 5: Performance Analysis

Compare metrics across A/B/C:

```bash
# Expected metrics files
reports/v2_ab_test/ETH_Q3_baseline.json
reports/v2_ab_test/ETH_Q3_shadow.json
reports/v2_ab_test/ETH_Q3_active.json
```

**Acceptance gates** (must meet ≥3 of 4):
- Profit Factor: +0.10 uplift
- Sharpe Ratio: +0.10 uplift
- Max Drawdown: ≤ baseline
- Trade Count: ≥ 80% of baseline

**Baseline vs Shadow validation:**
- PNL should be **identical** (proves shadow mode works)
- Shadow logs show what *would* have changed

## Step 6: Ablation Studies

If Active mode doesn't beat baseline, test hooks individually:

```bash
# Test with only one hook enabled at a time
for hook in boms fakeout pti frvp macro_echo; do
    # Edit config: enable only $hook
    python3 bin/compare_baseline_vs_ml.py \
      --features data/features_v2/ETH_1H_Q3_2024.parquet \
      --config configs/knowledge_v2/ETH_ablation_${hook}.json \
      --out reports/v2_ablations/ETH_${hook}_only.json
done
```

Keep the winners, adjust penalty/bonus magnitudes for losers.

## Current Limitations

**As of this commit:**
- ✅ Feature store builder works (all 104 columns)
- ✅ Test configs ready (baseline/shadow/active)
- ✅ Code quality refactored
- ❌ hybrid_runner needs `--features` parameter
- ❌ compare_baseline_vs_ml.py not yet implemented

**Next steps:**
1. Add feature store loading to hybrid_runner
2. OR create dedicated compare_baseline_vs_ml.py script
3. Run actual A/B/C tests with pre-built features

## Performance Notes

**Feature computation time** (ETH Q3 2024, 2,166 bars):
- On-the-fly: 40+ minutes per test (unusable)
- Pre-built once: 2-3 minutes total
- Reusing cache: Seconds per test

**Always pre-build features for testing.**

## File Manifest

```
configs/knowledge_v2/
  ├── ETH_baseline.json       # v2 disabled
  ├── ETH_shadow_mode.json    # v2 logs only
  └── ETH_v2_active.json      # v2 affects decisions

data/features_v2/
  ├── ETH_1H_2024-07-01_to_2024-09-30.parquet  # Full Q3 (318KB)
  └── ETH_1H_2024-09-01_to_2024-09-10.parquet  # Smoke test (100KB)

engine/
  ├── structure/          # Week 1: Internal/External, BOMS, Range, Squiggle
  ├── psychology/         # Week 2: Fakeout Intensity, PTI
  ├── volume/             # Week 3: FRVP
  ├── exits/              # Week 4: Macro Echo, Multi-Modal
  └── fusion/knowledge_hooks.py  # Integration layer

bin/
  └── build_feature_store_v2.py  # Feature store builder
```
