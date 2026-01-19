#!/usr/bin/env python3
"""
Debug the adaptive max-hold logic directly
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from bin.backtest_knowledge_v2 import KnowledgeParams

# Load SPY 2024 data
df = pd.read_parquet('data/features_mtf/SPY_1H_2024-01-01_to_2024-12-31.parquet')

print(f"Loaded {len(df)} bars\n")

# Check what Wyckoff phases we have
print("Wyckoff Phase Distribution:")
if 'tf1d_wyckoff_phase' in df.columns:
    print(df['tf1d_wyckoff_phase'].value_counts())
else:
    print("⚠️  tf1d_wyckoff_phase column not found!")

print("\nWyckoff Score Stats:")
if 'tf1d_wyckoff_score' in df.columns:
    print(f"  Min: {df['tf1d_wyckoff_score'].min():.2f}")
    print(f"  Max: {df['tf1d_wyckoff_score'].max():.2f}")
    print(f"  Mean: {df['tf1d_wyckoff_score'].mean():.2f}")
else:
    print("⚠️  tf1d_wyckoff_score column not found!")

# Check if there are any markup phases with high scores
print("\nMarkup Phases with High Scores:")
if 'tf1d_wyckoff_phase' in df.columns and 'tf1d_wyckoff_score' in df.columns:
    markup_high = df[(df['tf1d_wyckoff_phase'] == 'markup') & (df['tf1d_wyckoff_score'] >= 0.6)]
    print(f"  Count: {len(markup_high)} bars")
    if len(markup_high) > 0:
        print(f"  Date range: {markup_high.index.min()} to {markup_high.index.max()}")
else:
    print("  Cannot check - columns missing")

# Show first few rows with their Wyckoff data
print("\nFirst 10 Rows:")
cols_to_show = ['close', 'atr_14']
if 'tf1d_wyckoff_phase' in df.columns:
    cols_to_show.append('tf1d_wyckoff_phase')
if 'tf1d_wyckoff_score' in df.columns:
    cols_to_show.append('tf1d_wyckoff_score')

print(df[cols_to_show].head(10))
