#!/usr/bin/env python3
"""Quick triage check for feature store health - diagnose zeros/defaults."""

import pandas as pd
import sys

# Load feature store
store_path = "data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet"
print(f"Loading: {store_path}")

try:
    df = pd.read_parquet(store_path)
except FileNotFoundError:
    print(f"❌ File not found: {store_path}")
    sys.exit(1)

print(f"✅ Loaded {len(df)} bars × {len(df.columns)} columns\n")

# Check critical archetype features
checks = {
    "tf4h_boms_displacement": (df.get("tf4h_boms_displacement", pd.Series([0]*len(df))) > 0).mean() if "tf4h_boms_displacement" in df.columns else -1,
    "tf1d_boms_strength": (df.get("tf1d_boms_strength", pd.Series([0]*len(df))) > 0).mean() if "tf1d_boms_strength" in df.columns else -1,
    "tf4h_fusion_score": (df.get("tf4h_fusion_score", pd.Series([0]*len(df))) != 0).mean() if "tf4h_fusion_score" in df.columns else -1,
}

print("=" * 80)
print("FEATURE HEALTH CHECK (% non-zero)")
print("=" * 80)
for feature, pct in checks.items():
    status = "✅" if pct > 0.10 else "❌"
    if pct == -1:
        print(f"{status} {feature:30s} MISSING")
    else:
        print(f"{status} {feature:30s} {pct*100:5.1f}% non-zero")

print()
print("=" * 80)
print("SAMPLE VALUES (first 10 bars)")
print("=" * 80)

smc_cols = [c for c in df.columns if any(x in c.lower() for x in ["boms", "fusion", "pti"])]
if smc_cols:
    print(df[smc_cols].head(10).to_string())
else:
    print("❌ No SMC/fusion columns found!")
