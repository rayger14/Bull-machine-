#!/usr/bin/env python3
"""Equity drawdown check — called by health_check.sh"""
import csv, sys

path = sys.argv[1]
rows = []
with open(path) as f:
    for row in csv.reader(f):
        try:
            rows.append(float(row[1]))
        except:
            pass

if not rows:
    print("[WARN] No equity data found")
    sys.exit(0)

peak = max(rows)
current = rows[-1]
dd = (current - peak) / peak * 100
total_ret = (current - 100000) / 100000 * 100

if dd < -15:
    print(f"[FAIL] Equity: ${current:,.0f} | Peak: ${peak:,.0f} | Drawdown: {dd:.1f}% | Return: {total_ret:.1f}%")
    sys.exit(1)
elif dd < -8:
    print(f"[WARN] Equity: ${current:,.0f} | Peak: ${peak:,.0f} | Drawdown: {dd:.1f}% | Return: {total_ret:.1f}%")
    sys.exit(0)
else:
    print(f"[OK]   Equity: ${current:,.0f} | Peak: ${peak:,.0f} | Drawdown: {dd:.1f}% | Return: {total_ret:.1f}%")
    sys.exit(0)
