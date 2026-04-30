#!/usr/bin/env python3
"""Heartbeat recency check — called by health_check.sh"""
import json, sys
from datetime import datetime, timezone

path = sys.argv[1]
with open(path) as f:
    d = json.load(f)

ts = d.get('timestamp', '')
btc = d.get('btc_price', 0)
equity = d.get('equity', 0)
positions = d.get('open_positions', 0)
regime = d.get('regime', '?')

try:
    hb = datetime.fromisoformat(ts)
    now = datetime.now(timezone.utc)
    age_h = (now - hb).total_seconds() / 3600

    if age_h < 3:
        print(f"[OK]   Heartbeat: {ts[:16]} ({age_h:.1f}h ago) | BTC=${btc:,.0f} | equity=${equity:,.0f} | regime={regime} | positions={positions}")
        sys.exit(0)
    else:
        print(f"[FAIL] Heartbeat stale: {ts[:16]} ({age_h:.1f}h ago) — engine may be dead or stuck")
        sys.exit(1)
except Exception as e:
    print(f"[FAIL] Heartbeat parse error: {e}")
    sys.exit(1)
