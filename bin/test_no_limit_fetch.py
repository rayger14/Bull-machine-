#!/usr/bin/env python3
"""
Test fetching maximum data with large limit and no time filters.
"""

import requests
import os
from pathlib import Path
from datetime import datetime

# Load API key
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, _, value = line.partition('=')
                if key.strip() == 'COINGLASS_API_KEY':
                    os.environ['COINGLASS_API_KEY'] = value.strip()
                    break

api_key = os.getenv("COINGLASS_API_KEY")
base_url = "https://open-api-v4.coinglass.com/api/futures"
headers = {
    "CG-API-KEY": api_key,
    "accept": "application/json"
}

print("=" * 80)
print("TESTING MAXIMUM DATA FETCH")
print("=" * 80)

# Try fetching with maximum limit and no time filters
test_cases = [
    {
        "name": "Max limit, no time filters, 8h interval",
        "params": {
            "symbol": "BTCUSDT",
            "exchange": "Binance",
            "interval": "h8",
            "limit": 5000  # Try very large limit
        }
    },
    {
        "name": "Max limit, no time filters, 4h interval (minimum for hobbyist)",
        "params": {
            "symbol": "BTCUSDT",
            "exchange": "Binance",
            "interval": "h4",
            "limit": 5000
        }
    },
    {
        "name": "Try with 'time' parameter instead of 'endTime'",
        "params": {
            "symbol": "BTCUSDT",
            "exchange": "Binance",
            "interval": "h8",
            "limit": 1000,
            "time": int(datetime(2022, 1, 1).timestamp() * 1000)
        }
    },
    {
        "name": "Try with 'start' and 'end' parameters",
        "params": {
            "symbol": "BTCUSDT",
            "exchange": "Binance",
            "interval": "h8",
            "limit": 1000,
            "start": int(datetime(2022, 1, 1).timestamp() * 1000),
            "end": int(datetime(2022, 2, 1).timestamp() * 1000)
        }
    },
]

for test in test_cases:
    print(f"\n{'=' * 80}")
    print(f"TEST: {test['name']}")
    print(f"{'=' * 80}")
    print(f"Params: {test['params']}")

    try:
        url = f"{base_url}/funding-rate/history"
        response = requests.get(url, headers=headers, params=test['params'], timeout=30)

        if response.status_code == 200:
            data = response.json()

            if data.get('code') == '0' and data.get('data'):
                records = data['data']
                print(f"✅ Got {len(records)} records")

                timestamps = [int(r['time']) for r in records]
                oldest = datetime.fromtimestamp(min(timestamps)/1000)
                newest = datetime.fromtimestamp(max(timestamps)/1000)

                print(f"   Date range: {oldest} to {newest}")
                print(f"   Days of data: {(max(timestamps) - min(timestamps)) / (1000 * 86400):.1f}")
                print(f"   First record: {records[0]}")
            else:
                print(f"❌ Error: {data}")
        else:
            print(f"❌ HTTP {response.status_code}")
            print(f"   Body: {response.text[:300]}")

    except Exception as e:
        print(f"❌ Exception: {e}")

print(f"\n{'=' * 80}")
print("TESTS COMPLETE")
print("=" * 80)
