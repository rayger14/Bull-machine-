#!/usr/bin/env python3
"""
Test historical data access with different parameter combinations.
"""

import requests
import os
from pathlib import Path
from datetime import datetime
import json

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
print("TESTING HISTORICAL DATA ACCESS")
print("=" * 80)

# Test different parameter combinations for 2022-2023 period
test_cases = [
    {
        "name": "Using startTime + endTime",
        "params": {
            "symbol": "BTCUSDT",
            "exchange": "Binance",
            "interval": "h8",
            "limit": 100,
            "startTime": int(datetime(2022, 1, 1).timestamp() * 1000),
            "endTime": int(datetime(2022, 1, 31).timestamp() * 1000)
        }
    },
    {
        "name": "Using only startTime (early 2022)",
        "params": {
            "symbol": "BTCUSDT",
            "exchange": "Binance",
            "interval": "h8",
            "limit": 100,
            "startTime": int(datetime(2022, 1, 1).timestamp() * 1000)
        }
    },
    {
        "name": "Using endTime (early 2022)",
        "params": {
            "symbol": "BTCUSDT",
            "exchange": "Binance",
            "interval": "h8",
            "limit": 100,
            "endTime": int(datetime(2022, 1, 31).timestamp() * 1000)
        }
    },
    {
        "name": "Using endTime (late 2023)",
        "params": {
            "symbol": "BTCUSDT",
            "exchange": "Binance",
            "interval": "h8",
            "limit": 100,
            "endTime": int(datetime(2023, 12, 31).timestamp() * 1000)
        }
    },
    {
        "name": "No time params (get latest)",
        "params": {
            "symbol": "BTCUSDT",
            "exchange": "Binance",
            "interval": "h8",
            "limit": 100
        }
    },
]

for test in test_cases:
    print(f"\n{'=' * 80}")
    print(f"TEST: {test['name']}")
    print(f"{'=' * 80}")
    print(f"Params: {json.dumps(test['params'], indent=2)}")

    try:
        url = f"{base_url}/funding-rate/history"
        response = requests.get(url, headers=headers, params=test['params'], timeout=10)

        if response.status_code == 200:
            data = response.json()

            if data.get('code') == '0' and data.get('data'):
                records = data['data']
                print(f"✅ Got {len(records)} records")

                # Parse timestamps
                timestamps = [int(r['time']) for r in records]
                oldest = datetime.fromtimestamp(min(timestamps)/1000)
                newest = datetime.fromtimestamp(max(timestamps)/1000)

                print(f"   Date range: {oldest} to {newest}")
                print(f"   Sample (first): {records[0]}")
                print(f"   Sample (last): {records[-1]}")
            else:
                print(f"❌ Error: {data.get('msg', 'Unknown')}")
        else:
            print(f"❌ HTTP {response.status_code}")

    except Exception as e:
        print(f"❌ Exception: {e}")

print(f"\n{'=' * 80}")
print("TESTS COMPLETE")
print("=" * 80)
