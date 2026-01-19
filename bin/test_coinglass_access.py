#!/usr/bin/env python3
"""
Test CoinGlass API access and determine available data ranges.
"""

import requests
import os
from pathlib import Path
from datetime import datetime

# Load API key from .env
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
if not api_key:
    print("❌ API key not found")
    exit(1)

base_url = "https://open-api-v4.coinglass.com/api/futures"
headers = {
    "CG-API-KEY": api_key,
    "accept": "application/json"
}

print("=" * 80)
print("TESTING COINGLASS API ACCESS")
print("=" * 80)

# Test different endpoints
endpoints = [
    ("funding-rate/history", {
        "symbol": "BTCUSDT",
        "exchange": "Binance",
        "interval": "h8",
        "limit": 10,
        "endTime": int(datetime(2023, 12, 31).timestamp() * 1000)
    }),
    ("openInterest/ohlc-aggregated-history", {
        "symbol": "BTCUSDT",
        "interval": "h1",
        "limit": 10,
        "endTime": int(datetime(2023, 12, 31).timestamp() * 1000)
    }),
    ("openInterest/history", {
        "symbol": "BTCUSDT",
        "exchange": "Binance",
        "interval": "h1",
        "limit": 10,
        "endTime": int(datetime(2023, 12, 31).timestamp() * 1000)
    }),
    ("longShortRatio/history", {
        "symbol": "BTCUSDT",
        "exchange": "Binance",
        "interval": "h1",
        "limit": 10,
        "endTime": int(datetime(2023, 12, 31).timestamp() * 1000)
    }),
    ("liquidation/history", {
        "symbol": "BTCUSDT",
        "interval": "h1",
        "limit": 10,
        "endTime": int(datetime(2023, 12, 31).timestamp() * 1000)
    }),
]

for endpoint, params in endpoints:
    print(f"\n{'=' * 80}")
    print(f"Testing: {endpoint}")
    print(f"{'=' * 80}")
    print(f"Params: {params}")

    try:
        url = f"{base_url}/{endpoint}"
        response = requests.get(url, headers=headers, params=params, timeout=10)

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response code: {data.get('code')}")
            print(f"Message: {data.get('msg', 'N/A')}")

            if data.get('code') == '0' and data.get('data'):
                records = data['data']
                print(f"✅ SUCCESS - Got {len(records)} records")
                if records:
                    print(f"   Sample: {records[0]}")
            else:
                print(f"❌ FAILED - {data}")
        else:
            print(f"❌ HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")

    except Exception as e:
        print(f"❌ ERROR: {e}")

print(f"\n{'=' * 80}")
print("TEST COMPLETE")
print("=" * 80)
