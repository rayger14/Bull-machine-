#!/usr/bin/env python3
"""
Test CoinGlass v4 API Endpoints for Historical Funding Rate Data
"""

import requests
import json
from datetime import datetime

print("\n" + "="*80)
print("COINGLASS V4 API - FUNDING RATE HISTORY FETCH")
print("="*80)

# Endpoint 1: funding-rate/history
url1 = "https://open-api-v4.coinglass.com/api/futures/funding-rate/history"

print(f"\n📡 Testing Endpoint 1: funding-rate/history")
print(f"URL: {url1}")

# Try different parameter combinations
test_params = [
    {"symbol": "BTC", "exchange": "Binance"},
    {"symbol": "BTCUSDT", "exchange": "Binance"},
    {"symbol": "BTC"},
    {"exchange": "Binance"},
    {},  # No params to see API response
]

success = False
for i, params in enumerate(test_params, 1):
    print(f"\n  Test {i}: params={params}")
    try:
        response = requests.get(url1, params=params, timeout=10)
        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ SUCCESS!")
            print(f"  Response keys: {list(data.keys())}")
            print(f"  Response preview:")
            print(json.dumps(data, indent=2)[:1000])
            success = True
            break
        else:
            print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

if not success:
    print("\n⚠️  Endpoint 1 failed all tests")

print("\n" + "="*80)
print(f"📡 Testing Endpoint 2: funding-rate/oi-weight-history")
url2 = "https://open-api-v4.coinglass.com/api/futures/funding-rate/oi-weight-history"
print(f"URL: {url2}")

success2 = False
for i, params in enumerate(test_params, 1):
    print(f"\n  Test {i}: params={params}")
    try:
        response = requests.get(url2, params=params, timeout=10)
        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ SUCCESS!")
            print(f"  Response keys: {list(data.keys())}")
            print(f"  Response preview:")
            print(json.dumps(data, indent=2)[:1000])
            success2 = True
            break
        else:
            print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

if not success2:
    print("\n⚠️  Endpoint 2 failed all tests")

print("\n" + "="*80)
if success or success2:
    print("✅ At least one endpoint is accessible!")
else:
    print("❌ Both endpoints failed - may require authentication or different params")
print("="*80)
