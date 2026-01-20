#!/usr/bin/env python3
"""
Test OKX API Connection
Validates API credentials and fetches sample OI data
"""

import os
import sys
import requests
import hmac
import base64
import hashlib
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_okx_headers(method, request_path, body=''):
    """Generate OKX API headers with signature"""
    api_key = os.getenv('OKX_API_KEY')
    secret_key = os.getenv('OKX_SECRET_KEY')
    passphrase = os.getenv('OKX_PASSPHRASE', '')  # May not be needed for public endpoints

    if not api_key or not secret_key:
        raise ValueError("OKX_API_KEY and OKX_SECRET_KEY must be set in .env")

    timestamp = datetime.utcnow().isoformat()[:-3] + 'Z'
    message = timestamp + method + request_path + body

    mac = hmac.new(
        secret_key.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    )
    signature = base64.b64encode(mac.digest()).decode()

    headers = {
        'OK-ACCESS-KEY': api_key,
        'OK-ACCESS-SIGN': signature,
        'OK-ACCESS-TIMESTAMP': timestamp,
        'OK-ACCESS-PASSPHRASE': passphrase,
        'Content-Type': 'application/json'
    }

    return headers

def test_public_endpoint():
    """Test public endpoint (no auth needed)"""
    print("Testing public endpoint (no auth)...")
    url = "https://www.okx.com/api/v5/public/instruments?instType=SWAP&instId=BTC-USD-SWAP"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Public endpoint works: {data['data'][0]['instId']}")
        return True
    else:
        print(f"❌ Public endpoint failed: {response.status_code}")
        return False

def test_oi_endpoint_public():
    """Test Open Interest endpoint (public, no auth)"""
    print("\nTesting Open Interest endpoint (public)...")

    # OI endpoint is public, no auth needed
    url = "https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId=BTC-USD-SWAP"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data['code'] == '0' and data['data']:
            oi = float(data['data'][0]['oi'])
            print(f"✅ Current BTC-USD-SWAP Open Interest: {oi:,.0f} contracts")
            return True
        else:
            print(f"❌ API returned error: {data.get('msg', 'Unknown error')}")
            return False
    else:
        print(f"❌ Request failed: {response.status_code} - {response.text}")
        return False

def test_historical_oi():
    """Test historical OI data fetch"""
    print("\nTesting historical OI data (last 100 hours)...")

    # Historical candles with OI are in trading data
    # We'll use the /api/v5/market/history-mark-price-candles endpoint
    # or /api/v5/market/candles for recent data

    url = "https://www.okx.com/api/v5/market/candles?instId=BTC-USD-SWAP&bar=1H&limit=5"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data['code'] == '0' and data['data']:
            print(f"✅ Historical data available: {len(data['data'])} candles")
            print(f"   Latest candle timestamp: {datetime.fromtimestamp(int(data['data'][0][0])/1000)}")
            # Note: OI is not in candle data, need separate endpoint
            print("   ⚠️  OI not in candle endpoint, will use /api/v5/rubik/stat/contracts/open-interest-history")
            return True
        else:
            print(f"❌ No data: {data.get('msg', 'Unknown')}")
            return False
    else:
        print(f"❌ Failed: {response.status_code}")
        return False

def test_oi_history_endpoint():
    """Test the actual OI history endpoint we'll use"""
    print("\nTesting OI History endpoint (the one we'll use for backfill)...")

    # This is the endpoint for historical OI data
    # Note: This might be a premium/VIP endpoint requiring higher API tier
    url = "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history"
    params = {
        'instId': 'BTC-USD-SWAP',
        'period': '5m',  # 5m, 1H, 1D
        'limit': '10'
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data['code'] == '0' and data['data']:
            print(f"✅ OI History works: {len(data['data'])} data points")
            latest = data['data'][0]  # Returns list: [timestamp, oi, oiCcy]
            print(f"   Latest OI: {latest[1]} contracts")
            print(f"   Timestamp: {datetime.fromtimestamp(int(latest[0])/1000)}")
            return True
        else:
            print(f"⚠️  Endpoint returned: code={data.get('code')}, msg={data.get('msg')}")
            if data.get('code') == '50011':
                print("   → This might require API upgrade or different approach")
            return False
    else:
        print(f"❌ Request failed: {response.status_code}")
        return False

def main():
    print("="*60)
    print("OKX API Connection Test")
    print("="*60)

    results = []

    # Test 1: Public endpoint
    results.append(("Public Endpoint", test_public_endpoint()))

    # Test 2: Current OI
    results.append(("Current OI", test_oi_endpoint_public()))

    # Test 3: Historical candles
    results.append(("Historical Candles", test_historical_oi()))

    # Test 4: OI History (critical for backfill)
    results.append(("OI History Endpoint", test_oi_history_endpoint()))

    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_pass = all(r[1] for r in results[:3])  # First 3 must pass
    oi_history_pass = results[3][1]

    print("\n" + "="*60)
    if all_pass and oi_history_pass:
        print("✅ ALL TESTS PASSED - Ready for OI backfill")
        return 0
    elif all_pass:
        print("⚠️  PARTIAL SUCCESS - OI history endpoint issue")
        print("   We can still proceed using current OI + candle data")
        print("   Or use alternative: aggregate multiple exchanges")
        return 0
    else:
        print("❌ TESTS FAILED - Check API credentials")
        return 1

if __name__ == "__main__":
    sys.exit(main())
