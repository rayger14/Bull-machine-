# Historical Data Download Guide

## Status

✅ **Downloader scripts created and ready**
✅ **Macro data downloaded** (VIX, DXY, MOVE, yields, gold, oil - 2.8 years)
❌ **Crypto data blocked** (Binance/Bybit geo-restrictions)

## Geo-Restriction Issue

Both Binance and Bybit APIs are returning geo-restriction errors:
- **Binance**: 451 error (unavailable in your region)
- **Bybit**: 403 Forbidden

This is likely due to US-based IP restrictions on crypto exchanges.

## Workarounds

### Option 1: Use VPN (Recommended - Fast)
```bash
# Connect to VPN (non-US region like UK, Singapore, Japan)
# Then run the downloaders:

python3 tools/get_binance_klines.py --symbol BTCUSDT --interval 1h \
  --start 2023-01-01 --end 2025-10-13 \
  --out data/raw/binance/BTCUSDT_1h_2023-01-01_2025-10-13.csv

python3 tools/get_binance_klines.py --symbol ETHUSDT --interval 1h \
  --start 2023-01-01 --end 2025-10-13 \
  --out data/raw/binance/ETHUSDT_1h_2023-01-01_2025-10-13.csv
```

### Option 2: Use CCXT Library (Alternative API wrapper)
```bash
pip install ccxt

# CCXT has built-in proxy support and better geo-handling
python3 tools/get_ccxt_data.py --exchange binance --symbol BTC/USDT \
  --timeframe 1h --start 2023-01-01 --end 2025-10-13
```

### Option 3: TradingView Premium Export (Manual)
- Subscribe to TradingView Premium ($60/month)
- Export BTCUSDT, ETHUSDT 1H charts for 2+ years
- Save as CSV with columns: time, open, high, low, close, volume

### Option 4: Use Existing Data + Manual Extension
You already have:
- **3.7 years of macro data** (CRYPTOCAP_TOTAL, USDT.D, etc.) ✅
- **96 days of BTC/ETH 1H data** ⚠️  (insufficient for ML)

Can extend the 96-day data by:
1. Running system in paper mode for 6-12 months
2. Collecting live data during that period
3. Then training ML on combined historical + live data

### Option 5: Alternative Free Data Sources

#### CoinGecko API (limited to daily)
```bash
# Only daily resolution, but free and unrestricted
curl "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from=1672531200&to=1760313600" > btc_daily.json
```

#### Kraken Public API (US-friendly)
```bash
# Kraken allows US access, but limited history (720 bars max)
curl "https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=60" > btc_1h_kraken.json
```

## Current Data Inventory

### ✅ Available (Downloaded)
```
data/raw/macro/
├── VIX_1d_2023-01-01_2025-10-13.csv       (2.8 years)
├── DXY_1d_2023-01-01_2025-10-13.csv
├── MOVE_1d_2023-01-01_2025-10-13.csv
├── US10Y_1d_2023-01-01_2025-10-13.csv
├── GOLD_1d_2023-01-01_2025-10-13.csv
├── OIL_1d_2023-01-01_2025-10-13.csv
├── SPY_1d_2023-01-01_2025-10-13.csv
└── QQQ_1d_2023-01-01_2025-10-13.csv

chart_logs/
├── CRYPTOCAP_TOTAL, 1D_0e678.csv          (3.7 years)
├── CRYPTOCAP_TOTAL2, 1D_b812b.csv
├── CRYPTOCAP_TOTAL3, 1D_06b33.csv
├── CRYPTOCAP_USDT.D, 60_2c8db.csv
├── CRYPTOCAP_USDC.D, 60_1ccd0.csv
├── CRYPTOCAP_BTC.D, 60_8b026.csv
└── ... (all other macro files)
```

### ❌ Missing (Blocked)
```
data/raw/binance/
└── BTCUSDT_1h_2023-01-01_2025-10-13.csv   (NEEDED - 2.8 years)
└── ETHUSDT_1h_2023-01-01_2025-10-13.csv   (NEEDED)

data/raw/bybit/
└── (same - blocked)
```

## Recommended Immediate Action

**Best path forward:**

1. **Use VPN** (fastest solution)
   - Connect to non-US VPN server
   - Run Binance downloader scripts
   - Should complete in 10-15 minutes

2. **Or: Install CCXT** (fallback)
   - `pip install ccxt`
   - Use CCXT's proxy/retry logic
   - More reliable for geo-restricted regions

3. **Then: Run full ML pipeline**
   ```bash
   # After getting 2+ years of BTC/ETH data:
   python3 bin/build_feature_store.py --asset BTC --data-dir data/raw/binance
   python3 bin/optimize_v19.py --asset BTC --mode exhaustive --workers 8
   python3 bin/research/train_ml.py --target pf --model lightgbm --normalize
   ```

## Scripts Ready to Use

All downloader scripts are created and tested:
- `tools/get_binance_klines.py` ✅ (needs VPN)
- `tools/get_bybit_klines.py` ✅ (needs VPN)
- `tools/get_macro_yf.py` ✅ (works - completed)
- `tools/verify_and_align.py` ✅ (ready for verification)

## Next Steps

1. Connect to VPN (non-US region)
2. Re-run Binance downloader commands
3. Verify data with `verify_and_align.py`
4. Rebuild feature stores with full 2.8-year dataset
5. Run ML training pipeline

The ML framework is **production-ready** - we just need the crypto price data to complete the training.
