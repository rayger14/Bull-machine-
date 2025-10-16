#!/bin/bash
# Download 2+ years of historical data for Bull Machine v1.8.6

set -e

START_DATE="2023-01-01"
END_DATE="2025-10-13"

echo "📥 Downloading Bull Machine Historical Data"
echo "Period: $START_DATE to $END_DATE (2.8 years)"
echo "========================================================================"

# Create output directory
mkdir -p chart_logs_binance

# Crypto pairs (1H, 4H, 1D)
echo ""
echo "🪙 Downloading BTC data..."
python3 bin/download_binance_data.py --symbol BTCUSDT --interval 1h --start $START_DATE --end $END_DATE --output chart_logs_binance
python3 bin/download_binance_data.py --symbol BTCUSDT --interval 4h --start $START_DATE --end $END_DATE --output chart_logs_binance
python3 bin/download_binance_data.py --symbol BTCUSDT --interval 1d --start $START_DATE --end $END_DATE --output chart_logs_binance

echo ""
echo "🪙 Downloading ETH data..."
python3 bin/download_binance_data.py --symbol ETHUSDT --interval 1h --start $START_DATE --end $END_DATE --output chart_logs_binance
python3 bin/download_binance_data.py --symbol ETHUSDT --interval 4h --start $START_DATE --end $END_DATE --output chart_logs_binance
python3 bin/download_binance_data.py --symbol ETHUSDT --interval 1d --start $START_DATE --end $END_DATE --output chart_logs_binance

echo ""
echo "🪙 Downloading SOL data..."
python3 bin/download_binance_data.py --symbol SOLUSDT --interval 1h --start $START_DATE --end $END_DATE --output chart_logs_binance
python3 bin/download_binance_data.py --symbol SOLUSDT --interval 4h --start $START_DATE --end $END_DATE --output chart_logs_binance
python3 bin/download_binance_data.py --symbol SOLUSDT --interval 1d --start $START_DATE --end $END_DATE --output chart_logs_binance

echo ""
echo "========================================================================"
echo "✅ Download complete!"
echo ""
echo "📊 Files saved to: chart_logs_binance/"
echo ""
echo "Next steps:"
echo "  1. Copy macro data from existing folders:"
echo "     cp '/Users/raymondghandchi/Downloads/Chart Logs 4/CRYPTOCAP_'*.csv chart_logs_binance/"
echo "     cp '/Users/raymondghandchi/Downloads/Chart Logs 4/TVC_'*.csv chart_logs_binance/"
echo "     cp '/Users/raymondghandchi/Downloads/Chart Logs 4/OANDA_'*.csv chart_logs_binance/"
echo ""
echo "  2. Rebuild feature stores:"
echo "     python3 bin/build_feature_store.py --asset BTC --data-dir chart_logs_binance"
echo "     python3 bin/build_feature_store.py --asset ETH --data-dir chart_logs_binance"
echo ""
echo "  3. Run grid optimization:"
echo "     python3 bin/optimize_v19.py --asset BTC --mode exhaustive --workers 8"
