"""
Preflight Data Reality Check

Validates all required TradingView data files exist and contain real market data.
MUST PASS before running any backtests.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.io.tradingview_loader import load_tv, SYMBOL_MAP, RealDataRequiredError
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Run comprehensive data validation.

    Returns:
        bool: True if all checks pass, False otherwise
    """
    print("🔍 PREFLIGHT DATA REALITY CHECK")
    print("=" * 50)

    # Required data series for Bull Machine
    required = [
        "ETH_4H", "BTC_4H", "SOL_4H",  # Primary assets
        "DXY_1D", "US2Y_1D", "US10Y_1D",  # Macro core
        "GOLD_1D", "WTI_1D",  # Commodities
        "BTCD_1W", "USDTD_4H",  # Crypto dominance
        "ETHBTC_1D"  # Rotation
    ]

    # Optional but recommended
    optional = [
        "BTC_1D", "ETH_1D", "SOL_1D",  # Daily timeframes
        "TOTAL_4H", "TOTAL3_4H"  # Market cap indices
    ]

    missing = []
    flat = []
    loaded = []
    errors = []

    print(f"📋 Checking {len(required)} required series...")

    for symbol_key in required:
        try:
            print(f"  Loading {symbol_key}...")
            df = load_tv(symbol_key)

            # Validate data quality
            if len(df) == 0:
                errors.append((symbol_key, "Empty DataFrame"))
                continue

            # Check for flat/synthetic data
            close_std = df["close"].std()
            if close_std <= 1e-9:
                flat.append((symbol_key, f"std={close_std}"))
                continue

            # Check realistic volatility
            price_range = df["close"].max() - df["close"].min()
            mean_price = df["close"].mean()
            volatility_pct = (price_range / mean_price) * 100

            if volatility_pct < 0.1:  # Less than 0.1% total range
                flat.append((symbol_key, f"volatility={volatility_pct:.4f}%"))
                continue

            loaded.append((symbol_key, len(df), volatility_pct))
            print(f"    ✅ {len(df)} bars, volatility={volatility_pct:.2f}%")

        except RealDataRequiredError as e:
            missing.append((symbol_key, str(e)))
            print(f"    ❌ Missing: {e}")
        except Exception as e:
            errors.append((symbol_key, str(e)))
            print(f"    💥 Error: {e}")

    # Check optional series (non-blocking)
    print(f"\n📋 Checking {len(optional)} optional series...")
    optional_loaded = 0

    for symbol_key in optional:
        try:
            df = load_tv(symbol_key)
            if not df.empty and df["close"].std() > 1e-9:
                optional_loaded += 1
                print(f"  ✅ {symbol_key}: {len(df)} bars")
        except:
            print(f"  ⚠️  {symbol_key}: Not available")

    # Results summary
    print("\n" + "=" * 50)
    print("📊 PREFLIGHT RESULTS")
    print("=" * 50)

    success = len(missing) == 0 and len(flat) == 0 and len(errors) == 0

    if success:
        print("🎉 ALL CHECKS PASSED!")
        print(f"✅ {len(loaded)}/{len(required)} required series loaded")
        print(f"✅ {optional_loaded}/{len(optional)} optional series available")

        # Show data coverage
        if loaded:
            min_bars = min(bars for _, bars, _ in loaded)
            max_bars = max(bars for _, bars, _ in loaded)
            avg_volatility = sum(vol for _, _, vol in loaded) / len(loaded)

            print(f"\n📈 Data Quality Summary:")
            print(f"   • Bar range: {min_bars} - {max_bars}")
            print(f"   • Average volatility: {avg_volatility:.2f}%")
            print(f"   • All series have realistic variance ✅")

    else:
        print("❌ ISSUES FOUND - CANNOT PROCEED")

        if missing:
            print(f"\n🚨 Missing Files ({len(missing)}):")
            for symbol_key, error in missing:
                print(f"   • {symbol_key}: {error}")

        if flat:
            print(f"\n📊 Flat/Synthetic Data ({len(flat)}):")
            for symbol_key, reason in flat:
                print(f"   • {symbol_key}: {reason}")

        if errors:
            print(f"\n💥 Load Errors ({len(errors)}):")
            for symbol_key, error in errors:
                print(f"   • {symbol_key}: {error}")

        print(f"\n🔧 NEXT STEPS:")
        print(f"   1. Check chart_logs symlink: ls -la chart_logs")
        print(f"   2. Verify TradingView files exist with correct naming")
        print(f"   3. Pattern: 'PREFIX, TIMEFRAME_hash.csv'")
        print(f"   4. No synthetic/flat data allowed")

    # Available files debug info
    try:
        chart_logs = Path("chart_logs")
        if chart_logs.exists():
            csv_files = [f.name for f in chart_logs.iterdir() if f.suffix.lower() == '.csv']
            if csv_files:
                print(f"\n📁 Available CSV files ({len(csv_files)}):")
                for file in sorted(csv_files)[:15]:  # Show first 15
                    print(f"   • {file}")
                if len(csv_files) > 15:
                    print(f"   ... and {len(csv_files) - 15} more")
    except Exception as e:
        print(f"\n⚠️  Could not list chart_logs directory: {e}")

    print("\n" + "=" * 50)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)