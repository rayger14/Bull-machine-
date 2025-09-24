#!/usr/bin/env python3
"""
Signal Flow Diagnostic Harness
Prints one line per bar showing exactly why no signals are generated.
"""

import sys
import pandas as pd
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bull_machine.app.main_v13 import (
    load_csv_to_series, resample_to_timeframes, build_composite_range,
    resolve_bias, compute_eq_magnet, check_nested_confluence
)
from bull_machine.core.sync import decide_mtf_entry
from bull_machine.core.utils import extract_key_levels
from bull_machine.modules.wyckoff.advanced import AdvancedWyckoffAnalyzer
from bull_machine.modules.liquidity.advanced import AdvancedLiquidityAnalyzer
from bull_machine.modules.structure.advanced import AdvancedStructureAnalyzer
from bull_machine.modules.momentum.advanced import AdvancedMomentumAnalyzer
from bull_machine.modules.volume.advanced import AdvancedVolumeAnalyzer
from bull_machine.modules.context.advanced import AdvancedContextAnalyzer
from bull_machine.modules.fusion.diagnostic import DiagnosticFusionEngine
from bull_machine.config.loader import load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def validate_dataframe(df: pd.DataFrame, name: str):
    """Validate DataFrame sanity."""
    required_cols = {'open', 'high', 'low', 'close'}

    print(f"\n[VALIDATION] {name}:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Index dtype: {df.index.dtype}")

    # Check required columns
    missing_cols = required_cols - set(df.columns.str.lower())
    if missing_cols:
        print(f"  ❌ Missing columns: {missing_cols}")
        return False

    # Check for NaN values
    nan_counts = df[['open', 'high', 'low', 'close']].isna().sum()
    print(f"  NaN counts: {dict(nan_counts)}")

    # Check data range
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")

    return True

def validate_series(series, name: str):
    """Validate Series object."""
    print(f"\n[VALIDATION] {name} Series:")
    print(f"  Bars count: {len(series.bars)}")
    print(f"  Symbol: {series.symbol}")
    print(f"  Timeframe: {series.timeframe}")

    if len(series.bars) >= 5:
        print(f"  Last 5 bars:")
        for i, bar in enumerate(series.bars[-5:]):
            print(f"    [{i}] O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f} V:{bar.volume:.0f}")

    return len(series.bars) >= 200

def validate_analyzer_output(name: str, obj):
    """Validate analyzer output."""
    if obj is None:
        print(f"  ❌ {name}: returned None")
        return False

    issues = []

    # Check common fields
    for field in ['score', 'confidence', 'quality', 'bias']:
        if hasattr(obj, field):
            val = getattr(obj, field)
            if val is None:
                issues.append(f"{field}=None")
            elif isinstance(val, (int, float)) and not (val >= 0):
                issues.append(f"{field}={val} (invalid)")

    # Check dict fields for structure/momentum/volume/context
    if isinstance(obj, dict):
        for field in ['score', 'quality', 'bias']:
            if field in obj:
                val = obj[field]
                if val is None:
                    issues.append(f"{field}=None")
                elif isinstance(val, (int, float)) and not (val >= 0):
                    issues.append(f"{field}={val} (invalid)")

    if issues:
        print(f"  ❌ {name}: {', '.join(issues)}")
        return False
    else:
        print(f"  ✅ {name}: valid")
        return True

def format_analyzer_status(name: str, obj) -> str:
    """Format analyzer status for one-line output."""
    if obj is None:
        return f"{name}(NONE)"

    if isinstance(obj, dict):
        score = obj.get('score', 0)
        quality = obj.get('quality', 0)
        bias = obj.get('bias', 'neutral')
        return f"{name}(q={quality:.2f} s={score:.2f} b={bias[:1]})"
    else:
        # Object with attributes
        score = getattr(obj, 'score', getattr(obj, 'confidence', 0))
        quality = getattr(obj, 'quality', 0.5)
        bias = getattr(obj, 'bias', getattr(obj, 'pressure', 'neutral'))
        if bias == 'bullish':
            bias = 'long'
        elif bias == 'bearish':
            bias = 'short'
        return f"{name}(q={quality:.2f} s={score:.2f} b={bias[:1]})"

def diagnose_chart_file(csv_path: str, max_bars: int = 100):
    """Diagnose signal flow for a chart file."""
    print(f"\n{'='*80}")
    print(f"DIAGNOSING: {csv_path}")
    print(f"{'='*80}")

    # 1. Load and validate raw data
    try:
        series_ltf = load_csv_to_series(csv_path)
        if not validate_series(series_ltf, "LTF"):
            print("❌ LTF series validation failed")
            return
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return

    # 2. Create DataFrame for resampling
    df_data = []
    for bar in series_ltf.bars:
        df_data.append({
            "timestamp": bar.ts,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        })

    df = pd.DataFrame(df_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('timestamp')

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    if not validate_dataframe(df, "Base DataFrame"):
        return

    # 3. Resample to multiple timeframes
    try:
        data_dict = resample_to_timeframes(df, htf="1D", mtf="4H", ltf="1H")

        for tf_name, tf_df in data_dict.items():
            if not validate_dataframe(tf_df, f"{tf_name.upper()} TF"):
                return
    except Exception as e:
        print(f"❌ Resampling failed: {e}")
        return

    # 4. Initialize analyzers
    config = load_config("bull_machine/configs/diagnostic_v14_config.json")

    wyckoff_analyzer = AdvancedWyckoffAnalyzer(config)
    liquidity_analyzer = AdvancedLiquidityAnalyzer(config)
    structure_analyzer = AdvancedStructureAnalyzer(config)
    momentum_analyzer = AdvancedMomentumAnalyzer(config)
    volume_analyzer = AdvancedVolumeAnalyzer(config)
    context_analyzer = AdvancedContextAnalyzer(config)
    fusion_engine = DiagnosticFusionEngine(config)

    # 5. Build MTF contexts
    htf_range = build_composite_range(data_dict["htf"], "1D")
    htf_bias = resolve_bias(data_dict["htf"], "1D", two_bar=True)
    mtf_bias = resolve_bias(data_dict["mtf"], "4H", two_bar=True)

    print(f"\n[MTF CONTEXT]")
    print(f"  HTF bias: {htf_bias}")
    print(f"  MTF bias: {mtf_bias}")
    print(f"  HTF range: {htf_range}")

    # 6. Analyze last N bars
    print(f"\n[SIGNAL FLOW] Last {max_bars} bars:")
    print("TS               | Analyzers                                                    | Fusion   | Decision")
    print("-" * 120)

    start_idx = max(0, len(series_ltf.bars) - max_bars)

    for i in range(start_idx, len(series_ltf.bars)):
        bar = series_ltf.bars[i]
        bar_time = pd.to_datetime(bar.ts, unit='s').strftime('%Y-%m-%d %H:%M')

        # Create mini-series for this bar (last 200 bars for context)
        context_start = max(0, i - 199)
        mini_series = type(series_ltf)(
            bars=series_ltf.bars[context_start:i+1],
            timeframe=series_ltf.timeframe,
            symbol=series_ltf.symbol
        )

        if len(mini_series.bars) < 50:
            print(f"{bar_time} | WARMUP (only {len(mini_series.bars)} bars)")
            continue

        # Run analyzers
        try:
            wyckoff_result = wyckoff_analyzer.analyze(mini_series)
            liquidity_result = liquidity_analyzer.analyze(mini_series, wyckoff_result.bias)
            structure_result = structure_analyzer.analyze(mini_series)
            momentum_result = momentum_analyzer.analyze(mini_series)
            volume_result = volume_analyzer.analyze(mini_series)
            context_result = context_analyzer.analyze(mini_series)

            # Validate outputs
            all_valid = True
            for name, obj in [('wyckoff', wyckoff_result), ('liquidity', liquidity_result),
                             ('structure', structure_result), ('momentum', momentum_result),
                             ('volume', volume_result), ('context', context_result)]:
                if not validate_analyzer_output(name, obj):
                    all_valid = False

            if not all_valid:
                print(f"{bar_time} | ANALYZER_ERROR")
                continue

            # MTF sync
            ltf_bias = wyckoff_result.bias
            ltf_levels = extract_key_levels(liquidity_result)
            nested_ok = check_nested_confluence(htf_range, ltf_levels)
            eq_magnet = compute_eq_magnet(htf_range, bar.close, config.get('mtf', {}))

            sync_report = decide_mtf_entry(
                htf_bias, mtf_bias, ltf_bias,
                nested_ok, eq_magnet,
                config.get('mtf', {})
            )

            # Fusion
            signal = fusion_engine.fuse_with_mtf({
                "wyckoff": wyckoff_result,
                "liquidity": liquidity_result,
                "structure": structure_result,
                "momentum": momentum_result,
                "volume": volume_result,
                "context": context_result
            }, sync_report)

            # Format output
            analyzers_str = " ".join([
                format_analyzer_status("wy", wyckoff_result),
                format_analyzer_status("liq", liquidity_result),
                format_analyzer_status("str", structure_result),
                format_analyzer_status("mom", momentum_result),
                format_analyzer_status("vol", volume_result),
                format_analyzer_status("ctx", context_result)
            ])

            if signal:
                fusion_str = f"✅ {signal.side} {signal.confidence:.3f}"
                decision_str = "SIGNAL"
            else:
                fusion_str = "❌ None"
                vetoes = []
                if eq_magnet:
                    vetoes.append("eq_magnet")
                if sync_report and sync_report.decision == 'veto':
                    vetoes.append("mtf_veto")
                decision_str = f"BLOCKED({','.join(vetoes) if vetoes else 'threshold'})"

            print(f"{bar_time} | {analyzers_str:<60} | {fusion_str:<8} | {decision_str}")

        except Exception as e:
            print(f"{bar_time} | ERROR: {e}")

def main():
    """Main diagnostic function."""
    # Test files from Chart Logs 2
    test_files = [
        "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 240_c2b76.csv",
        "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv"
    ]

    for csv_path in test_files:
        if Path(csv_path).exists():
            diagnose_chart_file(csv_path, max_bars=50)
            break
    else:
        print("❌ No test files found")

if __name__ == "__main__":
    main()