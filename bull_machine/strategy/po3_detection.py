# PO3: Stack your confluences. The cleaner the displacement, the stronger the model.
import pandas as pd
import numpy as np
def detect_po3(df, irh, irl, vol_spike_threshold=1.4, reverse_bars=3):
    """Detect PO3: One-side sweep + aggressive break."""
    if len(df) < 2:
        return None

    # Calculate volume statistics with more flexible thresholds
    vol_mean = df['volume'].rolling(min(10, len(df))).mean().iloc[-1]
    if vol_mean == 0 or pd.isna(vol_mean):
        vol_mean = df['volume'].mean()

    # Look for volume spikes in recent bars (more lenient)
    recent_vols = df['volume'].tail(min(3, len(df)))
    vol_spike = any(vol > vol_spike_threshold * vol_mean for vol in recent_vols)

    # Look for sweep patterns across multiple bars
    recent_bars = df.tail(min(5, len(df)))

    # Check for sweeps in recent history
    sweep_low_detected = any(bar['low'] < irl for _, bar in recent_bars.iterrows())
    sweep_high_detected = any(bar['high'] > irh for _, bar in recent_bars.iterrows())

    # Current price levels
    current_high = df['high'].iloc[-1]
    current_low = df['low'].iloc[-1]
    current_close = df['close'].iloc[-1]

    # Prioritize patterns based on sequence: sweep then break
    # Pattern 1: High sweep followed by low break (reversal)
    if sweep_high_detected and vol_spike and current_close < irl:
        bojan_boost = 0.10 if has_bojan_high(df) else 0
        return {'po3_type': 'high_sweep_low_break', 'strength': 0.70 + bojan_boost}

    # Pattern 2: Low sweep followed by high break (reversal)
    if sweep_low_detected and vol_spike and current_close > irh:
        bojan_boost = 0.10 if has_bojan_high(df) else 0
        return {'po3_type': 'low_sweep_high_break', 'strength': 0.70 + bojan_boost}

    # Pattern 3: High sweep followed by high break (continuation)
    if sweep_high_detected and vol_spike and current_close > irh:
        bojan_boost = 0.10 if has_bojan_high(df) else 0
        return {'po3_type': 'high_sweep_high_break', 'strength': 0.70 + bojan_boost}

    # Pattern 4: Low sweep followed by low break (continuation)
    if sweep_low_detected and vol_spike and current_close < irl:
        bojan_boost = 0.10 if has_bojan_high(df) else 0
        return {'po3_type': 'low_sweep_low_break', 'strength': 0.70 + bojan_boost}

    # Pattern 5: Reversal patterns in range
    if sweep_low_detected and vol_spike and current_close > (irl + (irh - irl) * 0.6) and reverses(df, reverse_bars):
        bojan_boost = 0.10 if has_bojan_high(df) else 0
        return {'po3_type': 'low_sweep_high_break', 'strength': 0.70 + bojan_boost}

    if sweep_high_detected and vol_spike and current_close < (irl + (irh - irl) * 0.4) and reverses(df, reverse_bars):
        bojan_boost = 0.10 if has_bojan_high(df) else 0
        return {'po3_type': 'high_sweep_low_break', 'strength': 0.70 + bojan_boost}


    return None

def reverses(df, n_bars=3):
    """Check rapid reversal post-sweep (Crypto Chase style)."""
    if len(df) < 1:
        return False

    current_bar = df.iloc[-1]
    bar_range = current_bar['high'] - current_bar['low']

    if bar_range == 0:
        return False

    # Close >40% body from low indicates reversal strength (more lenient)
    close_position = (current_bar['close'] - current_bar['low']) / bar_range

    # Also check recent bars for reversal patterns
    if len(df) >= 2:
        prev_bar = df.iloc[-2]
        prev_range = prev_bar['high'] - prev_bar['low']
        if prev_range > 0:
            prev_close_position = (prev_bar['close'] - prev_bar['low']) / prev_range
            # Strong reversal if either current or previous bar shows reversal
            return bool(close_position > 0.4 or prev_close_position > 0.6)

    return bool(close_position > 0.4)

def has_bojan_high(df):
    """Check for Bojan high pattern (wick magnets)."""
    if len(df) < 1:
        return False

    recent_bars = df.tail(min(5, len(df)))
    for _, bar in recent_bars.iterrows():
        total_range = bar['high'] - bar['low']
        if total_range == 0:
            continue

        # Get open and close, handle missing open column
        if 'open' in bar:
            body_high = max(bar['open'], bar['close'])
            body_low = min(bar['open'], bar['close'])
        else:
            # If no open column, assume close is body
            body_high = bar['close']
            body_low = bar['close']

        upper_wick = bar['high'] - body_high
        lower_wick = body_low - bar['low']

        # Bojan high: upper wick >= 70% of total range
        if upper_wick / total_range >= 0.70:
            return True

    return False

def detect_po3_with_bojan_confluence(df, irh, irl, vol_spike_threshold=1.4, reverse_bars=3):
    """Enhanced PO3 detection with Bojan microstructure confluence"""
    # Get base PO3 result
    po3_result = detect_po3(df, irh, irl, vol_spike_threshold, reverse_bars)

    if not po3_result:
        return None

    # Check for Bojan confluence
    try:
        from bull_machine.modules.bojan.bojan import compute_bojan_score
        bojan_analysis = compute_bojan_score(df)

        # Enhance PO3 with Bojan signals
        bojan_score = bojan_analysis.get('bojan_score', 0.0)
        bojan_signals = bojan_analysis.get('signals', {})

        # Additional confluence checks
        confluence_boost = 0.0
        confluence_tags = []

        # Wick magnet at sweep zone
        if bojan_signals.get('wick_magnet', {}).get('is_magnet', False):
            confluence_boost += 0.05
            confluence_tags.append('bojan_wick_magnet')

        # Trap reset alignment with PO3 pattern
        trap_reset = bojan_signals.get('trap_reset', {})
        if trap_reset.get('is_trap_reset', False):
            confluence_boost += 0.10
            confluence_tags.append('bojan_trap_reset')

        # pHOB zones near IRH/IRL levels
        phob_zones = bojan_signals.get('phob_zones', {})
        if phob_zones.get('phob_detected', False):
            current_price = df['close'].iloc[-1]
            for zone in phob_zones.get('zones', []):
                zone_distance = min(abs(current_price - irh), abs(current_price - irl))
                price_range = irh - irl if irh > irl else 1
                if zone_distance / price_range < 0.02:  # Within 2% of key levels
                    confluence_boost += 0.08
                    confluence_tags.append('bojan_phob_confluence')
                    break

        # Fibonacci prime zone confluence
        fib_prime = bojan_signals.get('fib_prime', {})
        if fib_prime.get('in_prime_zone', False):
            confluence_boost += 0.12
            confluence_tags.append('bojan_fib_prime')

        # Apply confluence boost
        enhanced_strength = po3_result['strength'] + confluence_boost

        # Update result with Bojan enhancements
        po3_result['strength'] = min(enhanced_strength, 1.0)
        po3_result['bojan_confluence'] = confluence_boost > 0
        po3_result['bojan_score'] = bojan_score
        po3_result['confluence_tags'] = confluence_tags
        po3_result['bojan_signals'] = bojan_signals

        return po3_result

    except ImportError:
        # Fallback to original PO3 if Bojan module not available
        return po3_result