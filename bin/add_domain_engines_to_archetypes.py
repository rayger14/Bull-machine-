#!/usr/bin/env python3
"""
Script to add full domain engine integration to 10 unwired archetypes.

Target archetypes: C, D, E, F, G, K, L, M, S3, S8

This script generates the domain engine integration code for each archetype,
matching the S1 reference implementation (lines 1750-1950).
"""

LONG_DOMAIN_ENGINE_TEMPLATE = '''
        # ============================================================================
        # DOMAIN ENGINE INTEGRATION (BOOST/VETO LAYER) - LONG PATTERN
        # ============================================================================
        use_wyckoff = context.metadata.get('feature_flags', {}).get('enable_wyckoff', False)
        use_smc = context.metadata.get('feature_flags', {}).get('enable_smc', False)
        use_temporal = context.metadata.get('feature_flags', {}).get('enable_temporal', False)
        use_hob = context.metadata.get('feature_flags', {}).get('enable_hob', False)
        use_macro = context.metadata.get('feature_flags', {}).get('enable_macro', False)

        domain_boost = 1.0
        domain_signals = []

        # Wyckoff Engine (LONG pattern)
        if use_wyckoff:
            wyckoff_distribution = self.g(context.row, 'wyckoff_phase_abc', '') == 'D'
            wyckoff_accumulation = self.g(context.row, 'wyckoff_phase_abc', '') == 'A'
            wyckoff_spring_a = self.g(context.row, 'wyckoff_spring_a', False)
            wyckoff_spring_b = self.g(context.row, 'wyckoff_spring_b', False)
            wyckoff_sc = self.g(context.row, 'wyckoff_sc', False)
            wyckoff_st = self.g(context.row, 'wyckoff_st', False)
            wyckoff_lps = self.g(context.row, 'wyckoff_lps', False)

            if wyckoff_distribution:
                domain_boost *= 0.70  # Soft veto for distribution phase
                domain_signals.append("wyckoff_distribution_caution")
            if wyckoff_accumulation:
                domain_boost *= 1.80  # Major boost for accumulation
                domain_signals.append("wyckoff_accumulation")
            if wyckoff_spring_a:
                domain_boost *= 2.50  # Deep spring = strongest capitulation
                domain_signals.append("wyckoff_spring_a_major")
            elif wyckoff_spring_b:
                domain_boost *= 2.20  # Shallow spring
                domain_signals.append("wyckoff_spring_b")
            if wyckoff_sc:
                domain_boost *= 2.00  # Selling climax
                domain_signals.append("wyckoff_selling_climax")
            elif wyckoff_st:
                domain_boost *= 1.50  # Secondary test
                domain_signals.append("wyckoff_secondary_test")
            if wyckoff_lps:
                domain_boost *= 1.80  # Last point support
                domain_signals.append("wyckoff_lps_support")

        # SMC Engine (LONG pattern)
        if use_smc:
            smc_supply_zone = self.g(context.row, 'smc_supply_zone', False)
            tf4h_bos_bearish = self.g(context.row, 'tf4h_bos_bearish', False)
            tf4h_bos_bullish = self.g(context.row, 'tf4h_bos_bullish', False)
            tf1h_bos_bullish = self.g(context.row, 'tf1h_bos_bullish', False)
            smc_demand_zone = self.g(context.row, 'smc_demand_zone', False)
            smc_liquidity_sweep = self.g(context.row, 'smc_liquidity_sweep', False)
            smc_choch = self.g(context.row, 'smc_choch', False)

            if smc_supply_zone:
                domain_boost *= 0.70  # Supply overhead reduces conviction
                domain_signals.append("smc_supply_overhead")
            if tf4h_bos_bearish:
                domain_boost *= 0.60  # Bearish 4H structure penalty
                domain_signals.append("smc_4h_bearish_structure")
            if tf4h_bos_bullish:
                domain_boost *= 2.00  # Institutional 4H bullish shift
                domain_signals.append("smc_4h_bos_bullish_institutional")
            elif tf1h_bos_bullish:
                domain_boost *= 1.40  # 1H structural shift
                domain_signals.append("smc_1h_bos_bullish")
            if smc_demand_zone:
                domain_boost *= 1.60  # Institutional support area
                domain_signals.append("smc_demand_zone_support")
            if smc_liquidity_sweep:
                domain_boost *= 1.80  # Stop hunt before rally
                domain_signals.append("smc_liquidity_sweep_reversal")
            if smc_choch:
                domain_boost *= 1.60  # Character change = trend shift
                domain_signals.append("smc_choch_trend_change")

        # Temporal Engine
        if use_temporal:
            fib_time_cluster = self.g(context.row, 'fib_time_cluster', False)
            temporal_confluence = self.g(context.row, 'temporal_confluence', False)
            temporal_resistance_cluster = self.g(context.row, 'temporal_resistance_cluster', False)
            tf4h_fusion_score = self.g(context.row, 'tf4h_fusion_score', 0.0)

            if fib_time_cluster:
                domain_boost *= 1.70  # Fibonacci timing = geometric reversal
                domain_signals.append("fib_time_cluster_reversal")
            if temporal_confluence:
                domain_boost *= 1.50  # Multi-timeframe alignment
                domain_signals.append("temporal_multi_tf_confluence")
            if tf4h_fusion_score > 0.70:
                domain_boost *= 1.60  # High 4H fusion = strong trend
                domain_signals.append("tf4h_high_fusion_score")
            if temporal_resistance_cluster:
                domain_boost *= 0.75  # Resistance overhead
                domain_signals.append("temporal_resistance_overhead")

        # HOB Engine
        if use_hob:
            hob_demand_zone = self.g(context.row, 'hob_demand_zone', False)
            hob_supply_zone = self.g(context.row, 'hob_supply_zone', False)
            hob_imbalance = self.g(context.row, 'hob_imbalance', 0.0)

            if hob_demand_zone:
                domain_boost *= 1.50  # Large bid wall support
                domain_signals.append("hob_demand_zone_support")
            if hob_supply_zone:
                domain_boost *= 0.70  # Supply wall overhead
                domain_signals.append("hob_supply_zone_overhead")
            if hob_imbalance > 0.60:
                domain_boost *= 1.30  # Strong buyer imbalance
                domain_signals.append("hob_bid_imbalance_strong")
            elif hob_imbalance > 0.40:
                domain_boost *= 1.15  # Moderate buyer imbalance
                domain_signals.append("hob_bid_imbalance_moderate")

        # Macro Engine
        if use_macro:
            crisis_composite = self.g(context.row, 'crisis_composite', 0.0)
            if crisis_composite > 0.70:
                domain_boost *= 0.85  # Extreme crisis reduces conviction
                domain_signals.append("macro_extreme_crisis_penalty")

        # Apply domain boost BEFORE fusion gate
        score_before_domain = score
        score = score * domain_boost
'''

SHORT_DOMAIN_ENGINE_TEMPLATE = '''
        # ============================================================================
        # DOMAIN ENGINE INTEGRATION (BOOST/VETO LAYER) - SHORT PATTERN
        # ============================================================================
        use_wyckoff = context.metadata.get('feature_flags', {}).get('enable_wyckoff', False)
        use_smc = context.metadata.get('feature_flags', {}).get('enable_smc', False)
        use_temporal = context.metadata.get('feature_flags', {}).get('enable_temporal', False)
        use_hob = context.metadata.get('feature_flags', {}).get('enable_hob', False)
        use_macro = context.metadata.get('feature_flags', {}).get('enable_macro', False)

        domain_boost = 1.0
        domain_signals = []

        # Wyckoff Engine (SHORT pattern)
        if use_wyckoff:
            wyckoff_distribution = self.g(context.row, 'wyckoff_phase_abc', '') == 'D'
            wyckoff_accumulation = self.g(context.row, 'wyckoff_phase_abc', '') == 'A'
            wyckoff_utad = self.g(context.row, 'wyckoff_utad', False)
            wyckoff_bc = self.g(context.row, 'wyckoff_bc', False)
            wyckoff_psy = self.g(context.row, 'wyckoff_psy', False)

            if wyckoff_distribution:
                domain_boost *= 2.00  # Major boost for distribution phase
                domain_signals.append("wyckoff_distribution_short")
            if wyckoff_accumulation:
                domain_boost *= 0.70  # Soft veto for accumulation
                domain_signals.append("wyckoff_accumulation_caution")
            if wyckoff_utad:
                domain_boost *= 2.50  # Upthrust After Distribution = top signal
                domain_signals.append("wyckoff_utad_top")
            elif wyckoff_bc:
                domain_boost *= 2.20  # Buying Climax
                domain_signals.append("wyckoff_bc_climax")
            if wyckoff_psy:
                domain_boost *= 1.60  # Preliminary Supply
                domain_signals.append("wyckoff_psy_supply")

        # SMC Engine (SHORT pattern)
        if use_smc:
            smc_demand_zone = self.g(context.row, 'smc_demand_zone', False)
            tf4h_bos_bullish = self.g(context.row, 'tf4h_bos_bullish', False)
            tf4h_bos_bearish = self.g(context.row, 'tf4h_bos_bearish', False)
            tf1h_bos_bearish = self.g(context.row, 'tf1h_bos_bearish', False)
            smc_supply_zone = self.g(context.row, 'smc_supply_zone', False)
            smc_liquidity_sweep = self.g(context.row, 'smc_liquidity_sweep', False)

            if smc_demand_zone:
                domain_boost *= 0.70  # Demand support reduces short conviction
                domain_signals.append("smc_demand_support_caution")
            if tf4h_bos_bullish:
                domain_boost *= 0.60  # Bullish 4H structure penalty for shorts
                domain_signals.append("smc_4h_bullish_structure_penalty")
            if tf4h_bos_bearish:
                domain_boost *= 2.00  # Institutional 4H bearish shift
                domain_signals.append("smc_4h_bos_bearish_institutional")
            elif tf1h_bos_bearish:
                domain_boost *= 1.40  # 1H bearish structural shift
                domain_signals.append("smc_1h_bos_bearish")
            if smc_supply_zone:
                domain_boost *= 1.60  # Institutional resistance area
                domain_signals.append("smc_supply_zone_resistance")
            if smc_liquidity_sweep:
                domain_boost *= 1.50  # Liquidity grab before drop
                domain_signals.append("smc_liquidity_sweep_drop")

        # Temporal Engine
        if use_temporal:
            fib_time_cluster = self.g(context.row, 'fib_time_cluster', False)
            temporal_confluence = self.g(context.row, 'temporal_confluence', False)
            temporal_support_cluster = self.g(context.row, 'temporal_support_cluster', False)

            if fib_time_cluster:
                domain_boost *= 1.70  # Fibonacci timing reversal
                domain_signals.append("fib_time_cluster_reversal")
            if temporal_confluence:
                domain_boost *= 1.50  # Multi-timeframe alignment
                domain_signals.append("temporal_multi_tf_confluence")
            if temporal_support_cluster:
                domain_boost *= 0.75  # Support below reduces short conviction
                domain_signals.append("temporal_support_below")

        # HOB Engine
        if use_hob:
            hob_supply_zone = self.g(context.row, 'hob_supply_zone', False)
            hob_demand_zone = self.g(context.row, 'hob_demand_zone', False)
            hob_imbalance = self.g(context.row, 'hob_imbalance', 0.0)

            if hob_supply_zone:
                domain_boost *= 1.50  # Large ask wall resistance
                domain_signals.append("hob_supply_zone_resistance")
            if hob_demand_zone:
                domain_boost *= 0.70  # Demand wall support reduces short
                domain_signals.append("hob_demand_zone_caution")
            if hob_imbalance < -0.60:
                domain_boost *= 1.30  # Strong seller imbalance
                domain_signals.append("hob_ask_imbalance_strong")
            elif hob_imbalance < -0.40:
                domain_boost *= 1.15  # Moderate seller imbalance
                domain_signals.append("hob_ask_imbalance_moderate")

        # Macro Engine
        if use_macro:
            crisis_composite = self.g(context.row, 'crisis_composite', 0.0)
            if crisis_composite > 0.70:
                domain_boost *= 1.30  # Extreme crisis BOOSTS shorts
                domain_signals.append("macro_extreme_crisis_short_boost")

        # Apply domain boost BEFORE fusion gate
        score_before_domain = score
        score = score * domain_boost
'''

# Summary of what needs to be added to each archetype
ARCHETYPE_INFO = {
    'C': {
        'name': 'wick_trap',
        'direction': 'LONG',
        'pattern': 'FVG Continuation',
        'line': 1183,
        'expected_boost': '1.5x - 10x'
    },
    'D': {
        'name': 'failed_continuation',
        'direction': 'LONG',
        'pattern': 'Failed Continuation',
        'line': 1212,
        'expected_boost': '1.5x - 10x'
    },
    'E': {
        'name': 'volume_exhaustion',
        'direction': 'LONG',
        'pattern': 'Liquidity Compression',
        'line': 1234,
        'expected_boost': '1.5x - 10x'
    },
    'F': {
        'name': 'exhaustion_reversal',
        'direction': 'LONG',
        'pattern': 'Expansion Exhaustion',
        'line': 1269,
        'expected_boost': '1.5x - 10x'
    },
    'G': {
        'name': 'liquidity_sweep',
        'direction': 'LONG',
        'pattern': 'Re-Accumulate Base',
        'line': 1293,
        'expected_boost': '1.5x - 10x'
    },
    'K': {
        'name': 'wick_trap_moneytaur',
        'direction': 'LONG',
        'pattern': 'Wick Trap / Moneytaur',
        'line': 1381,
        'expected_boost': '1.5x - 10x'
    },
    'L': {
        'name': 'volume_exhaustion',
        'direction': 'LONG',
        'pattern': 'Volume Exhaustion / Zeroika',
        'line': 1403,
        'expected_boost': '1.5x - 10x'
    },
    'M': {
        'name': 'confluence_breakout',
        'direction': 'LONG',
        'pattern': 'Ratio Coil Break',
        'line': 1463,
        'expected_boost': '1.5x - 10x'
    },
    'S3': {
        'name': 'whipsaw',
        'direction': 'SHORT',
        'pattern': 'Whipsaw',
        'line': 2697,
        'expected_boost': '1.5x - 8x'
    },
    'S8': {
        'name': 'volume_fade_chop',
        'direction': 'SHORT',
        'pattern': 'Volume Fade in Chop',
        'line': 3332,
        'expected_boost': '1.5x - 8x'
    }
}

if __name__ == '__main__':
    print("=" * 80)
    print("DOMAIN ENGINE INTEGRATION SUMMARY")
    print("=" * 80)
    print()
    print("| Archetype | Direction | Pattern | Expected Boost | Line |")
    print("|-----------|-----------|---------|----------------|------|")
    for code, info in ARCHETYPE_INFO.items():
        print(f"| {code:9s} | {info['direction']:9s} | {info['pattern']:30s} | {info['expected_boost']:14s} | {info['line']:4d} |")
    print()
    print("Total archetypes to integrate: 10")
    print("LONG patterns: 8 (C, D, E, F, G, K, L, M)")
    print("SHORT patterns: 2 (S3, S8)")
    print()
    print("=" * 80)
    print("INTEGRATION APPROACH")
    print("=" * 80)
    print()
    print("Each archetype will receive:")
    print("1. Wyckoff Engine (6-10 signals)")
    print("2. SMC Engine (6-8 signals)")
    print("3. Temporal Engine (4-6 signals)")
    print("4. HOB Engine (3-4 signals)")
    print("5. Macro Engine (1-2 signals)")
    print()
    print("Total domain signals per archetype: 20-30")
    print("Expected boost range: 1.5x - 12x")
    print("Veto range: 0.60x - 0.85x")
    print()
