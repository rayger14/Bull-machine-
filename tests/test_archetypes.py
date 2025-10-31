#!/usr/bin/env python3
"""
PR#6A: Comprehensive Archetype System Tests

Tests all 11 archetypes (A-H + K, L, M) with synthetic data to ensure
proper detection logic and telemetry tracking.
"""

import pytest
import pandas as pd
import numpy as np
from engine.archetypes.logic import ArchetypeLogic
from engine.archetypes.telemetry import ArchetypeTelemetry


@pytest.fixture
def archetype_config():
    """Standard archetype configuration for testing."""
    return {
        'use_archetypes': True,
        'enable_A': True,
        'enable_B': True,
        'enable_C': True,
        'enable_D': True,
        'enable_E': True,
        'enable_F': True,
        'enable_G': True,
        'enable_H': True,
        'enable_K': True,
        'enable_L': True,
        'enable_M': True,
        'thresholds': {
            'min_liquidity': 0.30,
            'A': {'pti': 0.40, 'disp_atr': 0.80, 'fusion': 0.33},
            'B': {'boms_strength': 0.30, 'wyckoff': 0.35, 'fusion': 0.374},
            'C': {'disp_atr': 1.00, 'momentum': 0.45, 'fusion': 0.42, 'tf4h_fusion': 0.25},
            'D': {'rsi_max': 50, 'fusion': 0.42},
            'E': {'atr_pctile': 0.25, 'vol_cluster': 0.70, 'fusion': 0.35},
            'F': {'rsi_ext': 78, 'atr_pctile': 0.90, 'vol_z': 1.0, 'fusion': 0.38},
            'G': {'boms_strength': 0.40, 'liq': 0.40, 'fusion': 0.40},
            'H': {'adx': 25, 'liq_drop': 0.30, 'fusion': 0.35},
            'K': {'adx': 25, 'liq': 0.30, 'fusion': 0.36},
            'L': {'vol_z': 1.0, 'rsi_edge': 70, 'fusion': 0.38},
            'M': {'atr_pctile': 0.30, 'poc_dist': 0.50, 'boms_strength': 0.40, 'fusion': 0.35}
        }
    }


@pytest.fixture
def base_row():
    """Base row with minimal required features."""
    return pd.Series({
        'liquidity_score': 0.35,
        'wyckoff_score': 0.40,
        'momentum_score': 0.50,
        'atr_20': 100.0,
        'rsi_14': 50.0,
        'adx_14': 20.0,
        'tf1h_bos_bullish': False,
        'tf1h_bos_bearish': False,
        'tf1h_fvg_present': False,
        'tf4h_fvg_present': False,
        'tf1h_pti_trap_type': None,
        'pti_score': 0.0,
        'tf4h_boms_displacement': 50.0,
        'tf1d_boms_strength': 0.25,
        'tf4h_fusion_score': 0.30,
        'volume_zscore': 0.5,
        'atr_percentile': 0.50,
        'frvp_poc_distance': 0.10
    })


def test_archetype_logic_initialization(archetype_config):
    """Test ArchetypeLogic initialization."""
    logic = ArchetypeLogic(archetype_config)

    assert logic.use_archetypes is True
    assert logic.min_liquidity == 0.30
    assert all(logic.enabled[k] for k in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M'])


def test_archetype_a_trap_reversal(archetype_config, base_row):
    """Test Archetype A: Trap Reversal detection."""
    logic = ArchetypeLogic(archetype_config)

    # Create conditions for Archetype A
    row = base_row.copy()
    row['tf1h_pti_trap_type'] = 'spring'
    row['pti_score'] = 0.45
    row['tf4h_boms_displacement'] = 90.0  # > 0.80 * 100 ATR
    row['wyckoff_score'] = 0.40
    row['liquidity_score'] = 0.35
    row['momentum_score'] = 0.40

    df = pd.DataFrame([row])
    archetype, fusion, liq = logic.check_archetype(row, None, df, 0)

    assert archetype == 'trap_reversal'
    assert fusion >= 0.33
    assert liq >= 0.30


def test_archetype_b_order_block_retest(archetype_config, base_row):
    """Test Archetype B: Order Block Retest detection."""
    logic = ArchetypeLogic(archetype_config)

    row = base_row.copy()
    row['tf1h_bos_bullish'] = True
    row['tf1d_boms_strength'] = 0.35
    row['liquidity_score'] = 0.40
    row['wyckoff_score'] = 0.40
    row['momentum_score'] = 0.45

    df = pd.DataFrame([row])
    archetype, fusion, liq = logic.check_archetype(row, None, df, 0)

    assert archetype == 'order_block_retest'
    assert fusion >= 0.374
    assert liq >= 0.30


def test_archetype_c_fvg_continuation(archetype_config, base_row):
    """Test Archetype C: FVG Continuation detection."""
    logic = ArchetypeLogic(archetype_config)

    row = base_row.copy()
    row['tf4h_fvg_present'] = True
    row['tf4h_boms_displacement'] = 110.0  # > 1.0 * 100 ATR
    row['momentum_score'] = 0.50
    row['tf4h_fusion_score'] = 0.30
    row['liquidity_score'] = 0.35
    row['wyckoff_score'] = 0.40

    df = pd.DataFrame([row])
    archetype, fusion, liq = logic.check_archetype(row, None, df, 0)

    assert archetype == 'fvg_continuation'
    assert fusion >= 0.42
    assert liq >= 0.30


def test_archetype_d_failed_continuation(archetype_config, base_row):
    """Test Archetype D: Failed Continuation detection."""
    logic = ArchetypeLogic(archetype_config)

    row = base_row.copy()
    row['tf1h_fvg_present'] = True
    row['rsi_14'] = 45.0  # < 50 (bearish mean reversion)
    row['liquidity_score'] = 0.35
    row['wyckoff_score'] = 0.40
    row['momentum_score'] = 0.45

    df = pd.DataFrame([row])
    archetype, fusion, liq = logic.check_archetype(row, None, df, 0)

    assert archetype == 'failed_continuation'
    assert fusion >= 0.42
    assert liq >= 0.30


def test_liquidity_precheck(archetype_config, base_row):
    """Test that liquidity < min_threshold blocks all archetypes."""
    logic = ArchetypeLogic(archetype_config)

    row = base_row.copy()
    row['liquidity_score'] = 0.20  # Below 0.30 threshold
    row['tf1h_pti_trap_type'] = 'spring'
    row['pti_score'] = 0.45
    row['tf4h_boms_displacement'] = 90.0

    df = pd.DataFrame([row])
    archetype, fusion, liq = logic.check_archetype(row, None, df, 0)

    assert archetype is None
    assert liq == 0.20


def test_archetype_telemetry(archetype_config):
    """Test archetype telemetry tracking."""
    telemetry = ArchetypeTelemetry()

    # Simulate detections
    telemetry.check()
    telemetry.count('trap_reversal')

    telemetry.check()
    telemetry.count('trap_reversal')

    telemetry.check()
    telemetry.count('order_block_retest')

    telemetry.check()
    # No match

    summary = telemetry.summary()

    assert summary['total_checks'] == 4
    assert summary['total_matches'] == 3
    assert summary['match_rate_pct'] == 75.0
    assert summary['counts']['trap_reversal'] == 2
    assert summary['counts']['order_block_retest'] == 1
    assert summary['percentages']['trap_reversal'] == pytest.approx(66.67, rel=0.1)
    assert summary['percentages']['order_block_retest'] == pytest.approx(33.33, rel=0.1)


def test_archetype_priority(archetype_config, base_row):
    """Test that archetype priority ordering works correctly."""
    logic = ArchetypeLogic(archetype_config)

    # Create conditions that could match both A and B
    # Priority: A should win
    row = base_row.copy()
    row['tf1h_pti_trap_type'] = 'spring'
    row['pti_score'] = 0.45
    row['tf4h_boms_displacement'] = 90.0
    row['tf1h_bos_bullish'] = True
    row['tf1d_boms_strength'] = 0.35
    row['liquidity_score'] = 0.40
    row['wyckoff_score'] = 0.45
    row['momentum_score'] = 0.50

    df = pd.DataFrame([row])
    archetype, fusion, liq = logic.check_archetype(row, None, df, 0)

    # A has higher priority
    assert archetype == 'trap_reversal'


def test_disabled_archetype(archetype_config, base_row):
    """Test that disabled archetypes are not detected."""
    config = archetype_config.copy()
    config['enable_A'] = False

    logic = ArchetypeLogic(config)

    row = base_row.copy()
    row['tf1h_pti_trap_type'] = 'spring'
    row['pti_score'] = 0.45
    row['tf4h_boms_displacement'] = 90.0
    row['liquidity_score'] = 0.35
    row['wyckoff_score'] = 0.40
    row['momentum_score'] = 0.45

    df = pd.DataFrame([row])
    archetype, fusion, liq = logic.check_archetype(row, None, df, 0)

    # A is disabled, should not match
    assert archetype != 'trap_reversal'


def test_use_archetypes_false(archetype_config, base_row):
    """Test that use_archetypes=false disables all detection."""
    config = archetype_config.copy()
    config['use_archetypes'] = False

    logic = ArchetypeLogic(config)

    row = base_row.copy()
    row['tf1h_pti_trap_type'] = 'spring'
    row['pti_score'] = 0.45
    row['tf4h_boms_displacement'] = 90.0

    df = pd.DataFrame([row])
    archetype, fusion, liq = logic.check_archetype(row, None, df, 0)

    assert archetype is None
    assert fusion == 0.0
    assert liq == 0.0


def test_fusion_score_calculation(archetype_config, base_row):
    """Test fusion score calculation."""
    logic = ArchetypeLogic(archetype_config)

    row = base_row.copy()
    row['wyckoff_score'] = 0.50
    row['liquidity_score'] = 0.40
    row['momentum_score'] = 0.60

    fusion = logic.calculate_fusion_score(row)

    # Weighted: 0.30*0.50 + 0.25*0.40 + 0.30*0.60 + 0.15*0.0 = 0.43
    assert fusion == pytest.approx(0.43, rel=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
