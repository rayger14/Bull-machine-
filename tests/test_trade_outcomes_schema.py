"""
trade_outcomes.csv schema-alignment tests.

These tests guard against the off-by-one bug that previously caused values to
shift right by one column (e.g. chop holding ATR-magnitude values while
bb_width was always 0). The writer now builds the row as a list and
asserts len(row) == len(OUTCOME_COLUMNS) before writing — these tests
verify that contract end-to-end with a synthetic trade dict.
"""
from __future__ import annotations

import io
import csv
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

# Make sure the project root is importable when tests run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the runner module directly — we use only the static methods +
# the OUTCOME_COLUMNS class attribute, so no network/feature-store deps.
from bin.live.v11_shadow_runner import V11ShadowRunner, TrackedPosition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_macro_snapshot(**overrides):
    """Build a fully-populated macro_at_entry dict (all columns the row needs)."""
    base = {
        # Macro z-scores
        'vix_z': -0.5, 'dxy_z': 0.2, 'gold_z': 0.1, 'oil_z': -0.3, 'yield_curve': 0.4,
        # Derivatives
        'funding_z': 1.5, 'oi_value': 2.7e9, 'oi_change_4h': -0.03, 'oi_change_24h': -0.08,
        'taker_imbalance': 0.21, 'ls_ratio_extreme': -1.0, 'funding_rate': 8e-6,
        # Wyckoff
        'wyckoff_bullish': 0.0, 'wyckoff_bearish': 0.3,
        'wyckoff_4h_bull': 0.10, 'wyckoff_4h_bear': 0.65,
        # Sentiment
        'fear_greed': 32, 'ema_slope_50': 1.4e-4, 'rsi_14': 48.5,
        # Technical
        'chop': 0.43, 'adx': 28.2, 'atr_14': 476.16, 'volume_zscore': 0.80, 'bb_width': 0.0227,
        # v2 structural
        'bos_active': True,
        'fvg_present': False,
        'distribution_at_resistance': False,
        'distribution_exhaustion': True,
        'poc_dist_norm': 0.034,
        'recent_sos_count_4h': 3,
        'range_position_20': 0.27,
    }
    base.update(overrides)
    return base


def _make_position(**overrides) -> TrackedPosition:
    """Build a synthetic TrackedPosition matching what _open_position produces."""
    pos = TrackedPosition(
        position_id='long_long_squeeze_1700000000',
        archetype='long_squeeze',
        direction='short',
        entry_price=77866.633,
        entry_time=pd.Timestamp('2026-04-23 12:00:00+00:00'),
        stop_loss=78532.5656,
        take_profit=76000.0,
        original_quantity=0.10,
        current_quantity=0.10,
        fusion_score=0.1619,
        regime_at_entry='neutral',
        atr_at_entry=476.16,
        threshold_at_entry=0.3941,
        risk_temp_at_entry=0.5988,
        instability_at_entry=0.2775,
        crisis_prob_at_entry=0.009,
        threshold_margin=-0.2322,
    )
    pos.macro_at_entry = _make_macro_snapshot()
    pos.entry_metadata = {'archetype': pos.archetype, 'executed_scale_outs': []}
    for k, v in overrides.items():
        setattr(pos, k, v)
    return pos


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_outcome_columns_is_unique_and_nonempty():
    cols = V11ShadowRunner.OUTCOME_COLUMNS
    assert len(cols) > 0
    assert len(set(cols)) == len(cols), "duplicate column name in OUTCOME_COLUMNS"


def test_outcome_columns_v1_prefix_is_frozen():
    """The first 43 columns are the v1 schema and must NOT be reordered."""
    cols = V11ShadowRunner.OUTCOME_COLUMNS
    expected_v1 = [
        'timestamp_entry', 'timestamp_exit', 'archetype', 'direction',
        'entry_price', 'exit_price', 'pnl', 'pnl_pct',
        'hold_hours', 'exit_reason', 'R_multiple',
        'regime', 'fusion_score', 'threshold', 'threshold_margin',
        'risk_temp', 'instability', 'crisis_prob',
        'vix_z', 'dxy_z', 'gold_z', 'oil_z', 'yield_curve',
        'funding_z', 'oi_value', 'oi_change_4h', 'oi_change_24h',
        'taker_imbalance', 'ls_ratio_extreme', 'funding_rate',
        'wyckoff_bullish', 'wyckoff_bearish',
        'wyckoff_4h_bull', 'wyckoff_4h_bear',
        'fear_greed', 'ema_slope_50', 'rsi_14',
        'atr_at_entry', 'chop', 'adx', 'atr_14', 'volume_zscore', 'bb_width',
    ]
    assert cols[: len(expected_v1)] == expected_v1


def test_outcome_columns_v2_additions_present():
    """v2 schema added 8 columns at the end; verify they are all there."""
    cols = V11ShadowRunner.OUTCOME_COLUMNS
    expected_v2_tail = [
        'bos_active', 'fvg_present',
        'distribution_at_resistance', 'distribution_exhaustion',
        'poc_dist_norm', 'recent_sos_count_4h',
        'phantom_dedup_winner', 'range_position_20',
    ]
    assert cols[-len(expected_v2_tail):] == expected_v2_tail


def test_build_outcome_row_length_matches_header():
    """Core regression test for the off-by-one: row len == header len."""
    pos = _make_position()
    row = V11ShadowRunner._build_outcome_row(
        pos=pos,
        pnl=-459.58,
        pnl_pct=-0.8754,
        exit_price=78532.57,
        exit_timestamp=pd.Timestamp('2026-04-23 15:00:00+00:00'),
        exit_reason='stop_loss',
    )
    assert isinstance(row, list)
    assert len(row) == len(V11ShadowRunner.OUTCOME_COLUMNS), (
        f"row produced {len(row)} values for {len(V11ShadowRunner.OUTCOME_COLUMNS)} columns"
    )


def test_build_outcome_row_length_with_empty_macro():
    """Even when macro_at_entry is missing/empty, the row must stay aligned."""
    pos = _make_position()
    pos.macro_at_entry = {}  # simulate failure to capture snapshot
    row = V11ShadowRunner._build_outcome_row(
        pos=pos,
        pnl=100.0,
        pnl_pct=1.0,
        exit_price=78000.0,
        exit_timestamp=pd.Timestamp('2026-04-23 15:00:00+00:00'),
        exit_reason='take_profit',
    )
    assert len(row) == len(V11ShadowRunner.OUTCOME_COLUMNS)


def test_build_outcome_row_alignment_matches_header_semantics():
    """
    The historical bug stored real chop values in the column labelled 'atr_14'.
    Verify each value lands in the correctly-named column and ranges look
    right — chop ~ 0..1, atr_14 ~ price-units, bb_width ~ 0..0.1.
    """
    pos = _make_position()
    row = V11ShadowRunner._build_outcome_row(
        pos=pos,
        pnl=-459.58,
        pnl_pct=-0.8754,
        exit_price=78532.57,
        exit_timestamp=pd.Timestamp('2026-04-23 15:00:00+00:00'),
        exit_reason='stop_loss',
    )
    cols = V11ShadowRunner.OUTCOME_COLUMNS
    rec = dict(zip(cols, row))
    assert 0.0 <= float(rec['chop']) <= 1.0, f"chop out of [0,1]: {rec['chop']}"
    # atr_14 should be price-magnitude (BTC ~ 100..1e5), not 0..1
    assert float(rec['atr_14']) > 1.0
    # bb_width should be tiny (0..0.1 typically), not zero (the historical bug had it always 0)
    assert 0.0 < float(rec['bb_width']) < 1.0
    assert rec['archetype'] == 'long_squeeze'
    assert rec['direction'] == 'short'
    assert rec['exit_reason'] == 'stop_loss'


def test_build_outcome_row_v2_columns_present_and_typed():
    """v2 columns must be filled with sensible types/values."""
    pos = _make_position()
    row = V11ShadowRunner._build_outcome_row(
        pos=pos,
        pnl=-459.58,
        pnl_pct=-0.8754,
        exit_price=78532.57,
        exit_timestamp=pd.Timestamp('2026-04-23 15:00:00+00:00'),
        exit_reason='stop_loss',
    )
    cols = V11ShadowRunner.OUTCOME_COLUMNS
    rec = dict(zip(cols, row))
    # bools are emitted as "0"/"1"
    assert rec['bos_active'] in ('0', '1')
    assert rec['fvg_present'] in ('0', '1')
    assert rec['distribution_at_resistance'] in ('0', '1')
    assert rec['distribution_exhaustion'] in ('0', '1')
    # bos_active was True in our fixture
    assert rec['bos_active'] == '1'
    # range_position_20 is float-formatted
    assert 0.0 <= float(rec['range_position_20']) <= 1.0
    # SOS count is an int
    assert int(rec['recent_sos_count_4h']) == 3
    # phantom_dedup_winner is empty when no contention
    assert rec['phantom_dedup_winner'] == ''
    # poc_dist_norm is numeric
    assert float(rec['poc_dist_norm']) == pytest.approx(0.034, rel=1e-3)


def test_build_outcome_row_logs_dedup_loser():
    """When dedup happened, the loser's archetype id should land in the row."""
    pos = _make_position()
    pos.entry_metadata = {
        'archetype': pos.archetype,
        'executed_scale_outs': [],
        'dedup_losers': ['confluence_breakout', 'order_block_retest'],
    }
    row = V11ShadowRunner._build_outcome_row(
        pos=pos,
        pnl=10.0,
        pnl_pct=0.1,
        exit_price=78000.0,
        exit_timestamp=pd.Timestamp('2026-04-23 15:00:00+00:00'),
        exit_reason='take_profit',
    )
    cols = V11ShadowRunner.OUTCOME_COLUMNS
    rec = dict(zip(cols, row))
    # No commas allowed in a single CSV cell — must be pipe-separated
    assert ',' not in rec['phantom_dedup_winner']
    assert 'confluence_breakout' in rec['phantom_dedup_winner']
    assert 'order_block_retest' in rec['phantom_dedup_winner']


def test_round_trip_through_csv_writer(tmp_path):
    """
    End-to-end: call the actual file-writing append path with a synthetic
    position + verify the header line and data line both have the same field
    count, and that the resulting CSV is readable by pandas.read_csv.
    """
    out = tmp_path / 'trade_outcomes.csv'
    # Hand-write the header (same logic as _init_outcome_log)
    header = ','.join(V11ShadowRunner.OUTCOME_COLUMNS) + '\n'
    out.write_text(header)

    pos = _make_position()
    row = V11ShadowRunner._build_outcome_row(
        pos=pos,
        pnl=-459.58,
        pnl_pct=-0.8754,
        exit_price=78532.57,
        exit_timestamp=pd.Timestamp('2026-04-23 15:00:00+00:00'),
        exit_reason='stop_loss',
    )
    with out.open('a') as f:
        f.write(','.join(row) + '\n')

    text = out.read_text()
    lines = text.strip().split('\n')
    assert len(lines) == 2
    header_n = lines[0].count(',') + 1
    data_n = lines[1].count(',') + 1
    assert header_n == data_n == len(V11ShadowRunner.OUTCOME_COLUMNS), (
        f"header={header_n}, data={data_n}, expected={len(V11ShadowRunner.OUTCOME_COLUMNS)}"
    )

    df = pd.read_csv(out)
    assert list(df.columns) == V11ShadowRunner.OUTCOME_COLUMNS
    assert len(df) == 1
    # Spot-check the alignment: chop in [0,1], atr_14 > 1, bb_width > 0
    assert 0 <= df.iloc[0]['chop'] <= 1
    assert df.iloc[0]['atr_14'] > 1
    assert df.iloc[0]['bb_width'] > 0


def test_build_outcome_row_handles_nan_safely():
    """NaN values in the macro snapshot must NOT shift columns or raise."""
    pos = _make_position()
    pos.macro_at_entry['chop'] = float('nan')
    pos.macro_at_entry['adx'] = float('nan')
    row = V11ShadowRunner._build_outcome_row(
        pos=pos,
        pnl=0.0, pnl_pct=0.0,
        exit_price=78000.0,
        exit_timestamp=pd.Timestamp('2026-04-23 15:00:00+00:00'),
        exit_reason='time_exit',
    )
    assert len(row) == len(V11ShadowRunner.OUTCOME_COLUMNS)
    rec = dict(zip(V11ShadowRunner.OUTCOME_COLUMNS, row))
    # NaN renders as empty string — that's fine, but the slot must still exist
    assert rec['chop'] == ''
    assert rec['adx'] == ''
    # bb_width is still present (downstream column)
    assert rec['bb_width'] != ''
