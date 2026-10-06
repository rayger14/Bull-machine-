"""Separately bounded, continuous full-prefix adapter for the frozen constructors.

The old 2,048-hour API and its source are untouched. Semantic source IDs use its
unchanged contract; this adapter's implementation/cap are separately manifested.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research import causal_parent_ledger as native
from scripts.research.replay_clock import json_safe, validate_bars
from scripts.research.study_contract import protocol
from scripts.research.virtual_book_replay import side_effect_guard

_POLICY = protocol()
MAX_STUDY_HOURS = int((pd.Timestamp(_POLICY['source_end_exclusive']) - pd.Timestamp(_POLICY['seed'])) / pd.Timedelta('1h'))


def build_study_parents(hourly, *, instrument, data_stream_id, atr_contract, source_paths,
                        expected_hashes, anchor_timeframe='4H'):
    native._nonempty(instrument, 'instrument')
    native._nonempty(data_stream_id, 'data_stream_id')
    if anchor_timeframe not in native.ANCHOR_TIMEFRAMES:
        raise ValueError('unsupported parent anchor')
    if len(hourly) > MAX_STUDY_HOURS:
        raise ValueError('input exceeds frozen full-source study cap')
    validate_bars(hourly, '1h')
    bars = hourly.copy(deep=True)
    bars.index = bars.index.tz_convert('UTC')
    normalized_atr = native._validate_atr_contract(atr_contract)
    native._validate_atr(bars)
    native._validate_atr_availability(bars)
    source_bytes, source_hashes = native._validate_sources(source_paths, expected_hashes)
    buckets, anchors = native._independent_anchor_buckets(bars, anchor_timeframe)
    contract = native._contract_manifest(instrument, anchor_timeframe, 3, normalized_atr, source_hashes)
    records = []
    with side_effect_guard(records):
        htf = native._compile_source(source_bytes['htf'], source_paths['htf'], '_causal_parent_recovered_htf')
        ranges = native._compile_source(source_bytes['range'], source_paths['range'], '_causal_parent_recovered_range')
        for namespace, functions in ((htf, ('resample_htf', 'detect_fractal_pivots', '_broadcast')),
                                     (ranges, ('build_structural_range',))):
            if any(not callable(namespace.get(name)) for name in functions):
                raise ValueError('recovered source function missing')
        complete_hours = native._private_complete_hours(bars, buckets['completed'])
        native._assert_aggregation_parity(htf['resample_htf'](complete_hours, anchor_timeframe), anchors)
        source_anchor = anchors.copy()
        epoch = source_anchor.index.asi8.copy()
        source_anchor.index = source_anchor.index.tz_convert('UTC').tz_localize(None)
        source_anchor['close_time'] = source_anchor['close_time'].dt.tz_convert('UTC').dt.tz_localize(None)
        if not np.array_equal(epoch, source_anchor.index.tz_localize('UTC').asi8):
            raise ValueError('UTC epoch identity failure')
        pivots = htf['detect_fractal_pivots'](source_anchor, 3)
        private = bars.copy(deep=True)
        epoch = private.index.asi8.copy()
        private.index = private.index.tz_localize(None)
        if not np.array_equal(epoch, private.index.tz_localize('UTC').asi8):
            raise ValueError('UTC epoch identity failure')
        private['swing_low_50'] = htf['_broadcast'](pivots, private.index, 'pivot_low_level', 'is_swing_low')
        private['swing_high_50'] = htf['_broadcast'](pivots, private.index, 'pivot_high_level', 'is_swing_high')
        source_range = ranges['build_structural_range'](private, break_buffer_atr=0., break_confirm_bars=1)
    if records:
        raise ValueError('source side effects recorded')
    source_range.index = source_range.index.tz_localize('UTC')
    pivots.index = pivots.index.tz_localize('UTC')
    pivots['confirm_time'] = pd.to_datetime(pivots['confirm_time'], utc=True)
    pivot_records = native._pivot_records_from_source(pivots, anchor_buckets=anchors,
                    contract_id=contract['contract_id'], data_stream_id=data_stream_id,
                    instrument=instrument, anchor_timeframe=anchor_timeframe, pivot_n=3)
    versions, transitions = native._annotate_source_outputs(bars, source_range, pivot_records,
                    contract_id=contract['contract_id'], data_stream_id=data_stream_id)
    first, last = bars.index[0], bars.index[-1] + pd.Timedelta('1h')
    coverage = {'first_open': str(first), 'first_processed_close': str(first + pd.Timedelta('1h')),
                'last_processed_close': str(last), 'query_exclusive_end': str(last + pd.Timedelta('1h')),
                'input_hours': len(bars)}
    manifest = {**contract, 'instrument': instrument, 'data_stream_id': data_stream_id,
                'input_hash': native._input_hash(bars, instrument, data_stream_id, normalized_atr),
                'input_rows': len(bars), 'source_paths': {k: str(source_paths[k]) for k in ('htf', 'range')},
                'receipt_certified': False,
                'study_adapter': {'schema': 'continuous-study-parent-v1', 'sha256': native._file_hash(__file__),
                                  'frozen_adapter_sha256': native._file_hash(native.__file__),
                                  'maximum_input_hours': MAX_STUDY_HOURS,
                                  'restart_method': 'rebuild_same_full_prefix',
                                  'monthly_state_reset': False}}
    return json_safe({'certified': False,
                      'issues': ['recovered_strategy_source_only', 'no_source_receipt_certification',
                                 'no_native_signal_or_execution_integration', 'anchor_parameters_are_unselected_hypotheses'],
                      'manifest': manifest, 'coverage': coverage, 'anchor_buckets': buckets,
                      'pivots': pivot_records, 'versions': versions, 'transitions': transitions})
