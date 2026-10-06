"""Completed-candle and polling contracts, not Wyckoff profitability tests."""
import importlib
import importlib.util

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal


def bars(n=8, freq='1h', start='2026-01-01T00:00Z'):
    idx = pd.date_range(start, periods=n, freq=freq)
    price = np.arange(n, dtype=float) + 100
    return pd.DataFrame(dict(open=price, high=price+2, low=price-1,
                             close=price+1, volume=np.arange(n)+10.), index=idx)


def prepare(*args, **kwargs):
    name = 'engine.wyckoff.candle_integrity'
    assert importlib.util.find_spec(name) is not None, 'Missing completed Wyckoff candle preparer'
    return importlib.import_module(name).prepare_wyckoff_bars(*args, **kwargs)


def test_literal_complete_aggregate_and_close_availability():
    result = prepare(bars(4), '4h', '2026-01-01T04:00Z')
    expected = pd.DataFrame([dict(open=100., high=105., low=99., close=104., volume=46.)],
                            index=pd.DatetimeIndex(['2026-01-01T00:00Z']))
    assert_frame_equal(result.frame, expected, check_freq=False)
    assert result.status == 'available'
    assert result.source == 'hourly_aggregate'
    assert result.available_at == pd.Timestamp('2026-01-01T04:00Z')
    assert result.last_input_close == result.available_at


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'misaligned', 'nan', 'infinite',
                                    'negative_volume', 'negative_price', 'invalid_range'])
def test_invalid_closed_constituent_rejects_latest_bin(damage):
    frame = bars(4)
    if damage == 'missing':
        frame = frame.drop(frame.index[1])
    elif damage == 'duplicate':
        frame = pd.concat([frame, frame.iloc[[1]]])
    elif damage == 'misaligned':
        frame.index = pd.DatetimeIndex([frame.index[0], frame.index[1]+pd.Timedelta(minutes=1),
                                       *frame.index[2:]])
    else:
        column, value = {'nan': ('close', np.nan), 'infinite': ('volume', np.inf),
                         'negative_volume': ('volume', -1), 'negative_price': ('low', -1),
                         'invalid_range': ('close', 1000)}[damage]
        frame.loc[frame.index[1], column] = value
    result = prepare(frame, '4h', '2026-01-01T04:00Z')
    assert result.status == 'unavailable'
    assert result.frame.empty
    assert result.available_at is None


def test_current_unclosed_bin_is_not_confirmation_but_previous_remains_usable():
    full = bars(8)
    at5 = prepare(full, '4h', '2026-01-01T05:30Z')
    at4 = prepare(full.iloc[:4], '4h', '2026-01-01T04:00Z')
    assert_frame_equal(at5.frame, at4.frame)
    assert at5.status == 'available'
    assert at5.available_at == at4.available_at
    at3 = prepare(full, '4h', '2026-01-01T03:59Z')
    assert at3.status == 'unavailable'


def test_missing_latest_closed_bin_cannot_carry_older_as_current():
    result = prepare(bars(4), '4h', '2026-01-01T08:00Z')
    assert result.status == 'unavailable'
    assert result.reason == 'latest_closed_bin_missing_or_invalid'
    assert result.frame.empty


def test_internal_gap_resets_history_to_contiguous_suffix():
    frame = bars(16).drop(pd.Timestamp('2026-01-01T05:00Z'))
    result = prepare(frame, '4h', '2026-01-01T16:00Z', min_bars=2)
    assert list(result.frame.index) == list(pd.date_range('2026-01-01T08:00Z', periods=2, freq='4h'))
    assert result.status == 'available'
    insufficient = prepare(frame, '4h', '2026-01-01T16:00Z', min_bars=3)
    assert insufficient.status == 'unavailable'
    assert insufficient.reason == 'insufficient_contiguous_history'


def test_hourly_integrity_uses_same_gap_and_close_rules():
    frame = bars(8).drop(pd.Timestamp('2026-01-01T04:00Z'))
    result = prepare(frame, '1h', '2026-01-01T08:00Z')
    assert_frame_equal(result.frame, frame.iloc[-3:], check_freq=False)
    assert result.source == 'hourly'


def test_future_rows_and_bad_future_duplicate_do_not_change_prefix():
    prefix = prepare(bars(4), '4h', '2026-01-01T04:00Z')
    extended = bars(8)
    extended = pd.concat([extended, extended.iloc[[6]]])
    extended.loc[extended.index[-1], 'close'] = np.nan
    result = prepare(extended, '4h', '2026-01-01T04:00Z')
    assert_frame_equal(prefix.frame, result.frame)
    assert prefix.available_at == result.available_at


def test_native_daily_deep_history_and_partial_leading_hourly_day():
    hours = bars(40, start='2026-01-02T08:00Z')  # ends Jan 4 midnight
    daily = bars(4, '1d')
    result = prepare(hours, '1d', '2026-01-04T00:00Z', native_daily=daily, min_bars=3)
    assert result.status == 'available'
    assert result.source == 'native_daily+hourly_aggregate'
    assert list(result.frame.index) == list(daily.index[:3])
    assert result.frame.iloc[-1]['volume'] == hours.iloc[16:]['volume'].sum()
    assert result.frame.iloc[1]['volume'] == daily.iloc[1]['volume']
    assert result.available_at == pd.Timestamp('2026-01-04T00:00Z')


def test_native_daily_cannot_patch_known_internal_hourly_gap():
    hours = bars(72).drop(pd.Timestamp('2026-01-03T03:00Z'))
    result = prepare(hours, '1d', '2026-01-04T00:00Z', native_daily=bars(4, '1d'))
    assert result.status == 'unavailable'


@pytest.mark.parametrize('damage', ['duplicate', 'misaligned', 'gap', 'invalid'])
def test_native_daily_is_also_validated_without_claiming_hourly_constituents(damage):
    daily = bars(3, '1d')
    if damage == 'duplicate':
        daily = pd.concat([daily, daily.iloc[[-1]]])
    elif damage == 'misaligned':
        daily.index = daily.index + pd.Timedelta(hours=1)
    elif damage == 'gap':
        daily = daily.iloc[:2]
    else:
        daily.loc[daily.index[-1], 'volume'] = -1
    result = prepare(bars(0), '1d', '2026-01-04T00:00Z', native_daily=daily)
    assert result.status == 'unavailable'


def test_native_daily_without_hourly_history_has_explicit_provenance():
    result = prepare(bars(0), '1d', '2026-01-04T00:00Z', native_daily=bars(4, '1d'))
    assert result.status == 'available'
    assert result.source == 'native_daily'
    assert len(result.frame) == 3


def test_naive_timestamps_are_declared_utc_and_inputs_are_not_mutated():
    frame = bars(4)
    frame.index = frame.index.tz_localize(None)
    original = frame.copy(deep=True)
    result = prepare(frame, '4h', '2026-01-01T04:00Z')
    assert str(result.frame.index.tz) == 'UTC'
    assert_frame_equal(frame, original)


@pytest.fixture
def processor():
    from scripts.research.live_feature_replay import LiveFeatureProcessor
    return LiveFeatureProcessor()


def test_warmup_tail_computes_once_without_duplicate_or_history_growth(processor):
    from scripts.research.live_feature_replay import deny_network
    frame = bars(36)
    fc = processor.fc
    fc.ingest_candles(frame)
    candle = dict(frame.iloc[-1], timestamp=frame.index[-1], funding_rate=.001)
    with deny_network():
        first = fc.update(candle)
        history = list(fc._funding_history)
        second = fc.update(candle)
    assert len(fc._buf) == len(frame)
    assert fc._buf.index.is_unique
    assert len(history) == 1
    assert fc._funding_history == history
    assert_series_equal(first, second)
    first['close'] = -999
    with deny_network():
        assert fc.update(candle)['close'] == candle['close']


@pytest.mark.parametrize('kind', ['conflicting', 'older'])
def test_conflicting_or_out_of_order_poll_rejected_without_mutation(processor, kind):
    from scripts.research.live_feature_replay import deny_network
    fc = processor.fc
    frame = bars(36)
    fc.ingest_candles(frame)
    before = fc._buf.copy(deep=True)
    candle = dict(frame.iloc[-1 if kind == 'conflicting' else -2],
                  timestamp=frame.index[-1 if kind == 'conflicting' else -2])
    if kind == 'conflicting':
        candle['close'] += .1
    with deny_network(), pytest.raises(ValueError, match='duplicate|order'):
        fc.update(candle)
    assert_frame_equal(fc._buf, before)
    assert not fc._funding_history


def test_real_wyckoff_adapter_uses_only_completed_bins_with_explicit_timeframe(processor, monkeypatch):
    fc = processor.fc
    fc.ingest_candles(bars(721))  # completed hour Jan 31 00:00; daily still Jan 30
    calls = []
    original = processor.module.detect_all_wyckoff_events

    def observe(frame, cfg=None, **kwargs):
        calls.append((cfg.get('timeframe'), frame.copy()))
        return original(frame, cfg=cfg, **kwargs)

    monkeypatch.setattr(processor.module, 'detect_all_wyckoff_events', observe)
    out = fc._wyckoff_features()
    assert [tf for tf, _ in calls] == ['1d', '4h', '1h']
    for tf, frame in calls:
        assert frame.index[-1] + pd.Timedelta(tf) <= pd.Timestamp('2026-01-31T01:00Z')
    assert len(calls[0][1]) == 30
    assert len(calls[1][1]) == 180
    for prefix in ('', 'tf4h_', 'tf1d_'):
        assert out[prefix+'wyckoff_evidence_status'] == 'available'
    assert out['tf1d_wyckoff_available_at'] == '2026-01-31T00:00:00+00:00'


def test_hourly_valid_even_when_higher_timeframes_have_insufficient_history(processor):
    processor.fc.ingest_candles(bars(36))
    out = processor.fc._wyckoff_features()
    assert out['wyckoff_evidence_status'] == 'available'
    assert out['tf4h_wyckoff_evidence_status'] == 'unavailable'
    assert out['tf1d_wyckoff_evidence_status'] == 'unavailable'


def test_gap_resets_hourly_history_and_clears_stale_event_display(processor):
    frame = bars(80).drop(bars(80).index[-5])
    processor.fc.ingest_candles(frame)
    processor.fc.last_wyckoff_event_history = [{'event': 'old'}]
    processor.fc.last_wyckoff_conviction = {'old': .9}
    out = processor.fc._wyckoff_features()
    assert out['wyckoff_evidence_status'] == 'unavailable'
    assert out['wyckoff_evidence_reason'] == 'insufficient_contiguous_history'
    assert out['wyckoff_score'] == 0
    assert not processor.fc.last_wyckoff_event_history
    assert not processor.fc.last_wyckoff_conviction


def test_ema_proxy_is_separate_and_cannot_be_presented_as_wyckoff(processor, monkeypatch):
    processor.fc.ingest_candles(bars(40))
    monkeypatch.setattr(processor.module, 'WYCKOFF_AVAILABLE', False)
    out = processor.fc._wyckoff_features()
    assert out['wyckoff_ema_alignment_proxy'] > 0
    for key in ('wyckoff_score', 'tf1d_wyckoff_score', 'tf4h_wyckoff_phase_score'):
        assert out[key] == 0
    for prefix in ('', 'tf4h_', 'tf1d_'):
        assert out[prefix+'wyckoff_evidence_status'] == 'unavailable'
        assert out[prefix+'wyckoff_evidence_source'] == 'ema_proxy'


def test_daily_error_does_not_erase_valid_hourly_and_four_hour_evidence(processor, monkeypatch):
    fc = processor.fc
    fc.ingest_candles(bars(721))
    original = processor.module.detect_all_wyckoff_events

    def fail_daily(frame, cfg=None, **kwargs):
        if cfg.get('timeframe') == '1d':
            raise ValueError('fixture daily detector failure')
        return original(frame, cfg=cfg, **kwargs)

    monkeypatch.setattr(processor.module, 'detect_all_wyckoff_events', fail_daily)
    out = fc._wyckoff_features()
    assert out['tf1d_wyckoff_evidence_status'] == 'error'
    assert out['tf1d_wyckoff_score'] == 0
    assert out['tf4h_wyckoff_evidence_status'] == 'available'
    assert out['wyckoff_evidence_status'] == 'available'


def test_null_provenance_survives_numeric_fill(processor):
    source = pd.Series({'wyckoff_spring_a_candidate_index': None,
                        'wyckoff_available_at': None,
                        'tf1d_wyckoff_last_input_close': None,
                        'wyckoff_score': np.nan}, dtype=object)
    filled = processor.fc._fill_nans(source)
    assert filled['wyckoff_spring_a_candidate_index'] is None
    assert filled['wyckoff_available_at'] is None
    assert filled['tf1d_wyckoff_last_input_close'] is None
    assert filled['wyckoff_score'] == 0


def test_daily_ingest_does_not_hide_invalid_timestamp_or_duplicates(processor):
    frame = bars(3, '1d')
    frame.index += pd.Timedelta(minutes=1)
    frame = pd.concat([frame, frame.iloc[[-1]]])
    processor.fc.ingest_daily_candles(frame)
    assert_frame_equal(processor.fc._daily_buf, frame)
