"""Candidate-to-confirmation contracts, not a complete schematic benchmark."""
import pandas as pd
import pytest

from engine.wyckoff import events as w


def candle(low=101.0, close=105.0, high=110.0, z=0.0):
    return dict(open=close, high=high, low=low, close=close, volume_z=z)


def parent(cfg=None):
    sm = w.WyckoffStateMachine(cfg or {})
    sm.process_bar(0, candle(100, 102, 103, 3), {'sc': True})
    sm.process_bar(3, candle(105, 109, 110), {'ar': True})
    sm.process_bar(6, candle(101, 103, 106, -.5), {'st': True})
    return sm


def raw_frame(kind='spring_a', structure=True):
    idx = pd.date_range('2026-01-01', periods=66, freq='1h', tz='UTC')
    df = pd.DataFrame(dict(open=105., high=110., low=100., close=105.,
                           volume=1000., volume_z=0.), index=idx)
    for key in w._ACCUM_EVENTS + w._DISTRIB_EVENTS:
        df[f'wyckoff_{key}'] = False
        df[f'wyckoff_{key}_confidence'] = 0.0

    def put(i, key, row):
        for col, val in row.items():
            df.loc[idx[i], col] = val
        df.loc[idx[i], f'wyckoff_{key}'] = True
        df.loc[idx[i], f'wyckoff_{key}_confidence'] = .8

    if kind.startswith('spring'):
        if structure:
            put(30, 'sc', candle(100, 102, 103, 3))
            put(33, 'ar', candle(105, 109, 110))
            put(36, 'st', candle(101, 103, 106, -.5))
        df.loc[idx[60], ['low', 'close', 'volume_z']] = [98 if kind == 'spring_a' else 99.5, 101, 1]
        df.loc[idx[63], ['low', 'close']] = [103, 106]
        detect = w.detect_spring_type_a if kind == 'spring_a' else w.detect_spring_type_b
    else:
        if structure:
            put(30, 'bc', candle(104, 108, 110, 3))
            put(33, 'as', candle(100, 101, 105))
        df.loc[idx[60], ['high', 'close', 'volume_z']] = [113, 109, 1]
        df.loc[idx[63], ['high', 'close']] = [107, 104]
        detect = w.detect_upthrust
    df[f'wyckoff_{kind}'], df[f'wyckoff_{kind}_confidence'] = detect(df.copy(), {})
    assert df[f'wyckoff_{kind}'].iloc[63], 'raw-recognizer fixture must really fire'
    return df


def evidence(sm, **changes):
    cls = getattr(w, 'DelayedEventEvidence', None)
    assert cls is not None, 'Delayed events need an explicit provenance record'
    snap = sm.parent_snapshot()
    values = dict(event_type='spring_a', candidate_index=7, confirmation_index=10,
                  candidate_extreme=98., prior_swept_boundary=100.,
                  candidate_parent_id=snap['id'], candidate_parent_context=snap['context'],
                  candidate_parent_status=snap['status'],
                  candidate_timestamp=pd.Timestamp('2026-01-01T07:00Z'),
                  confirmation_timestamp=pd.Timestamp('2026-01-01T10:00Z'),
                  available_at=pd.Timestamp('2026-01-01T11:00Z'))
    values.update(changes)
    return cls(**values)


def test_rejected_spring_does_not_advance_phase():
    sm = parent()
    valid, _ = sm.process_bar(9, candle(100.5, 104, 106), {'spring_a': True})
    assert valid['spring_a'] is False
    assert sm.state == w.WyckoffState.ACCUM_ST
    assert sm.get_phase_dir() == 'A_accum'


@pytest.mark.parametrize('kind', ['spring_a', 'spring_b', 'ut'])
def test_actual_delayed_recognizer_uses_candidate_not_recovery_geometry(kind):
    out = w._apply_state_machine_validation(raw_frame(kind), {'timeframe': '1h'})
    assert bool(out[f'wyckoff_{kind}'].iloc[63])
    assert not out[f'wyckoff_{kind}'].iloc[60:63].any()
    assert out[f'wyckoff_{kind}_candidate_index'].iloc[63] == 60
    assert out[f'wyckoff_{kind}_evidence_status'].iloc[63] == 'structured'
    assert out[f'wyckoff_{kind}_available_at'].iloc[63] == '2026-01-03T16:00:00+00:00'


def test_missing_metadata_rejects_instead_of_using_current_geometry():
    sm = parent()
    valid, _ = sm.process_bar(10, candle(98, 104), {'spring_a': True}, event_metadata={})
    assert not valid['spring_a']
    assert sm.state == w.WyckoffState.ACCUM_ST


@pytest.mark.parametrize('changes', [
    {'candidate_extreme': float('nan')}, {'prior_swept_boundary': float('inf')},
    {'candidate_index': 11}, {'confirmation_index': 9},
    {'candidate_parent_context': 'distribution'}, {'candidate_parent_status': 'unestablished'},
    {'available_at': pd.Timestamp('2026-01-01T09:00Z')},
    {'confirmation_timestamp': pd.Timestamp('2026-01-01T09:00Z')},
])
def test_bad_provenance_cannot_advance_phase(changes):
    sm = parent()
    item = evidence(sm, **changes)
    valid, _ = sm.process_bar(10, candle(98, 104), {'spring_a': True},
                              event_metadata={'spring_a': item})
    assert not valid['spring_a']
    assert sm.state == w.WyckoffState.ACCUM_ST


@pytest.mark.parametrize('transition', ['replacement', 'invalidation', 'expiry', 'reset'])
def test_parent_lost_before_confirmation_cannot_reattach_or_fallback(transition):
    sm = parent()
    item = evidence(sm)
    if transition == 'replacement':
        sm.process_bar(8, candle(100, 102, 103, 3), {'sc': True})
        sm.process_bar(9, candle(105, 109, 110), {'ar': True})
    elif transition == 'invalidation':
        sm.process_bar(8, candle(97, 98, 100, 3), {})
    elif transition == 'expiry':
        sm.max_structure_bars = sm.bars_in_structure
    else:
        sm.reset()
    valid, mods = sm.process_bar(10, candle(98, 104), {'spring_a': True},
                                 event_metadata={'spring_a': item})
    assert not valid['spring_a']
    assert 'spring_a' not in mods
    assert sm.state != w.WyckoffState.ACCUM_SPRING


def test_no_parent_cannot_be_born_and_disappear_between_candidate_and_confirmation():
    sm = w.WyckoffStateMachine({})
    item = evidence(sm)
    sm.process_bar(8, candle(100, 102, 103, 3), {'sc': True})
    sm.reset()
    valid, _ = sm.process_bar(10, candle(98, 104), {'spring_a': True},
                              event_metadata={'spring_a': item})
    assert not valid['spring_a']


def test_unstructured_fallback_is_labeled_and_does_not_claim_parent_phase():
    out = w._apply_state_machine_validation(raw_frame(structure=False), {'timeframe': '1h'})
    assert bool(out.wyckoff_spring_a.iloc[63])
    assert out.wyckoff_spring_a_evidence_status.iloc[63] == 'unstructured'
    assert out.wyckoff_phase_dir.iloc[63] == 'neutral'


def test_adapter_candidate_cannot_attach_to_parent_born_later():
    df = raw_frame(structure=False)
    df.loc[df.index[61], 'wyckoff_sc'] = True
    df.loc[df.index[61], ['low', 'close', 'volume_z']] = [100, 102, 3]
    df.loc[df.index[62], 'wyckoff_ar'] = True
    df.loc[df.index[62], ['low', 'close']] = [105, 109]
    out = w._apply_state_machine_validation(df, {'timeframe': '1h'})
    assert not out.wyckoff_spring_a.iloc[63]
    assert out.wyckoff_spring_a_evidence_status.iloc[63] == 'rejected'


def test_rejected_spring_does_not_erase_independent_pending_climax():
    sm = parent()
    item = evidence(sm, candidate_extreme=101.)  # Invalid sweep, independently rejected.
    valid, _ = sm.process_bar(10, candle(100, 102, 110, 3),
                              {'spring_a': True, 'bc': True},
                              event_metadata={'spring_a': item})
    assert not valid['spring_a']
    # An opposing raw climax no longer has immediate replacement authority.
    assert not valid['bc']
    assert sm.pending_opposing_climax.event_type == 'bc'
    assert sm.get_phase_dir() == 'A_accum'


def test_metadata_and_events_are_prefix_invariant_even_after_future_gap():
    df = raw_frame()
    full = w._apply_state_machine_validation(df.copy(), {'timeframe': '1h'})
    assert 'wyckoff_spring_a_available_at' in full
    prefix = w._apply_state_machine_validation(df.iloc[:64].copy(), {'timeframe': '1h'})
    pd.testing.assert_frame_equal(full.iloc[:64], prefix)
    later = df.iloc[[-1]].copy()
    later.index = later.index + pd.Timedelta('2h')
    extended = w._apply_state_machine_validation(pd.concat([df, later]), {'timeframe': '1h'})
    pd.testing.assert_frame_equal(extended.iloc[:len(df)], full, check_freq=False)


def test_gap_between_candidate_and_confirmation_rejects_delayed_event():
    df = raw_frame()
    df.loc[df.index[63], 'low'] = 98.0
    shifted = list(df.index)
    shifted[61:] = [ts + pd.Timedelta('1h') for ts in shifted[61:]]
    df.index = pd.DatetimeIndex(shifted)
    out = w._apply_state_machine_validation(df, {'timeframe': '1h'})
    assert not out.wyckoff_spring_a.iloc[63]


def test_accepted_either_spring_can_advance_state_with_legacy_callers():
    sm = parent()
    valid, _ = sm.process_bar(9, candle(100.5, 104), {'spring_a': True, 'spring_b': True})
    assert not valid['spring_a']
    assert valid['spring_b']
    assert sm.state == w.WyckoffState.ACCUM_SPRING


def test_revalidating_frame_overwrites_provenance_without_duplicate_columns():
    first = w._apply_state_machine_validation(raw_frame(), {'timeframe': '1h'})
    second = w._apply_state_machine_validation(first.copy(), {'timeframe': '1h'})
    assert second.columns.is_unique
    pd.testing.assert_frame_equal(first, second)


def test_spring_b_same_bar_and_utad_share_causal_candidate_conventions():
    df = raw_frame('spring_b')
    df.loc[df.index[60], 'close'] = 105.
    cfg = {'timeframe': '1h', 'spring_b_recovery_bars': 1}
    df['wyckoff_spring_b'], df['wyckoff_spring_b_confidence'] = w.detect_spring_type_b(df, cfg)
    out = w._apply_state_machine_validation(df, cfg)
    assert out.wyckoff_spring_b.iloc[60]
    assert out.wyckoff_spring_b_candidate_index.iloc[60] == 60
    assert out.wyckoff_spring_b_available_at.iloc[60] == '2026-01-03T13:00:00+00:00'
    df = raw_frame('ut')
    df['wyckoff_utad'] = df.wyckoff_ut
    df['wyckoff_utad_confidence'] = df.wyckoff_ut_confidence
    out = w._apply_state_machine_validation(df, {'timeframe': '1h'})
    assert out.wyckoff_ut.iloc[63] and out.wyckoff_utad.iloc[63]
    assert out.wyckoff_ut_candidate_parent_id.iloc[63] == out.wyckoff_utad_candidate_parent_id.iloc[63]
