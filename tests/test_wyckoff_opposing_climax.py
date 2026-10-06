"""Causal parent replacement contracts, not a profitability/schematic benchmark."""
import json
import pandas as pd
import pytest

from engine.wyckoff import events as w


ORIGIN = pd.Timestamp('2026-02-01T00:00Z')


def row(i, low=212., close=224., high=238., z=0., side='bc'):
    values = dict(open=close, low=low, close=close, high=high, volume_z=z,
                  timestamp=ORIGIN + pd.Timedelta(hours=i),
                  available_at=ORIGIN + pd.Timedelta(hours=i + 1),
                  sc_confidence=.73, bc_confidence=.73)
    if side == 'sc':
        values.update(open=440 - close, close=440 - close,
                      low=440 - high, high=440 - low)
    return values


def established(side='bc', cfg=None):
    sm = w.WyckoffStateMachine(cfg or {})
    first, reaction = ('sc', 'ar') if side == 'bc' else ('bc', 'as')
    sm.process_bar(0, row(0, 200, 204, 209, 4, side), {first: True})
    sm.process_bar(2, row(2, 218, 237, 240, 0, side), {reaction: True})
    return sm


def clue(sm, side='bc', i=7, extra=None):
    raw = {side: True, **(extra or {})}
    result = sm.process_bar(i, row(i, 232, 246, 250, 3, side), raw)
    assert raw == {side: True, **(extra or {})}, 'do not mutate caller evidence'
    return result


@pytest.mark.parametrize('side,strength,context,state', [
    ('bc', 'sos', 'accumulation', w.WyckoffState.ACCUM_SOS),
    ('sc', 'sow', 'distribution', w.WyckoffState.DISTRIB_SOW),
])
def test_opposing_clue_preserves_parent_and_independent_strength(side, strength, context, state):
    sm = established(side)
    before = sm.parent_snapshot()
    valid, _ = clue(sm, side, extra={strength: True})
    assert not valid[side]
    assert valid[strength]
    assert sm.parent_snapshot() == before
    assert sm.context.value == context and sm.state == state
    assert sm.pending_opposing_climax.candidate_index == 7
    assert sm.climax_evidence[-1]['status'] == 'pending'


@pytest.mark.parametrize('side,reaction,context,state', [
    ('bc', 'as', 'distribution', w.WyckoffState.DISTRIB_AR),
    ('sc', 'ar', 'accumulation', w.WyckoffState.ACCUM_AR),
])
def test_later_reaction_confirms_atomically_at_availability(side, reaction, context, state):
    sm = established(side)
    old = sm.parent_snapshot()['id']
    clue(sm, side)
    # Competing follow-on labels must not cascade through the newly born parent.
    valid, _ = sm.process_bar(9, row(9, 224, 228, 240, 0, side),
                              {reaction: True, 'sos': True, 'sow': True,
                               'lps': True, 'lpsy': True, 'st': True})
    assert valid[side] and valid[reaction]
    assert not any(valid[k] for k in ('sos', 'sow', 'lps', 'lpsy', 'st'))
    assert sm.context.value == context and sm.state == state
    assert sm.parent_snapshot()['id'] == old + 1
    ref = sm.range_ref
    if side == 'bc':
        assert (ref.bc_bar_idx, ref.bc_high, ref.as_bar_idx, ref.as_low) == (7, 250, 9, 224)
    else:
        assert (ref.sc_bar_idx, ref.sc_low, ref.ar_bar_idx, ref.ar_high) == (7, 190, 9, 216)
    assert sm.pending_opposing_climax is None
    record = sm.climax_evidence[-1]
    assert record['status'] == 'confirmed' and record['candidate_parent_id'] == old
    assert record['candidate_timestamp'] == '2026-02-01T07:00:00+00:00'
    assert record['candidate_available_at'] == '2026-02-01T08:00:00+00:00'
    assert record['available_at'] == '2026-02-01T10:00:00+00:00'
    assert record['confidence'] == .73


@pytest.mark.parametrize('side,reaction', [('bc', 'as'), ('sc', 'ar')])
@pytest.mark.parametrize('mode', ['same_bar', 'reaction_only', 'cross_only', 'touch'])
def test_confirmation_requires_both_later_reaction_and_strict_close_cross(side, reaction, mode):
    sm = established(side)
    before = sm.parent_snapshot()
    clue(sm, side, extra={reaction: True} if mode == 'same_bar' else None)
    if mode != 'same_bar':
        close = 228 if mode == 'cross_only' else 232 if mode == 'touch' else 235
        valid, _ = sm.process_bar(8, row(8, 226, close, 240, 0, side),
                                  {} if mode == 'cross_only' else {reaction: True})
        assert not valid.get(side, False)
    assert sm.parent_snapshot() == before
    assert sm.pending_opposing_climax.candidate_index == 7


@pytest.mark.parametrize('side,reaction', [('bc', 'as'), ('sc', 'ar')])
def test_continuation_cancels_and_old_clue_cannot_later_confirm(side, reaction):
    sm = established(side)
    before = sm.parent_snapshot()
    clue(sm, side)
    sm.process_bar(8, row(8, 243, 252, 253, 0, side), {})
    assert sm.pending_opposing_climax is None
    assert sm.climax_evidence[-1]['status'] == 'cancelled_continuation'
    valid, _ = sm.process_bar(9, row(9, 224, 228, 240, 0, side), {reaction: True})
    assert not valid.get(side, False)
    assert sm.parent_snapshot() == before


@pytest.mark.parametrize('side,reaction', [('bc', 'as'), ('sc', 'ar')])
def test_repeated_clues_cannot_move_anchor_or_extend_expiry(side, reaction):
    sm = established(side, {'sm_ar_max_bars': 3})
    clue(sm, side)
    sm.process_bar(9, row(9, 230, 246, 254, 3, side), {side: True})
    assert sm.pending_opposing_climax.candidate_index == 7
    assert sm.pending_opposing_climax.high == (250 if side == 'bc' else 208)
    valid, _ = sm.process_bar(11, row(11, 224, 228, 240, 0, side), {reaction: True})
    assert not valid.get(side, False)
    assert sm.pending_opposing_climax is None
    assert sm.climax_evidence[-1]['status'] == 'expired'


@pytest.mark.parametrize('side,reaction', [('bc', 'as'), ('sc', 'ar')])
def test_confirmation_at_deadline_allowed(side, reaction):
    sm = established(side, {'sm_ar_max_bars': 3})
    clue(sm, side)
    valid, _ = sm.process_bar(10, row(10, 224, 228, 240, 0, side), {reaction: True})
    assert valid[side]


@pytest.mark.parametrize('side', ['bc', 'sc'])
@pytest.mark.parametrize('terminal', ['cancel', 'expire'])
def test_new_clue_after_cancellation_or_expiry_keeps_both_provenance_records(side, terminal):
    sm = established(side, {'sm_ar_max_bars': 3})
    clue(sm, side)
    i = 8 if terminal == 'cancel' else 11
    sm.process_bar(i, row(i, 242, 252 if terminal == 'cancel' else 246, 255, 3, side), {side: True})
    assert sm.pending_opposing_climax.candidate_index == i
    assert [item['status'] for item in sm.climax_evidence] == [
        'cancelled_continuation' if terminal == 'cancel' else 'expired', 'pending']
    assert [item['candidate_index'] for item in sm.climax_evidence] == [7, i]


@pytest.mark.parametrize('side,reaction', [('bc', 'as'), ('sc', 'ar')])
@pytest.mark.parametrize('loss', ['invalidation', 'age', 'reset', 'same_side'])
def test_parent_loss_clears_candidate_and_forbids_stale_confirmation(side, reaction, loss):
    sm = established(side)
    clue(sm, side)
    if loss == 'reset':
        sm.reset()
    elif loss == 'same_side':
        key = 'sc' if side == 'bc' else 'bc'
        sm.process_bar(8, row(8, 200, 204, 209, 4, side), {key: True})
    elif loss == 'age':
        sm.max_structure_bars = sm.bars_in_structure
    data = row(9, 196, 198, 232, 4, side) if loss == 'invalidation' else row(9, 224, 228, 240, 0, side)
    # A fresh opposing raw flag on the exact parent-loss bar cannot sneak through
    # the no-context initializer either.
    raw = {reaction: True, side: True} if loss in ('invalidation', 'age') else {reaction: True}
    valid, _ = sm.process_bar(9, data, raw)
    assert not valid.get(side, False)
    assert sm.pending_opposing_climax is None
    assert sm.context.value != ('distribution' if side == 'bc' else 'accumulation')


@pytest.mark.parametrize('side', ['bc', 'sc', None])
def test_dual_climax_is_ambiguous_not_order_dependent(side):
    sm = established(side) if side else w.WyckoffStateMachine({})
    if side:
        clue(sm, side)
    before = sm.parent_snapshot()
    valid, _ = sm.process_bar(8, row(8), {'sc': True, 'bc': True})
    assert not valid['sc'] and not valid['bc']
    assert sm.parent_snapshot() == before
    assert sm.pending_opposing_climax is None
    assert sm.climax_evidence[-1]['status'] == 'ambiguous'


def test_pending_bc_does_not_veto_valid_delayed_spring():
    sm = established()
    snap = sm.parent_snapshot()
    evidence = w.DelayedEventEvidence(
        'spring_a', 4, 7, 197., 200., snap['id'], snap['context'], snap['status'],
        ORIGIN + pd.Timedelta('4h'), ORIGIN + pd.Timedelta('7h'), ORIGIN + pd.Timedelta('8h'))
    valid, _ = sm.process_bar(7, row(7, 201, 218, 239, 3),
                              {'spring_a': True, 'bc': True}, {'spring_a': evidence})
    assert valid['spring_a'] and not valid['bc']
    assert sm.get_phase_dir() == 'C_accum'


@pytest.mark.parametrize('side', ['bc', 'sc'])
def test_no_parent_and_same_side_initialization_remain_available(side):
    sm = w.WyckoffStateMachine({})
    valid, _ = clue(sm, side)
    assert valid[side]
    first = sm.parent_snapshot()['id']
    valid, _ = clue(sm, side, i=8)
    assert valid[side] and sm.parent_snapshot()['id'] == first + 1


def adapter_frame(side='bc'):
    df = pd.DataFrame([row(i, side=side) for i in range(12)]).set_index('timestamp')
    df['volume'] = 1000.
    for key in w._ACCUM_EVENTS + w._DISTRIB_EVENTS:
        df[f'wyckoff_{key}'] = False
        df[f'wyckoff_{key}_confidence'] = 0.
    first, reaction = ('sc', 'ar') if side == 'bc' else ('bc', 'as')
    for i, key, values in [
        (0, first, row(0, 200, 204, 209, 4, side)),
        (2, reaction, row(2, 218, 237, 240, 0, side)),
        (7, side, row(7, 232, 246, 250, 3, side)),
        (9, 'as' if side == 'bc' else 'ar', row(9, 224, 228, 240, 0, side)),
    ]:
        for col in ('open', 'high', 'low', 'close', 'volume_z'):
            df.loc[df.index[i], col] = values[col]
        df.loc[df.index[i], f'wyckoff_{key}'] = True
        df.loc[df.index[i], f'wyckoff_{key}_confidence'] = .73
    return df


@pytest.mark.parametrize('side', ['bc', 'sc'])
def test_adapter_defers_confidence_without_backfill_and_records_identity(side):
    raw = adapter_frame(side)
    out = w._apply_state_machine_validation(raw.copy(), {'timeframe': '1h'})
    assert not out[f'wyckoff_{side}'].iloc[7]
    assert out[f'wyckoff_{side}_confidence'].iloc[7] == 0
    assert out[f'wyckoff_{side}'].iloc[9]
    assert out[f'wyckoff_{side}_confidence'].iloc[9] == .73
    assert json.loads(out.wyckoff_opposing_climax_evidence.iloc[7])[-1]['status'] == 'pending'
    evidence = json.loads(out.wyckoff_opposing_climax_evidence.iloc[9])[-1]
    assert evidence['status'] == 'confirmed' and evidence['candidate_index'] == 7
    assert evidence['candidate_parent_id'] != evidence['replacement_parent_id']
    assert evidence['available_at'] == '2026-02-01T10:00:00+00:00'
    prefix = w._apply_state_machine_validation(raw.iloc[:9].copy(), {'timeframe': '1h'})
    pd.testing.assert_frame_equal(prefix, out.iloc[:9])
    # Suppressed raw flags and their confidences must survive replay as raw
    # evidence, without treating a delayed confirmed flag as a new raw climax.
    again = w._apply_state_machine_validation(out.copy(), {'timeframe': '1h'})
    pd.testing.assert_frame_equal(out, again)


@pytest.mark.parametrize('side', ['bc', 'sc'])
def test_gap_or_missing_close_time_cannot_confirm_stored_candidate(side):
    raw = adapter_frame(side)
    times = list(raw.index)
    times[8:] = [ts + pd.Timedelta('1h') for ts in times[8:]]
    raw.index = pd.DatetimeIndex(times)
    out = w._apply_state_machine_validation(raw, {'timeframe': '1h'})
    assert not out[f'wyckoff_{side}'].iloc[9]
    assert json.loads(out.wyckoff_opposing_climax_evidence.iloc[8])[-1]['status'] == 'cancelled_time_discontinuity'


@pytest.mark.parametrize('side,reaction', [('bc', 'as'), ('sc', 'ar')])
def test_same_side_reset_takes_precedence_over_pending_confirmation(side, reaction):
    sm = established(side)
    clue(sm, side)
    key = 'sc' if side == 'bc' else 'bc'
    valid, _ = sm.process_bar(9, row(9, 200, 224, 239, 4, side), {key: True, reaction: True})
    assert valid[key] and not valid.get(side, False)
    assert sm.pending_opposing_climax is None
    assert sm.climax_evidence[-1]['status'] == 'cancelled_parent_replacement'


def test_fresh_detector_run_does_not_reuse_saved_raw_climax_inputs():
    raw = pd.DataFrame(dict(open=220., high=221., low=219., close=220., volume=1000.),
                       index=pd.date_range('2026-02-01', periods=70, freq='1h', tz='UTC'))
    cfg = {'timeframe': '1h', 'shadow_v2_enabled': False}
    clean = w.detect_all_wyckoff_events(raw.copy(), cfg)
    polluted = raw.copy()
    for key in ('sc', 'bc'):
        polluted[f'wyckoff_{key}_raw'] = True
        polluted[f'wyckoff_{key}_raw_confidence'] = .99
    fresh = w.detect_all_wyckoff_events(polluted, cfg)
    pd.testing.assert_frame_equal(clean.sort_index(axis=1), fresh.sort_index(axis=1))


def test_multiple_lifecycle_records_survive_actual_feature_numeric_fill():
    from scripts.research.live_feature_replay import LiveFeatureProcessor

    raw = adapter_frame()
    raw.loc[raw.index[8], ['open', 'low', 'close', 'high', 'volume_z']] = [252, 243, 252, 255, 3]
    raw.loc[raw.index[8], ['wyckoff_bc', 'wyckoff_bc_confidence']] = [True, .81]
    out = w._apply_state_machine_validation(raw, {'timeframe': '1h'})
    key = 'wyckoff_opposing_climax_evidence'
    filled = LiveFeatureProcessor().fc._fill_nans(pd.Series({key: out[key].iloc[8]}, dtype=object))
    assert [item['status'] for item in json.loads(filled[key])] == ['cancelled_continuation', 'pending']


@pytest.mark.parametrize('side', ['bc', 'sc'])
def test_missing_datetime_candidate_is_not_legacy_index_only_evidence(side):
    raw = adapter_frame(side)
    times = list(raw.index)
    times[7] = pd.NaT
    raw.index = pd.DatetimeIndex(times)
    out = w._apply_state_machine_validation(raw, {'timeframe': '1h'})
    assert not out[f'wyckoff_{side}'].iloc[9]
    assert out[f'wyckoff_{side}_confidence'].iloc[9] == 0.
    assert json.loads(out.wyckoff_opposing_climax_evidence.iloc[8])[-1]['status'] == 'cancelled_time_discontinuity'


@pytest.mark.parametrize('side,reaction,strength,parent,replacement', [
    ('bc', 'as', 'sos', 'accumulation', 'distribution'),
    ('sc', 'ar', 'sow', 'distribution', 'accumulation'),
])
def test_actual_raw_candles_allow_reversal_after_preserving_strength(side, reaction, strength, parent, replacement):
    """Fresh reviewer witness, no injected event flags or volume-z values."""
    idx = pd.date_range('2026-02-01', periods=65, freq='1h', tz='UTC')
    raw = pd.DataFrame(dict(open=220., high=221., low=219., close=220., volume=1000.), index=idx)
    for i, values in {
        50: (220, 221, 180, 185, 10000),
        51: (185, 201, 182, 195, 1000),
        52: (195, 206, 190, 205, 1000),
        53: (205, 217, 200, 215, 1000),
        54: (215, 222, 210, 220, 1000),
        55: (223, 250, 223, 246, 12000),
        56: (246, 247, 227, 236, 1000),
        57: (235, 239, 212, 216, 1000),
    }.items():
        raw.iloc[i] = values
    if side == 'sc':
        original = raw[['open', 'high', 'low', 'close']].copy()
        for dest, source in [('open', 'open'), ('high', 'low'), ('low', 'high'), ('close', 'close')]:
            raw[dest] = 460 - original[source]
    cfg = {'timeframe': '1h', 'shadow_v2_enabled': False}
    out = w.detect_all_wyckoff_events(raw.copy(), cfg)
    assert out[f'wyckoff_{side}_raw'].iloc[55]
    assert not out[f'wyckoff_{side}'].iloc[55]
    assert out[f'wyckoff_{side}_confidence'].iloc[55] == 0
    assert out[f'wyckoff_{strength}'].iloc[55]
    assert out.wyckoff_context.iloc[55] == parent
    assert out[f'wyckoff_{side}'].iloc[57] and out[f'wyckoff_{reaction}'].iloc[57]
    assert out.wyckoff_context.iloc[57] == replacement
    evidence = json.loads(out.wyckoff_opposing_climax_evidence.iloc[57])[-1]
    assert evidence['status'] == 'confirmed' and evidence['candidate_index'] == 55
    assert out[f'wyckoff_{side}_confidence'].iloc[57] == evidence['confidence']
    assert evidence['confidence'] == out[f'wyckoff_{side}_raw_confidence'].iloc[55]
    for length in (56, 58):
        prefix = w.detect_all_wyckoff_events(raw.iloc[:length].copy(), cfg)
        pd.testing.assert_frame_equal(prefix, out.iloc[:length])
