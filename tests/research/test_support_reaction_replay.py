from copy import deepcopy
import importlib
import importlib.util

import pandas as pd
import pytest

from scripts.research.support_reaction import assess, origin_record
from scripts.research.thesis_contract import event, protocol, seal, signed
from scripts.research.thesis_execution import replay_book
from scripts.research.thesis_sequence import compile_episode
from tests.research.support_reaction_fixtures import fixture


def api():
    assert importlib.util.find_spec('scripts.research.support_reaction_replay'), 'replay adapter missing'
    return importlib.import_module('scripts.research.support_reaction_replay')


def sample(**kwargs):
    raw, bars = fixture(**kwargs)
    r = assess(origin_record(raw), bars, as_of=raw['deadline'])
    return r, raw, bars


def test_a_is_exact_unchanged_fixed_runtime_and_all_arms_keep_raw_id():
    r, raw, bars = sample()
    a = api().replay([r], [raw], bars, arm='A')
    assert a['runtime'] == replay_book([raw], bars, 'thesis', 'fixed', capacity=False)
    assert a['runtime_policy_seal'] == seal(protocol())
    assert a['policy_seal'] != a['runtime_policy_seal']
    assert a['rows'][0]['episode_id'] == raw['id']
    assert a['rows'][0]['net'] == 0.


def test_fixed_entry_uses_original_stop_delay_and_cost_inclusive_risk():
    r, raw, bars = sample()
    bars.loc['2024-01-02T06:12Z', 'high'] = 140.
    out = api().replay([r], [raw], bars, arm='B')
    p = out['runtime']['positions'][raw['id']]
    assert p['entry_time'] == '2024-01-02T06:11:00+00:00'
    assert p['original_stop'] == 97.
    assert p['target'] == pytest.approx(136.6)
    assert p['original_qty'] == pytest.approx(100/(13.2+.0006*(110.2+97)))
    assert out['rows'][0]['reason'] == 'target'
    assert out['admission'][raw['id']]['status'] == 'allowed'
    assert out['admission'][raw['id']]['at'] == p['entry_time']
    assert out['rows'][0]['net'] > 0
    assert raw['entry_intents']['thesis'] is None


def test_unknown_volume_is_not_an_avoided_loss():
    r, raw, bars = sample(missing_volume=True)
    c = api().replay([r], [raw], bars, arm='C')
    assert c['rows'][0]['status'] == 'unknown'
    assert c['rows'][0]['net'] is None
    assert c['runtime']['positions'] == {}


def test_admission_gap_can_remove_room_but_preserves_pending_clock():
    r, raw, bars = sample()
    bars.loc['2024-01-02T06:11Z', ['open', 'high', 'low', 'close']] = [125., 126., 124., 125.]
    prefix = api().replay([r], [raw], bars, arm='B', as_of='2024-01-02T06:10Z')
    assert prefix['rows'][0]['status'] == 'pending'
    assert prefix['admission'][raw['id']]['status'] == 'pending'
    out = api().replay([r], [raw], bars, arm='B')
    assert out['rows'][0]['reason'] == 'admission_insufficient_room'
    assert out['rows'][0]['net'] == 0.
    assert out['admission'][raw['id']]['at'] == '2024-01-02T06:11:00+00:00'


def test_future_extremes_of_current_admission_bar_cannot_cancel_past_fill():
    r, raw, bars = sample()
    a = api().replay([r], [raw], bars, arm='B', as_of='2024-01-02T06:11Z')
    bars.loc['2024-01-02T06:11Z', ['high', 'low', 'close']] = [999., 1., 500.]
    bars.loc['2024-01-02T06:12Z':, ['high', 'close']] = float('nan')
    b = api().replay([r], [raw], bars, arm='B', as_of='2024-01-02T06:11Z')
    assert a == b
    assert b['rows'][0]['status'] == 'open'


@pytest.mark.parametrize('change,want', [('support', 'admission_support_failed'), ('missing', 'unknown_admission_price')])
def test_completed_pending_bar_can_cancel_or_make_unknown(change, want):
    r, raw, bars = sample()
    if change == 'support': bars.loc['2024-01-02T06:09Z', 'low'] = 106.
    else: bars = bars.drop(pd.Timestamp('2024-01-02T06:09Z'))
    out = api().replay([r], [raw], bars, arm='B')
    assert out['rows'][0]['reason'] == want
    expected = '2024-01-02T06:10:00+00:00' if change == 'support' else '2024-01-02T06:09:00+00:00'
    assert out['admission'][raw['id']]['at'] == expected
    assert out['rows'][0]['net'] == (0. if change == 'support' else None)


def test_pending_future_cancellation_does_not_free_capacity_early():
    r, raw, bars = sample()
    base = deepcopy(raw)
    base['parent']['id'] = 'second-parent'
    second = compile_episode(base, raw['events'])
    # Two valid parent lineages share a decision clock. The first reserves;
    # its cancellation two minutes later cannot retroactively free that slot.
    sr = assess(origin_record(second), bars, as_of=raw['deadline'])
    bars.loc['2024-01-02T06:11Z', ['open', 'high', 'low', 'close']] = [125., 126., 124., 125.]
    out = api().replay([r, sr], [raw, second], bars, arm='B', capacity=True)
    rows = {x['episode_id']: x for x in out['rows']}
    first, later = sorted([raw['id'], second['id']])
    assert rows[first]['reason'] == 'admission_insufficient_room'
    assert rows[later]['status'] == 'busy'


def test_foreign_origin_record_is_not_mapped_to_existing_runtime_packet():
    r, raw, bars = sample()
    other = deepcopy(raw); other['parent']['range_high'] = 151.
    other = signed(other)
    with pytest.raises(ValueError): api().replay([r], [other], bars, arm='B')


def test_assessment_cannot_be_replayed_on_changed_predecision_prices():
    r, raw, bars = sample()
    bars.loc['2024-01-02T06:08Z', 'close'] = 109.8
    with pytest.raises(ValueError, match='source'):
        api().replay([r], [raw], bars, arm='B')


def test_missing_completed_close_preserves_earlier_open_fill_and_unknown_clock():
    r, raw, bars = sample()
    bars.loc['2024-01-02T06:11Z', 'close'] = float('nan')
    at_open = api().replay([r], [raw], bars, arm='B', as_of='2024-01-02T06:11Z')
    at_close = api().replay([r], [raw], bars, arm='B', as_of='2024-01-02T06:12Z', capacity=True)
    assert at_open['rows'][0]['status'] == 'open'
    assert at_close['runtime']['entry_tape'] == at_open['runtime']['entry_tape']
    assert at_close['runtime']['positions'][raw['id']]['unknown_at'] == '2024-01-02T06:12:00+00:00'
    assert at_close['runtime']['positions'][raw['id']]['cashflows'][0]['kind'] == 'entry'
    assert at_close['runtime']['unknown_occupancy'] is True
    assert at_close['rows'][0]['net'] is None


def test_missing_open_is_unknown_at_opening_not_later_close():
    r, raw, bars = sample()
    bars.loc['2024-01-02T06:11Z', 'open'] = float('nan')
    out = api().replay([r], [raw], bars, arm='B', as_of='2024-01-02T06:12Z')
    assert out['rows'][0]['net'] is None
    assert not out['runtime']['entry_tape']
    assert out['admission'][raw['id']]['at'] == '2024-01-02T06:11:00+00:00'


def test_comparison_has_test_fold_metrics_with_reset_occupied_books():
    r, raw, bars = sample()
    out = api().compare([r], [raw], bars)
    assert len(out['chronological_reports']) == 3
    for report in out['chronological_reports']:
        assert report['test_raw_episodes'] == 0
        assert report['occupied_boundary'] == 'reset_to_empty_at_test_window'
        assert report['arms']['B']['raw_episodes'] == 0
        assert report['pairs']['B_to_C']['complete_pairs'] == 0


def test_test_fold_excludes_boundary_overlap_from_actual_metrics():
    def shift_sample(at):
        raw, bars = fixture(); delta = pd.Timestamp(at)-pd.Timestamp(raw['origin']['available_at'])
        shifted = lambda e: event(e['kind'], e['timeframe'], pd.Timestamp(e['start'])+delta,
                                 pd.Timestamp(e['end'])+delta, e['payload'], stream_id=e['stream_id'])
        base = deepcopy(raw); base['origin'] = shifted(raw['origin'])
        base['parent']['available_at'] = (pd.Timestamp(base['parent']['available_at'])+delta).isoformat()
        base['parent']['id'] = str(at)
        p = compile_episode(base, [shifted(e) for e in raw['events']])
        bars.index += delta
        # Close the synthetic position quickly; no multi-month empty span costs.
        bars.loc[pd.Timestamp(at)+pd.Timedelta('2h12min'), 'high'] = 140.
        r = assess(origin_record(p), bars, as_of=p['deadline'])
        return r, p, bars
    included = shift_sample('2025-04-23T04:00Z')
    censored = shift_sample('2025-04-28T04:00Z')
    bars = pd.concat([included[2], censored[2]])
    bars = bars.loc[~bars.index.duplicated(keep='last')].sort_index()
    out = api().compare([included[0], censored[0]], [included[1], censored[1]], bars)
    first = out['chronological_reports'][0]
    assert out['arms']['B']['raw_episodes'] == 2
    assert first['test_raw_episodes'] == 1
    assert first['split']['excluded'][censored[1]['id']] == 'test_label_outside_window'
    assert first['arms']['B']['closed_fills'] == 1
    assert first['occupied_arms']['B']['closed_fills'] == 1
    assert first['pairs']['B_to_C']['complete_pairs'] == 1


def test_stress_costs_delay_and_equal_b_c_entries_when_evidence_allows():
    r, raw, bars = sample()
    books = [api().replay([r], [raw], bars, arm=a, execution=protocol()['stress'], as_of='2024-01-02T06:13Z') for a in 'BC']
    assert books[0]['runtime']['entry_tape'] == books[1]['runtime']['entry_tape']
    assert books[0]['runtime']['entry_tape'][0]['at'] == '2024-01-02T06:12:00+00:00'
    assert books[0]['runtime']['entry_tape'][0]['quantity'] == pytest.approx(100/(13.2+.0012*(110.2+97.)))


def test_exact_room_boundary_is_also_admitted_at_delayed_open():
    r, raw, bars = sample(ceiling=136.6)
    out = api().replay([r], [raw], bars, arm='B', as_of='2024-01-02T06:11Z')
    assert out['rows'][0]['status'] == 'open'


def test_paired_attribution_reports_unknown_separately_and_never_claims_edge():
    r, raw, bars = sample(missing_volume=True)
    bars.loc['2024-01-02T06:12Z', 'high'] = 140.
    out = api().compare([r], [raw], bars)
    assert out['raw_episodes'] == 1
    assert out['edge_demonstrated'] is False
    assert out['arms']['C']['complete_net'] is None
    assert out['pairs']['B_to_C']['unknown_pairs'] == 1
    assert out['pairs']['B_to_C']['winners_missed'] == 0


def test_chronological_folds_purge_maximum_horizon_not_realized_exit():
    raw, _ = fixture()
    origins = []
    for i, at in enumerate(['2024-08-24T00:00Z', '2024-08-25T00:00Z', '2024-09-01T00:00Z', '2025-04-28T00:00Z']):
        o = origin_record(raw)
        o['id'] = str(i); o['origin']['available_at'] = at
        origins.append(o)
    split = api().folds(origins)[0]
    assert split['train_ids'] == ['0']
    assert split['test_ids'] == ['2']
    assert split['excluded']['1'] == 'training_label_overlap_or_gap'
    assert split['excluded']['3'] == 'test_label_outside_window'
    assert split['fitted'] is False
