"""Four independent books; unavailable judgment must never look like profit."""
import importlib

import pandas as pd
import pytest

from tests.research.test_lc_single_campaign import prepared, api as campaign_api, delivery, metadata
from tests.research.test_lc_published_assessment import raw_answer


def api():
    name = 'scripts.research.lc_single_accounting'
    assert importlib.util.find_spec(name), 'single accounting missing'
    return importlib.import_module(name)


def prices():
    return pd.DataFrame(dict(open=105., high=106., low=104., close=105.),
        index=pd.date_range('2026-01-01T04:00Z', '2026-01-02T04:00Z', freq='min'))


def test_unlocked_outcomes_cannot_be_scored(prepared):
    path, _ = prepared
    with campaign_api().SingleCampaign(path) as run:
        with pytest.raises(ValueError, match='terminal'):
            api().score_single_campaign(run, prices())


@pytest.mark.parametrize('plan,interpretation,expected', [
    ('enter', 'support', -60.), ('reject', 'oppose', 0.),
    (None, 'uncertain', None),
])
def test_literal_costs_nulls_and_unreviewed_label(prepared, plan, interpretation, expected):
    path, source = prepared; clock = [0]
    with campaign_api().SingleCampaign(path, monotonic_ns=lambda: clock[0]) as run:
        run.authorize('test'); reservation = run.reserve('LC1')
        raw = raw_answer(source, reservation['wrapper']['request'],
                         plan_id=plan, interpretation=interpretation).encode()
        clock[0] += 90_000_000_000
        run.capture('LC1', raw, metadata(), delivery(reservation)); run.lock_terminals()
        result = api().score_single_campaign(run, prices())
    primary = result['scenarios']['fixed_12bps_90s']['arms']
    assert primary['immediate']['summary']['policy_net_pnl'] == -60.
    assert primary['mechanical_wait']['summary']['policy_net_pnl'] == 0.
    assert primary['reject_all']['summary']['policy_net_pnl'] == 0.
    assert primary['agent']['summary']['policy_net_pnl'] == expected
    assert result['review_status'] == 'unreviewed'
    assert result['execution_authorized'] is False
    assert len(result['scenarios']) == 8
    if expected is None:
        assert result['reliability']['unavailable_count'] == 1
    else:
        assert result['reliability']['usable_count'] == 1


def test_missing_outcome_bars_do_not_turn_into_zero(prepared):
    path, source = prepared
    with campaign_api().SingleCampaign(path) as run:
        run.authorize('test'); reservation = run.reserve('LC1')
        run.capture('LC1', raw_answer(source, reservation['wrapper']['request']).encode(),
                    metadata(), delivery(reservation)); run.lock_terminals()
        result = api().score_single_campaign(run, prices().iloc[:1])
    assert result['scenarios']['fixed_12bps_90s']['arms']['agent']['summary']['policy_net_pnl'] is None


def test_instant_winner_still_has_entry_fee_drawdown(prepared):
    path, source = prepared; bars = prices()
    bars.loc['2026-01-01T04:02Z', ['high', 'close']] = [120., 116.]
    with campaign_api().SingleCampaign(path) as run:
        run.authorize('test'); reservation = run.reserve('LC1')
        run.capture('LC1', raw_answer(source, reservation['wrapper']['request']).encode(),
                    metadata(), delivery(reservation)); run.lock_terminals()
        result = api().score_single_campaign(run, bars)
    book = result['scenarios']['fixed_12bps_90s']['arms']['immediate']
    assert book['summary']['policy_net_pnl'] > 0
    assert book['mtm']['max_drawdown_dollars'] == 60.
