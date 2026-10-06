from copy import deepcopy

import pytest

from scripts.research.study_report import (campaign_decision, paired_interval, paired_months,
                                           render_report, summarize_book)


def ops(count=10):
    return [{'id': str(i), 'origin_time': '2024-01-31T23:00:00Z', 'family': 'R3',
             'instrument': 'BTC', 'data_stream_id': 'fixture'} for i in range(count)]


def book(count=10, pnl=100):
    return {'rows': [{'opportunity_id': str(i), 'net_pnl': pnl if i == 0 else 0,
                      'status': 'closed' if i == 0 else 'not_entered', 'reason': 'target' if i == 0 else 'no_signal',
                      'position': {'entry_time': '2024-01-31T23:01:00Z', 'exit_time': '2024-02-01T02:00:00Z',
                                   'entry_price': 100, 'exit_price': 102, 'quantity': 50, 'initial_risk': 100,
                                   'entry_fee': 1, 'exit_fee': 1, 'funding': 1} if i == 0 else None}
                     for i in range(count)], 'marks': [], 'events': [], 'blockers': []}


def test_common_denominator_includes_nonentries_and_origin_not_exit_month():
    monthly = paired_months(ops(), book(pnl=50), book(pnl=100), ['2024-01', '2024-02'])
    assert monthly[0]['opportunity_count'] == 10
    assert monthly[0]['repair_net'] == 100
    assert monthly[0]['baseline_net'] == 50
    assert monthly[1]['opportunity_count'] == 0
    assert monthly[1]['repair_net'] == 0
    interval = paired_interval(monthly)
    assert interval['estimate'] == pytest.approx(.05)
    assert interval['undefined_draws'] > 0
    assert interval['undefined_limit_failed'] is True


def test_zero_months_preserved_and_bootstrap_recomputes_pooled_ratio():
    monthly = [{'month': 'm' + str(i), 'opportunity_count': 1 if i % 2 == 0 else 9,
                'baseline_net': 0., 'repair_net': 100. if i % 2 == 0 else 0., 'unresolved_count': 0}
               for i in range(32)]
    result = paired_interval(monthly)
    assert result['estimate'] == pytest.approx(.1)
    assert result['draws'] == 5000
    assert result['seed'] == 20260930
    assert result['undefined_draws'] == 0
    assert result == paired_interval(monthly)
    assert result['quantiles'] == [0.0083333333, 0.9916666667]


def test_unknown_pair_cannot_be_zero_or_partial_inference():
    baseline, repair = book(), book()
    repair['rows'][4].update(status='unknown', net_pnl=None)
    monthly = paired_months(ops(), baseline, repair, ['2024-01', '2024-02'])
    assert monthly[0]['unresolved_count'] == 1
    assert monthly[0]['repair_net'] is None
    assert paired_interval(monthly)['status'] == 'blocked'


def test_duplicate_missing_foreign_or_outside_calendar_rows_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        paired_months(ops() + [ops()[0]], book(), book(), ['2024-01'])
    missing = book()
    missing['rows'].pop()
    with pytest.raises(ValueError, match='denominator'):
        paired_months(ops(), book(), missing, ['2024-01'])
    with pytest.raises(ValueError, match='calendar'):
        paired_months(ops(), book(), book(), ['2024-02'])


def valid_summary():
    return {'required_data_ok': True, 'blockers': [], 'repair_fills': 80,
            'origin_months_with_repair_fills': 16, 'undefined_fraction': 0.,
            'primary_repair_net': 1000., 'incremental_estimate': .03,
            'scenario_repair_net': [1000., 800., 750., 600.], 'positive_blocks': 4,
            'repair_net_without_top3': 400., 'incremental_lower_bound': .005}


@pytest.mark.parametrize('changes,decision,reason', [
    ({'required_data_ok': False, 'blockers': ['missing_model']}, 'blocked', 'required_contract_failed'),
    ({'repair_fills': 49}, 'insufficient_evidence', 'repair_fill_floor'),
    ({'origin_months_with_repair_fills': 11}, 'insufficient_evidence', 'origin_month_floor'),
    ({'undefined_fraction': .011}, 'insufficient_evidence', 'undefined_bootstrap_limit'),
    ({'primary_repair_net': 0.}, 'park', 'nonpositive_repair_net'),
    ({'incremental_estimate': 0.}, 'park', 'nonpositive_increment'),
    ({'scenario_repair_net': [10., 20., 30., 0.]}, 'park', 'stress_scenario_failure'),
    ({'positive_blocks': 2}, 'park', 'period_fragility'),
    ({'repair_net_without_top3': 0.}, 'park', 'top_three_concentration'),
    ({'incremental_lower_bound': 0.}, 'insufficient_evidence', 'nonpositive_adjusted_lower_bound'),
    ({}, 'eligible_for_forward_proposal', 'all_frozen_checks_passed'),
])
def test_fixed_decision_table_order(changes, decision, reason):
    summary = dict(valid_summary(), **changes)
    result = campaign_decision(summary)
    assert result['decision'] == decision
    assert result['reason'] == reason
    assert result['execution_authorized'] is False


def test_floor_precedes_negative_performance_but_secondary_failures_retained():
    result = campaign_decision(dict(valid_summary(), repair_fills=4, primary_repair_net=-10.))
    assert result['decision'] == 'insufficient_evidence'
    assert 'nonpositive_repair_net' in result['secondary_failures']
    assert campaign_decision(dict(valid_summary(), undefined_fraction=.01))['decision'] == 'eligible_for_forward_proposal'


def test_incomplete_scenario_grid_or_nonfinite_summary_cannot_advance():
    for changes in ({'scenario_repair_net': [100.]}, {'primary_repair_net': float('nan')}, {'repair_fills': True}):
        assert campaign_decision(dict(valid_summary(), **changes))['decision'] == 'blocked'


def test_empty_qualified_population_is_insufficient_not_a_data_failure():
    interval = paired_interval(paired_months([], {'rows': []}, {'rows': []}, ['2024-01']))
    result = campaign_decision(dict(valid_summary(), repair_fills=0,
        origin_months_with_repair_fills=0, undefined_fraction=interval['undefined_fraction'],
        primary_repair_net=0., incremental_estimate=interval['estimate'],
        incremental_lower_bound=interval['lower'], scenario_repair_net=[0.] * 4,
        positive_blocks=0, repair_net_without_top3=0.))
    assert result['decision'] == 'insufficient_evidence'
    assert result['reason'] == 'repair_fill_floor'
    assert 'undefined_bootstrap_limit' in result['secondary_failures']
    assert campaign_decision(dict(valid_summary(), incremental_lower_bound=None))['decision'] == 'blocked'


def test_book_diagnostics_and_report_do_not_claim_live_readiness():
    summary = summarize_book(book(), ops())
    assert summary['raw_opportunities'] == 10
    assert summary['completed_fills'] == 1
    assert summary['net_pnl'] == 100
    assert summary['net_per_100_risk_per_opportunity'] == .1
    assert summary['origin_months_with_fills'] == 1
    report = render_report({'family': 'R3', 'decision': campaign_decision(valid_summary()),
                            'baseline': summary, 'repair': summary, 'interval': {'estimate': 0.1}})
    assert 'not live-ready' in report.lower()
    assert 'exposed' in report.lower()


def test_excursions_include_completed_entry_bar_but_exclude_exit_open_future_extremes():
    from tests.research.study_fixtures import minutes
    rows = book(1)
    position = rows['rows'][0]['position']
    position.update(entry_time='2024-01-31T23:01:00Z', exit_time='2024-01-31T23:03:00Z')
    data = minutes(4, start='2024-01-31T23:00:00Z', price=101.)
    data.iloc[0, data.columns.get_loc('high')] = 1000.
    data.iloc[1, data.columns.get_loc('high')] = 105.
    data.iloc[2, data.columns.get_loc('low')] = 97.
    data.iloc[3, data.columns.get_loc('high')] = 2000.
    data.iloc[3, data.columns.get_loc('low')] = 1.
    excursion = summarize_book(rows, ops(1), data)['excursions'][0]
    assert excursion['mfe_r_upper_bound'] == 2.5
    assert excursion['mae_r_upper_bound'] == 1.5
