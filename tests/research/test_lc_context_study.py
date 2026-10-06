import importlib

import pandas as pd
import pytest


def api():
    return importlib.import_module('scripts.research.lc_context_study')


def sample(values):
    cases, rows, marks = [], [], []
    for i, value in enumerate(values):
        t = pd.Timestamp('2024-01-01T00:00Z')+pd.Timedelta(days=i*32)
        cid = str(i)
        cases.append({'candidate_id': cid, 'decision_time': t.isoformat()})
        position = None if value in (0, None) else {'entry_time': (t+pd.Timedelta('2min')).isoformat(),
            'exit_time': (t+pd.Timedelta('3min')).isoformat(), 'initial_risk': 100., 'net_r': value/100.}
        rows.append({'candidate_id': cid, 'decision_time': t.isoformat(), 'status':
                     'unknown' if value is None else 'closed' if position else 'not_entered',
                     'net_pnl': value, 'position': position, 'reason': 'fixture'})
        if position:
            for at, amount, kind in [(position['entry_time'], -2., 'entry'),
                                     (position['exit_time'], value, 'exit')]:
                marks.append({'candidate_id': cid, 'available_at': at,
                              'liquidation_value': amount, 'kind': kind})
    return cases, {'rows': rows, 'marks': marks}


def test_full_raw_denominator_calendar_and_cost_r():
    cases, book = sample([10., -20., 0., 0.])
    result = api().summarize(book, cases)
    assert result['candidate_count'] == 4
    assert result['filled_count'] == 2
    assert result['net_pnl'] == -10.
    assert result['net_dollars_per_candidate'] == -2.5
    assert result['net_r_per_candidate'] == pytest.approx(-.025)
    assert result['max_drawdown_dollars'] == 20.
    assert len(result['calendar']) == 32
    assert sum(r['net_pnl'] for r in result['calendar']) == -10.
    assert sum(r['candidate_count'] for r in result['calendar']) == 4
    assert result['filled_months'] == 2
    assert result['exposure_seconds'] == 120.


def test_mtm_initial_zero_peak_counts_entry_cost_drawdown():
    cases, book = sample([1.])
    assert api().summarize(book, cases)['max_drawdown_dollars'] == 2.


def test_pairing_includes_rejections_and_winner_loser_accounting():
    cases, base = sample([100., -100., 0., 20.])
    _, context = sample([0., 0., -10., 40.])
    result = api().paired(cases, base, context)
    assert result['estimate_dollars_per_candidate'] == 2.5
    assert len(result['months']) == 32
    assert result['winners_preserved'] == 1
    assert result['winners_missed'] == 1
    assert result['losers_avoided'] == 1
    assert result['losses_introduced_from_nonloss'] == 1
    assert result['bootstrap_draws'] == 5000
    assert result['bootstrap_seed'] == 20261002
    assert result['lower'] <= result['estimate_dollars_per_candidate'] <= result['upper']
    assert api().paired(cases, base, context) == result


def test_unknown_economics_blocks_totals_and_paired_intervals():
    cases, base = sample([100., -100.])
    _, context = sample([20., None])
    result = api().summarize(context, cases)
    assert result['known_net_subtotal'] == 20.
    assert result['net_pnl'] is None
    assert result['max_drawdown_dollars'] is None
    result = api().paired(cases, base, context)
    assert result['lower'] is None
    assert result['estimate_dollars_per_candidate'] is None


def test_dropped_duplicate_or_foreign_ids_cannot_shrink_denominator():
    cases, book = sample([100., -100.])
    with pytest.raises(ValueError, match='denominator'):
        api().summarize(dict(book, rows=book['rows'][:1]), cases)
    with pytest.raises(ValueError):
        api().summarize(dict(book, rows=book['rows']+[book['rows'][0]]), cases)
    with pytest.raises(ValueError):
        api().paired(cases, book, dict(book, rows=book['rows'][:1]))


def screen(**kwargs):
    values = dict(primary_net=100., estimate=1., lower=.1, fills=50, filled_months=12,
                  stress_nets=[100., 80., 50., 30.], complete=True)
    values.update(kwargs)
    return api().verdict(**values)


@pytest.mark.parametrize('change,want', [({'primary_net': -1.}, 'unsupported'),
    ({'primary_net': 0.}, 'inconclusive'), ({'fills': 49}, 'inconclusive'),
    ({'filled_months': 11}, 'inconclusive'), ({'estimate': 0.}, 'inconclusive'),
    ({'lower': 0.}, 'inconclusive'), ({'lower': None}, 'inconclusive'),
    ({'stress_nets': [1., 1., -1., 1.]}, 'inconclusive'),
    ({'complete': False}, 'inconclusive'), ({}, 'worth_forward_test')])
def test_every_forward_screen_condition(change, want):
    assert screen(**change)['verdict'] == want


def test_empty_or_no_fill_book_is_not_evidence_of_edge():
    cases, book = sample([0., 0.])
    assert api().summarize(book, cases)['net_pnl'] == 0.
    assert screen(primary_net=0., fills=0, filled_months=0)['verdict'] == 'inconclusive'
    result = api().paired([], {'rows': []}, {'rows': []})
    assert result['estimate_dollars_per_candidate'] is None
    assert result['undefined_draws'] == 5000


def test_review_requires_verified_source_and_current_code(tmp_path):
    from scripts.research.lc_context_contract import seal
    from scripts.research.lc_context_source import sha
    source = tmp_path/'source'; source.mkdir()
    receipt = source/'receipt.json'; receipt.write_text('{}')
    code = tmp_path/'code.py'; code.write_text('v1')
    review = {'decision': 'GO', 'stage': 'lc_context_economics',
              'source_receipt_sha256': sha(receipt), 'files': {str(code): sha(code)},
              'software_review': 'independent', 'quant_review': 'independent',
              'execution_authorized': False}
    api().validate_review(review, source)
    code.write_text('v2')
    with pytest.raises(ValueError, match='changed'):
        api().validate_review(review, source)
    with pytest.raises(ValueError):
        api().validate_review(dict(review, decision='WAIT'), source)
