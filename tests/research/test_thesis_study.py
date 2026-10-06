from copy import deepcopy
import json

import pytest

from scripts.research.thesis_contract import signed
from scripts.research.thesis_study import compare, validate_source_review, new_output, reconcile
from scripts.research.run_thesis_study import main
from scripts.research.thesis_source import verify_files, sha
from tests.research.thesis_fixtures import packet, minutes


def test_four_independent_arms_and_identical_entry_attribution():
    bars = minutes(periods=250)
    bars.loc['2024-01-01T07:05Z', 'low'] = 96.
    result = compare([packet()], bars)
    assert set(result['books']) == {'simple_fixed', 'simple_adaptive', 'thesis_fixed', 'thesis_adaptive'}
    assert result['paired_entry_equality'] == {'simple': True, 'thesis': True}
    assert result['raw_episodes'] == 1
    assert all(v['raw_episodes'] == 1 for v in result['summary'].values())
    assert result['claim'] == 'engineering_only_no_edge_verdict'
    assert result['reconciliation']['positions_checked'] == 8


def test_unknown_is_null_not_zero_and_audit_catches_changed_fee():
    result = compare([packet()], minutes(periods=200))
    assert result['summary']['simple_fixed']['net'] is None
    altered = deepcopy(result['books']['simple_fixed'])
    pos = next(iter(altered['positions'].values()))
    pos['fees'] += 1.
    with pytest.raises(ValueError, match='fee'):
        reconcile(altered)


def test_source_review_and_output_guards(tmp_path):
    source = signed({'economic_outcomes_computed': False, 'issues': [], 'packets': [packet()],
                     'policy_seal': 'p', 'files': {'a': 'b'}})
    review = signed({'source_seal': source['seal'], 'source_file_sha256': 'file', 'verdict': 'clear_engineering',
                     'execution_authorized': False, 'reviewer': 'quant', 'witnesses': ['causality', 'source']})
    validate_source_review(source, review, 'file')
    bad = dict(review, source_file_sha256='different')
    with pytest.raises(ValueError):
        validate_source_review(source, signed(bad), 'file')
    out = tmp_path/'unique'
    new_output(out)
    with pytest.raises(FileExistsError):
        new_output(out)


@pytest.mark.parametrize('args', [['full', '--output', 'unused'], ['engineering', '--output', 'unused'],
                                 ['source', '--output', 'unused', '--review', 'foreign']])
def test_cli_rejects_full_campaign_and_unreviewed_economics(args):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_input_or_implementation_hash_change_is_rejected(tmp_path):
    path = tmp_path/'bound.json'
    path.write_text('{"version":1}')
    files = {str(path): sha(path)}
    verify_files(files)
    path.write_text('{"version":2}')
    with pytest.raises(ValueError, match='hash changed'):
        verify_files(files)
