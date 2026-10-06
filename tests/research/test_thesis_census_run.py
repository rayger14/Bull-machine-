"""Local runner boundary tests, with no launch of the historical campaign."""
import importlib.util
import json

import pytest

from scripts.research.thesis_contract import signed, verify
from scripts.research.thesis_source import sha


def api():
    assert importlib.util.find_spec('scripts.research.run_thesis_census'), 'bounded census launcher missing'
    from scripts.research import run_thesis_census
    return run_thesis_census


def test_output_is_exclusive_and_cumulative_cap_is_enforced(tmp_path):
    out = tmp_path/'run'
    writer = api().Output(out, limit=120, failure_reserve=20)
    writer.write('one.json', {'x': 'a'*35})
    writer.write('two.json', {'x': 'a'*35})
    with pytest.raises(ValueError, match='cap'):
        writer.write('three.json', {'x': 'a'*35})
    assert not (out/'three.json').exists()
    with pytest.raises(FileExistsError):
        api().Output(out)
    assert len(list(out.iterdir())) == 2


def test_bound_execution_saves_launch_then_failure_without_success_receipt(tmp_path):
    dependency = tmp_path/'input.json'
    dependency.write_text('{}')
    files = {str(dependency): sha(dependency)}
    def fail():
        raise ValueError('example coverage failure')
    out = tmp_path/'failure'
    with pytest.raises(ValueError, match='coverage'):
        api().bounded_stage(out, files, fail)
    launch = json.loads((out/'launch.json').read_text())
    failure = json.loads((out/'failure.json').read_text())
    verify(launch)
    verify(failure)
    assert launch['files'] == files
    assert failure['status'] == 'failed'
    assert failure['launch_seal'] == launch['seal']
    assert not (out/'receipt.json').exists()


def test_mutated_dependency_cannot_get_success_receipt(tmp_path):
    dependency = tmp_path/'input.json'
    dependency.write_text('{}')
    files = {str(dependency): sha(dependency)}
    def mutate():
        dependency.write_text('{"changed":true}')
        return {'source.json': signed({'x': 1})}
    out = tmp_path/'changed'
    with pytest.raises(ValueError, match='hash changed'):
        api().bounded_stage(out, files, mutate)
    assert (out/'failure.json').exists()
    assert not (out/'receipt.json').exists()


def test_success_receipt_binds_all_written_artifacts_and_never_authorizes_trading(tmp_path):
    dependency = tmp_path/'input.json'
    dependency.write_text('{}')
    out = tmp_path/'success'
    receipt = api().bounded_stage(out, {str(dependency): sha(dependency)},
                                  lambda: {'source.json': signed({'x': 1})})
    verify(receipt)
    assert receipt['artifacts']['source.json'] == sha(out/'source.json')
    assert receipt['artifacts']['launch.json'] == sha(out/'launch.json')
    assert receipt['economic_outcomes_computed'] is False
    assert receipt['execution_authorized'] is False
    assert receipt['peak_rss_bytes'] > 0
    assert receipt['legacy_policy_seal']


def test_cli_cannot_launch_economics_or_override_calendar(capsys):
    with pytest.raises(SystemExit) as ex:
        api().main(['--output', '/unused', '--start', '2025-01-01'])
    assert ex.value.code == 2
    assert 'unrecognized arguments' in capsys.readouterr().err


def test_january_witness_rejects_changed_packet():
    from scripts.research.thesis_census import build_census
    from tests.research.test_thesis_source import source_fixture
    bars, parents = source_fixture()
    source = build_census(bars, parents, '2024-01-01T00:00Z', '2024-01-02T00:00Z',
                          seed='2023-12-28T00:00Z', source_end='2024-01-09T00:00Z', stream='fixture')
    reference = signed(dict(source, end='2024-01-02T00:00:00+00:00'))
    assert api().january_witness(source, reference)['packets_equal'] is True
    reference['packets'][0]['original_stop'] = 1.
    reference = signed(reference)
    # Use a fresh source: avoid changing both sides of the comparison by alias.
    source = build_census(bars, parents, '2024-01-01T00:00Z', '2024-01-02T00:00Z',
                          seed='2023-12-28T00:00Z', source_end='2024-01-09T00:00Z', stream='fixture')
    with pytest.raises(ValueError, match='January'):
        api().january_witness(source, reference)


def test_real_partition_witness_compares_full_tail_packets_and_decisions():
    from scripts.research.thesis_census import build_census
    from tests.research.test_thesis_source import source_fixture
    bars, parents = source_fixture()
    source = build_census(bars, parents, '2024-01-01T00:00Z', '2024-01-03T00:00Z',
                          seed='2023-12-28T00:00Z', source_end='2024-01-09T00:00Z', stream='fixture')
    witness = api().partition_witness(bars, parents, source, '2024-01-02T00:00Z')
    assert witness['packets_equal'] is True
    assert witness['decisions_equal'] is True
    assert witness['continuation_equal'] is True
    assert witness['prefix_catalog_equal'] is True


@pytest.mark.parametrize('state', ['missing', 'changed'])
def test_production_preflight_reference_failure_is_bounded_and_recorded(tmp_path, monkeypatch, state):
    pinned = tmp_path/'reference.json'
    if state == 'changed':
        pinned.write_text('{}')
    module = api()
    monkeypatch.setattr(module, 'REFERENCE_FILES', {str(pinned): '0'*64})
    out = tmp_path/'preflight'
    with pytest.raises((FileNotFoundError, ValueError)):
        module.run(out)
    assert (out/'launch.json').exists()
    failure = json.loads((out/'failure.json').read_text())
    verify(failure)
    assert failure['status'] == 'failed'
    assert not (out/'receipt.json').exists()
