import pytest

from scripts.research.run_archetype_study import main


@pytest.mark.parametrize('stage', ['census', 'score'])
def test_unreviewed_expensive_stages_cannot_launch(stage, capsys):
    assert main([stage]) == 2
    assert 'review' in capsys.readouterr().err.lower()


def test_cli_requires_explicit_pilot_output():
    with pytest.raises(SystemExit):
        main(['pilot'])


def test_preflight_reports_readonly_result(monkeypatch, capsys):
    from scripts.research import run_archetype_study as cli
    monkeypatch.setattr(cli, 'preflight', lambda: {'source_ready': False, 'blockers': ['fixture']})
    assert main(['preflight']) == 2
    assert 'fixture' in capsys.readouterr().out
