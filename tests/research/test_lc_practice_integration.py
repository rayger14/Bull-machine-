"""Offline prepare -> raw capture -> lock -> real replay functions -> report."""
import importlib
import json
import subprocess
import sys

import pytest

from scripts.research.conditional_assessment import digest
from scripts.research.lc_practice_runtime import PracticeRun
from tests.research.lc_practice_fixtures import prepared
from tests.research.test_lc_practice_runtime import response, metadata


def report_api():
    name='scripts.research.lc_practice_report'
    assert importlib.util.find_spec(name), 'practice report missing'
    return importlib.import_module(name)


def test_end_to_end_captures_lock_then_prices_report_and_idempotent_rerender(tmp_path):
    from scripts.research.lc_practice_replay import score_run
    root, m=prepared(tmp_path,3); now=[0]
    with PracticeRun(root,monotonic_ns=lambda:now[0]) as run:
        with pytest.raises(ValueError,match='terminal'): score_run(run)
        run.authorize('Synthetic integration test, no external calls',m['sha256'])
        for i,decision in enumerate(['enter_proposal','reject','reject']):
            res=run.reserve(f'LC{i}'); raw,delivered=response(res,decision)
            if decision=='enter_proposal':
                answer=json.loads(raw); answer['plan']['horizon_minutes']=1440
                answer['opposing'][0]['text']='<script>alert("untrusted")</script>'
                raw=json.dumps(answer).encode()
            if i==2: raw=b'bad JSON'
            run.attach(f'LC{i}',f'agent-{i}'); now[0]+=90000000000
            run.capture(f'LC{i}',raw,metadata(f'agent-{i}'),delivered)
        run.lock_terminals(); result=score_run(run)
        assert result['cases'][0]['arms']['agent']['status']=='filled'
        assert result['cases'][0]['arms']['agent']['net_pnl']==pytest.approx(79.562520404832)
        assert result['cases'][1]['arms']['agent']['status']=='rejected'
        assert result['cases'][2]['arms']['agent']['net_pnl'] is None
        assert result['cases'][0]['arms']['legacy_immediate']['status']=='filled'
        assert result['summary']['overall']['agent']['total_net_pnl'] is None
        original=(root/'case_results.json').read_bytes()
        paths=report_api().render_report(run)
        page=paths['html'].read_text()
        assert 'LC0' in page and 'LC1' in page and 'LC2' in page
        assert 'data:image/png;base64,' in page
        assert '<script>alert(' not in page and '&lt;script&gt;' in page
        assert 'not a holdout' in page and 'excludes funding/impact' in page
        markdown=paths['markdown'].read_text()
        assert '|---|---|---:|---:|\n| LC0 |' in markdown
        assert 'Matched coverage:' in page and 'Full accounting and coverage' in page
        report_api().render_report(run)
        assert (root/'case_results.json').read_bytes()==original
        assert score_run(run)==result


def test_all_not_run_report_is_honest_and_does_not_dispatch(tmp_path):
    root,m=prepared(tmp_path,1)
    with PracticeRun(root) as run:
        run.mark_not_run('LC0','Synthetic abandoned run');run.lock_terminals()
        paths=report_api().render_report(run)
        page=paths['html'].read_text()
        assert 'not_run' in page and 'unavailable' in page
        assert not (root/'authorization.json').exists()


def test_cli_status_and_owner_stdin_bridge_require_explicit_authorization(tmp_path):
    root,m=prepared(tmp_path,1)
    module='scripts.research.lc_practice'
    assert importlib.util.find_spec(module), 'practice CLI missing'
    status=subprocess.run([sys.executable,'-m',module,'status','--run',str(root)],capture_output=True,text=True)
    assert status.returncode==0, status.stderr
    assert json.loads(status.stdout)['pending']==1
    bridge=subprocess.run([sys.executable,'-m',module,'owner','--run',str(root)],
        input=json.dumps({'op':'reserve','case_id':'LC0'})+'\n',capture_output=True,text=True)
    lines=[json.loads(line) for line in bridge.stdout.splitlines()]
    assert any('not authorized' in line.get('error','') for line in lines)
    assert not (root/'cases/000/reservation.json').exists()


@pytest.mark.parametrize('missing_at,want', [('2026-01-01T04:02:00Z','unavailable'),
                                           ('2026-01-01T04:10:00Z','filled')])
def test_nullable_future_prices_are_gaps_not_whole_run_crashes(tmp_path,missing_at,want):
    import pandas as pd
    from tests.research.lc_practice_fixtures import saved_inputs
    from scripts.research.lc_judgment_runner import _save_equal,_sha
    from scripts.research.lc_practice_runtime import prepare_run
    from scripts.research.lc_practice_replay import score_run
    paths,lock,archive=saved_inputs(tmp_path,1)
    frame=pd.read_parquet(archive).astype('Float64')
    frame.loc[missing_at,'high']=pd.NA
    frame.to_parquet(archive)
    body=dict(roster=['LC0'],files={str(p.resolve()):_sha(p) for p in paths+[archive]})
    lock=_save_equal(tmp_path/'nullable_lock.json',dict(body,sha256=digest(body)))
    root=tmp_path/'run';m=prepare_run(root,paths,input_lock=lock,archive=archive,limit=1)
    assert m['cases'][0]['source_status']=='verified'
    now=[0]
    with PracticeRun(root,monotonic_ns=lambda:now[0]) as run:
        run.authorize('Synthetic nullable-data regression',m['sha256'])
        res=run.reserve('LC0');raw,delivered=response(res,'enter_proposal')
        answer=json.loads(raw);answer['plan']['horizon_minutes']=1440;raw=json.dumps(answer).encode()
        run.attach('LC0','nullable-role');now[0]=90000000000
        run.capture('LC0',raw,metadata('nullable-role'),delivered);run.lock_terminals()
        result=score_run(run)
        assert result['cases'][0]['arms']['agent']['status']==want
        assert (result['cases'][0]['arms']['agent']['net_pnl'] is None)==(want=='unavailable')
        page=report_api().render_report(run)['html'].read_text()
        assert 'LC0' in page and 'data:image/png;base64,' in page
        invalid=[r for r in result['cases'][0]['future_minutes'] if r['high'] is None]
        assert len(invalid)==1 and pd.Timestamp(invalid[0]['open_time'])==pd.Timestamp(missing_at)
