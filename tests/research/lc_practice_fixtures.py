"""Synthetic on-disk practice inputs. Never imports private market archives."""
from pathlib import Path
import pandas as pd

from scripts.research.lc_judgment_runner import _save_equal, _sha, _digest
from tests.research.lc_structure_fixtures import structure_source
from tests.research.test_lc_master_assessment import inputs


def saved_inputs(tmp_path, count=2):
    paths = []
    for i in range(count):
        source = structure_source(lambda p: p.update(case_id=f'LC{i}'))
        paths.append(_save_equal(tmp_path / f'LC{i}.json', source))
    archive = tmp_path / 'minute.parquet'
    bars = inputs()[1].rename(columns={'volume': 'vol'})
    future = pd.DataFrame(dict(open=105.,high=130.,low=100.,close=111.,vol=1.),
        index=pd.date_range('2026-01-01T04:02:00Z','2026-01-02T04:16:00Z',freq='min'))
    bars = pd.concat([bars.loc[bars.index < future.index[0]], future])
    bars.index.name = 'ts'
    bars.to_parquet(archive)
    body = dict(roster=[f'LC{i}' for i in range(count)],
                files={str(p.resolve()): _sha(p) for p in paths + [archive]})
    lock = _save_equal(tmp_path / 'input_lock.json', dict(body, sha256=_digest(body)))
    return paths, lock, archive


def prepared(tmp_path, count=2):
    from scripts.research.lc_practice_runtime import prepare_run
    paths, lock, archive = saved_inputs(tmp_path, count)
    root = tmp_path / 'run'
    manifest = prepare_run(root, paths, input_lock=lock, archive=archive, limit=count)
    return root, manifest
