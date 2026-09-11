"""Reproducible bounded LFC experiment. Never a profitability certificate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from functools import partial

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.research.gate_observability import load_configs
from scripts.research.live_feature_replay import run_replay
from scripts.research.engine_signal_replay import EXPECTED_ARCHETYPES, run_signal_replay
from scripts.research.replay_clock import Observation, digest, json_safe, utc, validate_bars


def file_hashes(paths):
    return {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            if Path(p).exists() else None for p in paths}


def assert_unchanged(expected):
    if file_hashes(expected) != expected:
        raise ValueError('Input, configuration or source files changed during replay')


def manifest_files(manifest):
    if not isinstance(manifest, dict):
        return {}
    if 'files' in manifest:
        return {str((ROOT/Path(path)).resolve()): value for path, value in manifest['files'].items()}
    result = {}
    for child in manifest.values():
        result.update(manifest_files(child))
    return result


def exercise(bars, observations, *, instrument, timeframe, emit_from=None,
             cuts=None, config=ROOT/'configs/champion_paper.json', progress=None,
             include_signals=False):
    step = validate_bars(bars, timeframe)
    observations = list(observations)
    cuts = sorted(set(cuts if cuts is not None else [len(bars)//2, len(bars)-1]))
    cuts = [cut for cut in cuts if 0 < cut < len(bars)]
    configs, paths = load_configs(config)
    config_hashes = file_hashes(paths)
    runner = partial(run_signal_replay, config=config) if include_signals else run_replay
    notify = progress or (lambda message: None)
    notify(f'Full replay: {len(bars)} {timeframe} bars')
    assert_unchanged(config_hashes)
    full = runner(bars, observations, instrument=instrument, timeframe=timeframe, emit_from=emit_from)
    checks = []
    for cut in cuts:
        notify(f'Prefix/restart check at bar {cut}/{len(bars)}')
        end = bars.index[cut-1] + step
        available = [record for record in observations if record.visible_at() <= end]
        prefix = runner(bars.iloc[:cut], available, instrument=instrument,
                            timeframe=timeframe, emit_from=emit_from)
        resumed = runner(bars, observations, instrument=instrument, timeframe=timeframe,
                            emit_from=emit_from, checkpoint=prefix['checkpoint'])
        before = [row for row in full['rows'] if utc(row['decision_time']) <= end]
        after = [row for row in full['rows'] if utc(row['decision_time']) > end]
        checks.append(dict(cut=cut, close_time=str(end), prefix_equal=digest(prefix['rows']) == digest(before),
                           prefix_rows_compared=len(before), restart_rows_compared=len(after),
                           contract_equal=full['contract_id'] == prefix['contract_id'] == resumed['contract_id'],
                           restart_rows_equal=digest(resumed['rows']) == digest(after),
                           restart_state_equal=digest(resumed['state']) == digest(full['state'])))
    blockers = set(full['blockers'])
    blockers.add('candle_availability_assumed_at_close')
    if not checks:
        blockers.add('no_prefix_restart_checks')
    if any(not c[k] for c in checks for k in ('prefix_equal', 'restart_rows_equal', 'restart_state_equal', 'contract_equal')):
        blockers.add('prefix_or_restart_mismatch')
    if not full['rows']:
        blockers.add('no_emitted_evidence')
    if not any(row['output']['hourly_updated'] for row in full['rows']):
        blockers.add('no_hourly_feature_evidence')
    if set(configs) != EXPECTED_ARCHETYPES:
        blockers.add('archetype_inventory_mismatch')
    assert_unchanged(config_hashes)
    assert_unchanged(manifest_files(full['source_manifest']))
    archetypes = {name: 'signal_layer_not_replayed' for name in configs}
    if include_signals:
        archetypes = {name: dict(evaluations=0, native_candidates=0, selected=0) for name in configs}
        for row in full['rows']:
            signal = row['output']['engine_signal']
            if signal is None:
                continue
            for name, outcome in signal['archetypes'].items():
                archetypes[name]['evaluations'] += 1
                archetypes[name]['native_candidates'] += int(outcome['native_signal'] is not None)
                archetypes[name]['selected'] += int(outcome['selected'])
    return dict(certified=False, scope=full['scope'], blockers=sorted(blockers),
                coverage=dict(input_bars=len(bars), emitted_rows=len(full['rows']),
                              hourly_updates=full['state']['hourly_updates'],
                              signal_engine_hours_including_warmup=(full['state']['engine_signal']['state']['bar_index']
                                                                   if include_signals else 0),
                              first_open=str(bars.index[0]), last_close=str(bars.index[-1]+step),
                              final_feature_count=len(full['state']['features'] or {})),
                assumptions=dict(startup='cold replay of supplied original prehistory',
                                 candle_availability='assumed exact bar close; receipt times unknown',
                                 deep_daily=False, external_observation_count=len(observations)),
                checks=checks, state_hash=digest(full['state']), rows_hash=digest(full['rows']),
                contract_id=full['contract_id'], source_manifest=full['source_manifest'],
                input_rows_hash=digest(dict(index=list(bars.index), rows=bars.to_dict('records'))),
                config_hashes=config_hashes, attribution_guard='before/after configuration and manifest source hashes',
                inventory=dict(expected=sorted(EXPECTED_ARCHETYPES), actual=sorted(configs)),
                archetype_count_scope='emitted hourly evaluations only', archetypes=archetypes,
                replay=full)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bars', type=Path, required=True)
    parser.add_argument('--timeframe', choices=['1h', '1min'], required=True)
    parser.add_argument('--instrument', required=True, help='Explicit source label; not independently verified')
    parser.add_argument('--start', required=True, help='Inclusive UTC open time, including prehistory')
    parser.add_argument('--end', required=True, help='Exclusive UTC open time')
    parser.add_argument('--emit-from', help='First emitted candle open; earlier input is replayed warmup')
    parser.add_argument('--volume-column', default='volume')
    parser.add_argument('--observations', type=Path, help='JSONL Observation records with explicit availability')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/champion_paper.json')
    parser.add_argument('--cuts', type=int, nargs='+')
    parser.add_argument('--assume-bar-close-available', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-certification', action='store_true')
    parser.add_argument('--include-signals', action='store_true')
    args = parser.parse_args(argv)
    if not args.assume_bar_close_available:
        parser.error('Historical candles lack receipt provenance; explicitly opt into --assume-bar-close-available for an uncertified experiment')
    inputs = [args.bars, Path(__file__)] + ([args.observations] if args.observations else [])
    input_hashes = file_hashes(inputs)
    frame = pd.read_parquet(args.bars, columns=['open', 'high', 'low', 'close', args.volume_column])
    frame = frame.rename(columns={args.volume_column: 'volume'})
    frame = frame.loc[(frame.index >= utc(args.start)) & (frame.index < utc(args.end))]
    observations = []
    if args.observations:
        observations = [Observation(**json.loads(line)) for line in args.observations.read_text().splitlines() if line.strip()]
    report = exercise(frame, observations, instrument=args.instrument, timeframe=args.timeframe,
                      emit_from=args.emit_from, cuts=args.cuts, config=args.config,
                      include_signals=args.include_signals,
                      progress=lambda message: print(message, flush=True))
    assert_unchanged(input_hashes)
    report['files'] = input_hashes
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(coverage=report['coverage'], checks=report['checks'],
                          certified=False, output=str(args.out)), indent=2))
    return 2 if args.require_certification else 0


if __name__ == '__main__':
    raise SystemExit(main())
