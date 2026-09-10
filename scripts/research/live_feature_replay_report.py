"""Reproducible bounded LFC experiment. Never a profitability certificate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.research.gate_observability import load_configs
from scripts.research.live_feature_replay import run_replay
from scripts.research.replay_clock import Observation, digest, json_safe, utc, validate_bars


def exercise(bars, observations, *, instrument, timeframe, emit_from=None,
             cuts=None, config=ROOT/'configs/champion_paper.json', progress=None):
    step = validate_bars(bars, timeframe)
    observations = list(observations)
    cuts = sorted(set(cuts if cuts is not None else [len(bars)//2, len(bars)-1]))
    cuts = [cut for cut in cuts if 0 < cut < len(bars)]
    configs, paths = load_configs(config)
    notify = progress or (lambda message: None)
    notify(f'Full replay: {len(bars)} {timeframe} bars')
    full = run_replay(bars, observations, instrument=instrument, timeframe=timeframe, emit_from=emit_from)
    checks = []
    for cut in cuts:
        notify(f'Prefix/restart check at bar {cut}/{len(bars)}')
        end = bars.index[cut-1] + step
        available = [record for record in observations if record.visible_at() <= end]
        prefix = run_replay(bars.iloc[:cut], available, instrument=instrument,
                            timeframe=timeframe, emit_from=emit_from)
        resumed = run_replay(bars, observations, instrument=instrument, timeframe=timeframe,
                            emit_from=emit_from, checkpoint=prefix['checkpoint'])
        before = [row for row in full['rows'] if utc(row['decision_time']) <= end]
        after = [row for row in full['rows'] if utc(row['decision_time']) > end]
        checks.append(dict(cut=cut, close_time=str(end), prefix_equal=digest(prefix['rows']) == digest(before),
                           restart_rows_equal=digest(resumed['rows']) == digest(after),
                           restart_state_equal=digest(resumed['state']) == digest(full['state'])))
    blockers = set(full['blockers'])
    blockers.add('candle_availability_assumed_at_close')
    if not checks:
        blockers.add('no_prefix_restart_checks')
    if any(not c[k] for c in checks for k in ('prefix_equal', 'restart_rows_equal', 'restart_state_equal')):
        blockers.add('prefix_or_restart_mismatch')
    if not full['rows']:
        blockers.add('no_emitted_evidence')
    return dict(certified=False, scope=full['scope'], blockers=sorted(blockers),
                coverage=dict(input_bars=len(bars), emitted_rows=len(full['rows']),
                              hourly_updates=full['state']['hourly_updates'],
                              first_open=str(bars.index[0]), last_close=str(bars.index[-1]+step),
                              final_feature_count=len(full['state']['features'] or {})),
                assumptions=dict(startup='cold replay of supplied original prehistory',
                                 candle_availability='assumed exact bar close; receipt times unknown',
                                 deep_daily=False, external_observation_count=len(observations)),
                checks=checks, state_hash=digest(full['state']), rows_hash=digest(full['rows']),
                contract_id=full['contract_id'], source_manifest=full['source_manifest'],
                input_rows_hash=digest(dict(index=list(bars.index), rows=bars.to_dict('records'))),
                config_hashes={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
                               hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                archetypes={name: 'signal_layer_not_replayed' for name in configs},
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
    args = parser.parse_args(argv)
    if not args.assume_bar_close_available:
        parser.error('Historical candles lack receipt provenance; explicitly opt into --assume-bar-close-available for an uncertified experiment')
    frame = pd.read_parquet(args.bars, columns=['open', 'high', 'low', 'close', args.volume_column])
    frame = frame.rename(columns={args.volume_column: 'volume'})
    frame = frame.loc[(frame.index >= utc(args.start)) & (frame.index < utc(args.end))]
    observations = []
    if args.observations:
        observations = [Observation(**json.loads(line)) for line in args.observations.read_text().splitlines() if line.strip()]
    report = exercise(frame, observations, instrument=args.instrument, timeframe=args.timeframe,
                      emit_from=args.emit_from, cuts=args.cuts, config=args.config,
                      progress=lambda message: print(message, flush=True))
    inputs = [args.bars, Path(__file__)] + ([args.observations] if args.observations else [])
    report['files'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(coverage=report['coverage'], checks=report['checks'],
                          certified=False, output=str(args.out)), indent=2))
    return 2 if args.require_certification else 0


if __name__ == '__main__':
    raise SystemExit(main())
