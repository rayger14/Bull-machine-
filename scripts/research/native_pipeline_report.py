"""Native-source pipeline diagnostics, explicitly NOT an executable backtest."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.research.engine_signal_replay import EXPECTED_ARCHETYPES
from scripts.research.gate_observability import load_configs
from scripts.research.live_feature_replay_report import assert_unchanged, file_hashes, manifest_files
from scripts.research.replay_clock import Observation, digest, json_safe, utc, validate_bars


def validate_output(path, protected):
    target = Path(path).resolve()
    if target.suffix != '.json' or target in {Path(p).resolve() for p in protected}:
        raise ValueError('Report output is protected or is not a separate .json file')
    try:
        relative = target.relative_to(ROOT/'results')
    except ValueError as exc:
        raise ValueError('Report output is protected: use a repository research results directory') from exc
    if len(relative.parts) < 2 or not re.fullmatch(
            r'research(?:_validation_[0-9]{4}_[0-9]{2}_[0-9]{2})?', relative.parts[0]):
        raise ValueError('Report output is protected: only results/research or results/research_validation_YYYY_MM_DD')


def summarize(result):
    archetypes = {name: dict(evaluations=0, native_candidates=0, selected=0,
        entries=0, rejected=0, exit_events=0) for name in sorted(EXPECTED_ARCHETYPES)}
    rejections = Counter()
    hours = 0
    for row in result['rows']:
        output = row['output']
        signal, book = output['engine_signal'], output['native_book']
        if not output['hourly_updated']:
            if signal is not None or book is not None:
                raise ValueError('Non-hourly carry row contains signal/book activity')
            continue
        if signal is None or book is None:
            raise ValueError('Completed hourly update lacks signal/book evidence')
        if set(signal['archetypes']) != EXPECTED_ARCHETYPES:
            raise ValueError('All 17 archetype evaluations required')
        hours += 1
        for name, outcome in signal['archetypes'].items():
            archetypes[name]['evaluations'] += 1
            archetypes[name]['native_candidates'] += int(outcome['native_signal'] is not None)
            archetypes[name]['selected'] += int(outcome['selected'])
        for event in book['acted_signals']:
            if event['action'] == 'ENTRY':
                archetypes[event['archetype']]['entries'] += 1
        for event in book['last_bar_signals']:
            if event['status'] == 'rejected':
                archetypes[event['archetype']]['rejected'] += 1
                rejections[event['rejection_stage']] += 1
        for event in book['records']:
            if event['kind'] == 'outcome_log':
                archetypes[event['payload']['position']['archetype']]['exit_events'] += 1
    state = result['state']['native_book']['state']
    blockers = set(result['blockers']) | {'native_accounting_not_executable_returns'}
    if not hours:
        blockers.add('no_emitted_book_evidence')
    if not any(value['entries'] for value in archetypes.values()):
        blockers.add('no_emitted_entry_evidence')
    return dict(certified=False, execution_causality_certified=False, scope=result['scope'],
        blockers=sorted(blockers), archetypes=archetypes, rejection_stages=dict(rejections),
        coverage=dict(emitted_rows=len(result['rows']), emitted_book_hours=hours,
                      book_hours_including_warmup=result['state']['hourly_updates']),
        native_accounting=dict(cash=state['cash'],
            equity=state['equity_curve'][-1] if state['equity_curve'] else state['cash'],
            open_positions=len(state['positions']), exit_events=len(state['trades']),
            phantom_open_positions=len(state['phantom_positions']),
            phantom_exit_events=len(state['phantom_trades']),
            scope='Final native book including full prehistory; exit events are not independent positions'),
        assumptions=dict(count_scope='Emitted hourly events only; final book includes prehistory positions',
            warmup_policy='Trade through all supplied prehistory; suppress output only',
            execution='Native close-price fills, backdated source labels; not measured executable prices',
            average_initial_stop_risk='Not computed by this diagnostic; no risk-adjusted performance claim'))


def exercise(bars, observations, *, instrument, timeframe, initial_cash, commission_rate,
             slippage_bps, emit_from=None, cuts=None, progress=None):
    from scripts.research.native_pipeline_replay import run_pipeline_replay
    step = validate_bars(bars, timeframe)
    observations = list(observations)
    _, config_paths = load_configs(ROOT/'configs/champion_paper.json')
    config_hashes = file_hashes(config_paths)
    costs = dict(initial_cash=initial_cash, commission_rate=commission_rate, slippage_bps=slippage_bps)
    kwargs = dict(instrument=instrument, timeframe=timeframe, emit_from=emit_from, **costs)
    notify = progress or (lambda message: None)
    notify(f'Native pipeline: {len(bars)} {timeframe} bars')
    assert_unchanged(config_hashes)
    full = run_pipeline_replay(bars, observations, **kwargs)
    checks = []
    for cut in sorted(set(cuts if cuts is not None else [len(bars)//2, len(bars)-1])):
        if not 0 < cut < len(bars):
            continue
        notify(f'Native pipeline prefix/restart at {cut}/{len(bars)}')
        end = bars.index[cut-1] + step
        prefix = run_pipeline_replay(bars.iloc[:cut],
            [o for o in observations if o.visible_at() <= end], **kwargs)
        resumed = run_pipeline_replay(bars, observations, checkpoint=prefix['checkpoint'], **kwargs)
        before = [r for r in full['rows'] if utc(r['decision_time']) <= end]
        after = [r for r in full['rows'] if utc(r['decision_time']) > end]
        checks.append(dict(cut=cut, close_time=str(end), prefix_rows_compared=len(before),
            restart_rows_compared=len(after), prefix_equal=digest(prefix['rows']) == digest(before),
            restart_rows_equal=digest(resumed['rows']) == digest(after),
            restart_state_equal=digest(resumed['state']) == digest(full['state']),
            contract_equal=full['contract_id'] == prefix['contract_id'] == resumed['contract_id']))
    assert_unchanged(manifest_files(full['source_manifest']))
    assert_unchanged(config_hashes)
    report = summarize(full)
    blockers = set(report['blockers']) | {'candle_availability_assumed_at_close'}
    if not checks:
        blockers.add('no_prefix_restart_checks')
    if any(not c[key] for c in checks for key in
           ('prefix_equal', 'restart_rows_equal', 'restart_state_equal', 'contract_equal')):
        blockers.add('prefix_or_restart_mismatch')
    report['blockers'] = sorted(blockers)
    report['coverage'].update(input_bars=len(bars), first_open=str(bars.index[0]),
                              last_close=str(bars.index[-1]+step))
    report.update(checks=checks, economic_parameters=costs, config_hashes=config_hashes,
        contract_id=full['contract_id'],
        source_manifest=full['source_manifest'], state_hash=digest(full['state']),
        rows_hash=digest(full['rows']),
        input_rows_hash=digest(dict(index=list(bars.index), rows=bars.to_dict('records'))), replay=full)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bars', type=Path, required=True)
    parser.add_argument('--timeframe', choices=['1h', '1min'], required=True)
    parser.add_argument('--instrument', required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True, help='Exclusive candle-open bound')
    parser.add_argument('--emit-from')
    parser.add_argument('--volume-column', default='volume')
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--initial-cash', type=float, required=True)
    parser.add_argument('--commission-rate', type=float, required=True)
    parser.add_argument('--slippage-bps', type=float, required=True)
    parser.add_argument('--cuts', type=int, nargs='+')
    parser.add_argument('--assume-bar-close-available', action='store_true')
    parser.add_argument('--require-certification', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.assume_bar_close_available:
        parser.error('Explicit --assume-bar-close-available required; native fills stay uncertified')
    inputs = [args.bars, Path(__file__)] + ([args.observations] if args.observations else [])
    try:
        validate_output(args.out, inputs)
    except ValueError as exc:
        parser.error(str(exc))
    hashes = file_hashes(inputs)
    bars = pd.read_parquet(args.bars, columns=['open', 'high', 'low', 'close', args.volume_column])
    bars = bars.rename(columns={args.volume_column: 'volume'})
    bars = bars.loc[(bars.index >= utc(args.start)) & (bars.index < utc(args.end))]
    observations = [Observation(**json.loads(line)) for line in args.observations.read_text().splitlines()
                    if line.strip()] if args.observations else []
    report = exercise(bars, observations, instrument=args.instrument, timeframe=args.timeframe,
        initial_cash=args.initial_cash, commission_rate=args.commission_rate,
        slippage_bps=args.slippage_bps, emit_from=args.emit_from, cuts=args.cuts,
        progress=lambda message: print(message, flush=True))
    assert_unchanged(hashes)
    validate_output(args.out, set(hashes) | set(report['config_hashes']) |
                    set(manifest_files(report['source_manifest'])))
    report['files'] = hashes
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in ('coverage', 'checks', 'certified')}, indent=2))
    return 2 if args.require_certification else 0


if __name__ == '__main__':
    raise SystemExit(main())
