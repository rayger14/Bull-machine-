#!/usr/bin/env python3
"""Execute unmodified live/backtest adaptive-threshold loops on controlled inputs.

This is a boundary test, NOT full replay parity: feature computation, structural
checks, upstream gates/cooldown, dedup, sizing and exits are outside this probe.
No runner is constructed; no network or execution adapter is invoked.
"""
import argparse
import ast
from functools import lru_cache
import hashlib
import itertools
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=4)
def extract_loop(text, filename):
    tree = ast.parse(text, filename=filename)
    matches = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and node.target.id == 's' and isinstance(node.iter, ast.Name)
                and node.iter.id == 'signals'):
            continue
        comparisons = [n for n in ast.walk(node) if isinstance(n, ast.Compare)
            and isinstance(n.left, ast.Name) and n.left.id == 'adjusted_fusion'
            and len(n.ops) == 1 and isinstance(n.ops[0], ast.GtE)
            and isinstance(n.comparators[0], ast.Name) and n.comparators[0].id == 'arch_threshold']
        if comparisons:
            matches.append(node)
    if len(matches) != 1:
        raise ValueError(f'Expected one adaptive signal loop, found {len(matches)} in {filename}')
    loop = matches[0]
    module = ast.Module(body=[loop], type_ignores=[])
    return compile(module, filename, 'exec'), loop.lineno, loop.end_lineno


def run_boundary(source_path, kind, case):
    if kind not in ('live', 'backtest'):
        raise ValueError('Unknown boundary kind')
    source_path = Path(source_path)
    text = source_path.read_text()
    code, start, end = extract_loop(text, str(source_path))
    score, threshold = float(case['score']), float(case['threshold'])
    if not math.isfinite(score) or not math.isfinite(threshold):
        raise ValueError('Finite boundary inputs required')
    aid = 'research_probe'
    metadata = dict(hard_gates_passed=case['gates_passed'], hard_gates_failed_reason='fixture gate')
    signal = SimpleNamespace(archetype_id=aid, fusion_score=score, metadata=metadata)
    adaptive = dict(enforce_gates_under_bypass=case['enforce'])
    arch = dict(bypass_fusion_threshold=case['archetype_bypass'])
    if case['opt_out']:
        arch['enforce_gates_under_bypass'] = False
    # Only dependencies of the selected boundary are provided. The phantom
    # recorder is intentionally inert; it is logging, not part of acceptance.
    context = SimpleNamespace(adaptive_fusion=adaptive,
        bypass_threshold=case['global_bypass'],
        engine=SimpleNamespace(archetype_configs={aid:arch}, archetypes={}),
        last_bar_signals=[{}], signals_rejected=0,
        _open_phantom=lambda *args: None)
    accepted = []
    env = dict(self=context, signals=[signal], adjusted_signals=accepted,
        logger=logging.getLogger('research_boundary'),
        crisis_penalty=1., per_arch_thresholds={}, base_threshold=threshold,
        risk_temp=1., temp_range=0., instability=0., instab_range=0.,
        crisis_prob=0., current_regime='neutral', features={}, sig_index={id(signal):0},
        _regime_weight_mode='fusion_multiplier', _regime_weight_hard_block_floor=.2,
        _regime_weight_blocks=0,
        _bypass_archetypes={aid} if case['archetype_bypass'] else set(),
        _gate_exempt_archetypes={aid} if case['opt_out'] else set(),
        _enforce_gates_under_bypass_default=case['enforce'])
    exec(code, env)
    return dict(accepted=bool(accepted), source=str(source_path),
        source_lines=[start,end], sha256=hashlib.sha256(text.encode()).hexdigest(),
        rejection_stage=context.last_bar_signals[0].get('rejection_stage'),
        bypass_gate_blocks=getattr(context,'bypass_gate_blocks',0))


def compare_boundaries(root):
    root=Path(root)
    paths={'live':root/'bin/live/v11_shadow_runner.py',
           'backtest':root/'bin/backtest_v11_standalone.py'}
    results=[]
    names=['gates_passed','global_bypass','archetype_bypass','enforce','opt_out']
    for score in [.1,.3,.5]:
        for flags in itertools.product([False,True],repeat=5):
            # Include two threshold levels so above/equal/below paths are
            # independently exercised at more than one numeric boundary.
            for threshold in [.3,.4]:
                case=dict(zip(names,flags)); case.update(score=score,threshold=threshold)
                outputs={kind:run_boundary(path,kind,case) for kind,path in paths.items()}
                results.append(dict(inputs=case, **outputs,
                    matches=outputs['live']['accepted']==outputs['backtest']['accepted']))
    mismatches=[r for r in results if not r['matches']]
    return dict(cases=results,mismatches=mismatches,parity_passed=not mismatches,
        limitations=['Controlled hypothetical signals at the adaptive-threshold boundary only.',
            'No proof that every fixture is reachable from actual upstream gates.',
            'No feature/structural/cooldown/dedup/allocation/exit or current-server parity claim.',
            'Regime-weight adjustment is held in its default inactive mode.',
            'Source loops are executed unchanged; runner constructors and external adapters are not used.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--require-parity',action='store_true')
    args=p.parse_args()
    result=compare_boundaries(args.root)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'decision_boundary_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cases=len(result['cases']),mismatches=len(result['mismatches']),
                          parity_passed=result['parity_passed']),indent=2))
    if args.require_parity and not result['parity_passed']:
        raise SystemExit(2)


if __name__=='__main__':
    main()
