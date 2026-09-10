#!/usr/bin/env python3
"""Read-only gate-input audit using the actual ArchetypeInstance evaluator.

All input bars are profiled, not just trade entries. Missing derived inputs can
be defaulted by production and pass: that is distinguished from an actual skip.
"""
import argparse
import ast
from collections import Counter
from functools import lru_cache
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from engine.archetypes import archetype_instance as production


@lru_cache(None)
def derived_dependencies():
    tree = ast.parse(inspect.getsource(production))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DERIVED_FEATURES' for t in node.targets):
            return {key.value: sorted({n.args[0].value for n in ast.walk(value)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'get' and n.args and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)})
                for key, value in zip(node.value.keys, node.value.values)}
    raise ValueError('Cannot find production derived-feature registry')


def dependencies(gate):
    key = gate.get('feature', '')
    return derived_dependencies().get(key.split(':', 1)[1], []) if key.startswith('derived:') else [key]


def missing(value):
    return value is None or bool(pd.isna(value))


@lru_cache(None)
def evaluator(serialized_gate):
    gate = json.loads(serialized_gate)
    return production.ArchetypeInstance(production.ArchetypeConfig(
        name='research_gate_probe', direction='long', hard_gates=[gate]))


def gate_observation(gate, features):
    key = gate.get('feature', '')
    deps = dependencies(gate)
    absent = [k for k in deps if missing(features.get(k))]
    invalid = []
    for dep in deps:
        val = features.get(dep)
        if dep in absent:
            continue
        try:
            if not math.isfinite(float(val)):
                invalid.append(dep)
        except (TypeError, ValueError):
            # Text-valued context is a valid input to context-derived gates.
            if dep not in ('wyckoff_context', 'wyckoff_phase_abc'):
                invalid.append(dep)
    derived = key.startswith('derived:')
    resolution = 'value'
    if derived:
        fn = production.DERIVED_FEATURES.get(key.split(':', 1)[1])
        if fn is None:
            value, resolution = None, 'unknown_derived'
        else:
            try:
                value = fn(features)
            except Exception:
                value, resolution = None, 'compute_error'
    else:
        value = features.get(key)
    probe = evaluator(json.dumps(gate, sort_keys=True))
    try:
        passed, reason, penalty = probe._evaluate_gates(features)
        unavailable = missing(value) or resolution != 'value'
        status = ('skip' if passed else 'fail') if unavailable else 'pass' if passed else 'fail'
        if gate.get('op', 'bool_true') not in ('min', 'max', 'bool_true', 'bool_false', 'in_range', 'eq'):
            status = 'unsupported_operator'
    except Exception as exc:
        passed, reason, penalty, status = None, type(exc).__name__, None, 'error'
    return dict(status=status, runtime_passed=passed, runtime_reason=reason,
                runtime_penalty=penalty, missing_inputs=absent, invalid_inputs=invalid,
                derived_defaulted=derived and bool(absent), resolution=resolution,
                value=None if missing(value) else value)


def audit_frame(frame, configs, source):
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError('Timezone-aware input index required')
    result = []
    for name, cfg in configs.items():
        gates = cfg.get('hard_gates', [])
        if frame.empty or not gates:
            result.append(dict(source=source, archetype=name, year=None,
                               rows=len(frame), gate_index=None, status='no_rows' if frame.empty else 'no_gates'))
            continue
        for gi, gate in enumerate(gates):
            deps = dependencies(gate)
            columns = [k for k in deps if k in frame]
            for year, part in frame.groupby(frame.index.year):
                counts, resolutions, values = Counter(), Counter(), set()
                missing_rows = defaulted_rows = invalid_rows = 0
                for features in part[columns].to_dict('records'):
                    obs = gate_observation(gate, features)
                    counts[obs['status']] += 1; resolutions[obs['resolution']] += 1
                    missing_rows += bool(obs['missing_inputs'])
                    defaulted_rows += obs['derived_defaulted']
                    invalid_rows += bool(obs['invalid_inputs'])
                    if obs['value'] is not None:
                        values.add(str(obs['value']))
                result.append(dict(source=source, archetype=name, year=int(year), rows=len(part),
                    gate_index=gi, gate=gate, direction=cfg.get('direction'),
                    gate_mode=cfg.get('gate_mode', 'hard'), enabled=cfg.get('enabled', True),
                    bypass_fusion_threshold=cfg.get('bypass_fusion_threshold', False),
                    enforce_gates_under_bypass=cfg.get('enforce_gates_under_bypass', 'default'),
                    dependencies=deps, missing_columns=[k for k in deps if k not in frame],
                    counts=dict(counts), resolutions=dict(resolutions),
                    missing_input_rows=missing_rows, derived_defaulted_rows=defaulted_rows,
                    invalid_input_rows=invalid_rows, unique_values=len(values),
                    constant_nonmissing=len(values) == 1,
                    constant_value=next(iter(values)) if len(values) == 1 else None))
    return result


def load_configs(config_path):
    config_path = Path(config_path).resolve()
    root_config = json.loads(config_path.read_text())
    directory = Path(root_config['archetype_config_dir'])
    if not directory.is_absolute():
        directory = config_path.parent.parent/directory
    configs, paths = {}, [config_path]
    for p in sorted(directory.glob('*.yaml')):
        if 'example' in p.stem:
            continue
        cfg = yaml.safe_load(p.read_text())
        if not isinstance(cfg, dict) or not cfg.get('name'):
            continue
        name = cfg['name']
        if name in configs:
            raise ValueError('Duplicate archetype name')
        if root_config.get('gate_mode') is not None:
            cfg['gate_mode'] = root_config['gate_mode']
        overrides = root_config.get('gate_overrides', {}).get(name, {})
        for gate in cfg.get('hard_gates', []):
            if gate['feature'] in overrides:
                gate['value'] = overrides[gate['feature']]
        cfg['enabled'] = cfg.get('enabled', True) and name not in root_config.get('disabled_archetypes', [])
        configs[name] = cfg; paths.append(p)
    if not configs:
        raise ValueError('No archetype configs resolved')
    return configs, paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--store', type=Path, required=True)
    p.add_argument('--live-jsonl', type=Path, nargs='*', default=[])
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    configs, cfg_paths = load_configs(args.config)
    import pyarrow.parquet as pq
    wanted = {k for cfg in configs.values() for g in cfg.get('hard_gates', []) for k in dependencies(g)}
    schema = pq.ParquetFile(args.store).schema.names
    frame = pd.read_parquet(args.store, columns=sorted(wanted.intersection(schema)))
    frames = [('store', frame)]
    if args.live_jsonl:
        records = []
        for path in args.live_jsonl:
            with path.open() as f:
                records.extend(json.loads(line) for line in f if line.strip())
        live = pd.DataFrame(records)
        if live.empty:
            live.index = pd.DatetimeIndex([], tz='UTC')
        else:
            live.index = pd.to_datetime(live.pop('timestamp'), utc=True)
        frames.append(('archived_live', live))
    result = dict(roster=list(configs), profiles=[], sources={},
        limitations=['All input bars, not conditioned on structural eligibility or actual entries.',
            'A derived gate may default missing dependencies; absence of any referenced alternative is conservatively flagged.',
            'Coverage does not establish as-of availability, feed correctness or strategy value.',
            'Historical snapshots may contain repeated timestamps; counts preserve rows, not independent trials.',
            'Gate mode is local policy; downstream threshold/dedup behavior is separate.'])
    for label, df in frames:
        result['sources'][label] = dict(rows=len(df), duplicate_timestamps=int(df.index.duplicated().sum()),
            start=str(df.index.min()) if len(df) else None, end=str(df.index.max()) if len(df) else None)
        result['profiles'].extend(audit_frame(df, configs, label))
    result['hashes'] = {str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in cfg_paths+[args.store]+args.live_jsonl+[Path(__file__), Path(production.__file__)]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'gate_observability.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(archetypes=len(configs), profiles=len(result['profiles']), sources=result['sources']), indent=2))


if __name__ == '__main__':
    main()
