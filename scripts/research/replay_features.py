"""Selected-feature research reference/candidate; never a full-LFC certificate."""
import ast
from copy import deepcopy
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import sys
from typing import Any, Dict

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from engine.archetypes.archetype_instance import ArchetypeConfig, ArchetypeInstance, DERIVED_FEATURES
from scripts.research.gate_observability import gate_observation
from scripts.research.decision_boundary_parity import run_boundary
from scripts.research.replay_clock import digest

SOURCE=ROOT/'bin/live/live_feature_computer.py'
LIQUIDITY_INPUTS=('volume_zscore','atr_percentile','tf1h_fvg_present','oi_change_4h')
FUSION_INPUTS=('rsi_14','adx','tf1h_bos_detected','tf4h_bos_bullish','tf4h_bos_bearish')
FUSION_DESCENDANTS=('fusion_liquidity','tf1h_fusion_score','tf4h_fusion_score',
                    'tf1d_fusion_score','fusion_total')
CONTEXT_DESCENDANTS=('risk_temperature','risk_temp','instability_score','crisis_prob',
                     'regime_label','derivatives_heat')
DEPENDENCIES={
    'funding': ['funding_history','funding_Z'],
    'oi_change_4h': ['liquidity_score',*CONTEXT_DESCENDANTS],
    'liquidity_score': list(FUSION_DESCENDANTS),
}


def finite(value):
    return isinstance(value,(int,float)) and math.isfinite(value)


@lru_cache(maxsize=4)
def helpers(source_text):
    tree=ast.parse(source_text)
    wanted=['_liquidity_score_from','_fusion_scores']
    found=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in wanted]
    if sorted(n.name for n in found)!=sorted(wanted):
        raise ValueError('Ambiguous or missing selected source helpers')
    env={'Dict':Dict,'Any':Any}
    exec(compile(ast.Module(body=found,type_ignores=[]),str(SOURCE),'exec'),env)
    return env


def compare_features(features):
    source=SOURCE.read_text(); funcs=helpers(source)
    ref=deepcopy(dict(features)); candidate=deepcopy(ref)
    # Named candidate only: NaN/invalid presence is not positive FVG evidence.
    for k in ('fvg_present','tf1h_fvg_present','tf4h_fvg_present'):
        if k in candidate and not finite(candidate[k]):
            candidate[k]=False if k=='fvg_present' else 0.
    missing=[k for k in LIQUIDITY_INPUTS if not finite(features.get(k))]
    invalidated=[]; refreshed=[]
    if missing:
        invalidated.extend(['liquidity_score',*FUSION_DESCENDANTS])
    else:
        candidate['liquidity_score']=funcs['_liquidity_score_from'](candidate)
        candidate['fusion_liquidity']=candidate['liquidity_score']
        refreshed.extend(['liquidity_score','fusion_liquidity'])
        missing_fusion=[k for k in FUSION_INPUTS if not finite(candidate.get(k))]
        if missing_fusion:
            missing.extend(missing_fusion)
            invalidated.extend(FUSION_DESCENDANTS[1:])
        else:
            fusion=funcs['_fusion_scores'](None,candidate)
            for k in FUSION_DESCENDANTS:
                candidate[k]=fusion[k]
            refreshed.extend(FUSION_DESCENDANTS[1:])
    # No declaration that unimplemented regime descendants have been repaired.
    invalidated.extend(k for k in CONTEXT_DESCENDANTS if k in candidate)
    for k in invalidated:
        candidate.pop(k,None)
    return dict(reference=dict(features=ref,any_fvg=bool(DERIVED_FEATURES['any_fvg'](ref))),
        candidate=dict(policy='finite-fvg-oi-liquidity-v1',features=candidate,
            any_fvg=bool(DERIVED_FEATURES['any_fvg'](candidate)),refreshed=refreshed,
            invalidated=invalidated,missing_dependencies=missing,
            uncertified=['funding_Z','funding_history','regime_context','asof_provenance']),
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        dependency_inventory=DEPENDENCIES,
        scope='Selected saved-feature reference and correction candidate; no full LFC or strategy parity')


def decision_probe(features,cfg,case,root=ROOT):
    """Supplied score is held fixed pre-gate; not a full archetype fusion replay."""
    comparison=compare_features(features); output={}
    for track in ('reference','candidate'):
        f=comparison[track]['features']
        mode=cfg.get('gate_mode','hard')
        if mode not in ('hard','soft'):
            raise ValueError('Unknown gate mode')
        probe=ArchetypeInstance(ArchetypeConfig(name='contract_probe',direction='long',
                                             hard_gates=cfg.get('hard_gates',[])))
        passed,reason,penalty=probe._evaluate_gates(f)
        pre=float(case['score']); post=pre*penalty if mode=='soft' else pre
        boundary_case=dict(case,score=post,gates_passed=passed)
        live=run_boundary(Path(root)/'bin/live/v11_shadow_runner.py','live',boundary_case)
        backtest=run_boundary(Path(root)/'bin/backtest_v11_standalone.py','backtest',boundary_case)
        blocked=mode=='hard' and not passed
        output[track]=dict(gates=[gate_observation(g,f) for g in cfg.get('hard_gates',[])],
            gate_mode=mode,gates_passed=passed,gate_failure=reason,gate_penalty=penalty,
            pre_gate_score=pre,post_gate_score=post,adjusted_score=post,
            threshold=case['threshold'],bypass={k:case[k] for k in
                ('global_bypass','archetype_bypass','enforce','opt_out')},
            hard_gate_blocked=blocked,live=live,backtest=backtest,
            composed_live_accepted=not blocked and live['accepted'],
            composed_backtest_accepted=not blocked and backtest['accepted'])
    output['scope']='Actual gates + controlled threshold boundary; supplied pre-gate score held fixed, regime adjustment inactive; no structural/cooldown/dedup/book'
    return output


class SelectedFeatureProcessor:
    """Clock adapter for selected snapshot dependencies, not detector state."""
    def __init__(self):
        paths=[SOURCE,ROOT/'engine/archetypes/archetype_instance.py',Path(__file__)]
        self.contract_id=digest({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        self.count=0; self.last=None

    def update(self,candle,observations):
        self.count+=1
        self.last=compare_features(dict(observations,**candle))
        return deepcopy(self.last)

    def snapshot(self):
        return dict(count=self.count,last=deepcopy(self.last),
                    state_scope='Selected feature snapshots only; no funding or detector reconstruction')
