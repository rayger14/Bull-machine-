#!/usr/bin/env python3
"""Version-aware paired-input diagnostics. Never certifies full live parity.

No network or live runner construction. Missing historical provenance remains
a blocker, not an inferred timestamp/source or an excuse to rewrite raw data.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.research.gate_observability import gate_observation,load_configs
from scripts.research.replay_features import compare_features,finite,SOURCE,DEPENDENCIES
from scripts.research.replay_clock import json_safe,utc
from scripts.research.decision_boundary_parity import compare_boundaries


def build_report(store,live_records,configs,hashes):
    if not isinstance(store.index,pd.DatetimeIndex) or store.index.tz is None:
        raise ValueError('Store requires timezone-aware index')
    if not store.index.is_unique or not store.index.is_monotonic_increasing or store.index.hasnans:
        raise ValueError('Store timestamps must be unique, ordered and finite')
    records=list(live_records)
    live=pd.DataFrame(records)
    if not records:
        live.index=pd.DatetimeIndex([],tz='UTC')
    else:
        if 'timestamp' not in live:
            raise ValueError('Live timestamp missing')
        live.index=pd.DatetimeIndex([utc(t,'live timestamp') for t in live.pop('timestamp')])
        if not live.index.is_monotonic_increasing:
            # The archive is an input inventory, not a clock replay. Do not
            # silently reorder it and imply the original sequence was valid.
            order_issue=True
        else:
            order_issue=False
    duplicates=live.index.duplicated(keep=False)
    # Preserve sparse JSON key absence. A rectangular DataFrame inserts NaN
    # and can change bool(NaN)-sensitive production behavior.
    identity_rows=[any(store.attrs.get(k) and r.get(k) and store.attrs[k]!=r[k]
                       for k in ('instrument','venue')) for r in records]
    raw_by_time={t:r for t,r,dup,bad_id in zip(live.index,records,duplicates,identity_rows)
                 if not dup and not bad_id}
    common=store.index.intersection(pd.DatetimeIndex(list(raw_by_time),tz='UTC')).sort_values()
    srows=store.loc[common].to_dict('records')
    lrows=[dict(raw_by_time[t]) for t in common]
    pairs=[(compare_features(s),compare_features(l)) for s,l in zip(srows,lrows)]
    blockers=['full_pipeline_not_replayed','historical_state_not_reconstructed']
    if not len(common): blockers.append('no_paired_evidence')
    if len(records) and order_issue: blockers.append('archive_out_of_order')
    if duplicates.any(): blockers.append('ambiguous_duplicate_timestamps')
    if any(identity_rows): blockers.append('instrument_or_venue_mismatch')
    if not records or any(not r.get('available_at') for r in records):
        blockers.append('missing_available_at')
    else:
        try:
            for r in records:
                if utc(r['available_at']) < utc(r['timestamp']):
                    raise ValueError('Availability precedes event label')
        except (ValueError,TypeError): blockers.append('invalid_available_at')
    store_version=store.attrs.get('source_version')
    versions={r.get('source_version') for r in records if r.get('source_version')}
    if not store_version or len(versions)!=1 or any(not r.get('source_version') for r in records):
        blockers.append('missing_source_version')
    if store_version and versions and versions!={store_version}:
        blockers.append('source_version_mismatch')
    if not store.attrs.get('instrument') or not store.attrs.get('venue') or any(
            not r.get('instrument') or not r.get('venue') for r in records):
        blockers.append('missing_instrument_or_venue_provenance')
    archetypes=[]
    for name,cfg in configs.items():
        gates=[]
        for i,gate in enumerate(cfg.get('hard_gates',[])):
            counts=Counter();statuses={k:Counter() for k in ('store_reference','store_candidate','live_reference','live_candidate')}
            for sr,lr in pairs:
                observations={f'{source}_{track}':gate_observation(gate,pair[track]['features'])
                    for source,pair in [('store',sr),('live',lr)] for track in ('reference','candidate')}
                for key,o in observations.items(): statuses[key][o['status']]+=1
                for source in ('store','live'):
                    counts[f'{source}_candidate_changes']+=observations[f'{source}_reference']['runtime_passed']!=observations[f'{source}_candidate']['runtime_passed']
                counts['paired_reference_differences']+=observations['store_reference']['runtime_passed']!=observations['live_reference']['runtime_passed']
            gates.append(dict(index=i,feature=gate['feature'],gate=gate,rows=len(common),
                **{k:int(counts[k]) for k in ('store_candidate_changes','live_candidate_changes','paired_reference_differences')},
                statuses={k:dict(v) for k,v in statuses.items()}))
        archetypes.append(dict(name=name,enabled=cfg.get('enabled',True),direction=cfg.get('direction'),
            gate_mode=cfg.get('gate_mode','hard'),paired_rows=len(common),gates=gates,
            status='no_data' if not len(common) else 'no_gates' if not gates else 'profiled'))
    feature_summary={}
    for label,pair_index in [('store',0),('archived_live',1)]:
        fvg_changes=0;liq_changes=0;liq_n=0;invalid=Counter()
        for pair in pairs:
            r=pair[pair_index];ref=r['reference'];cand=r['candidate']
            fvg_changes+=ref['any_fvg']!=cand['any_fvg']
            a=ref['features'].get('liquidity_score');b=cand['features'].get('liquidity_score')
            if finite(a) and finite(b):
                liq_n+=1;liq_changes+=abs(a-b)>1e-10+1e-8*abs(b)
            invalid.update(cand['invalidated'])
        feature_summary[label]=dict(any_fvg_changes=fvg_changes,liquidity_comparable=liq_n,
            liquidity_changes=liq_changes,invalidated=dict(invalid))
    boundary=compare_boundaries(ROOT)
    if not boundary['parity_passed']: blockers.append('threshold_boundary_mismatch')
    return json_safe(dict(certified=False,blockers=sorted(set(blockers)),
        coverage=dict(store_rows=len(store),live_rows=len(records),paired_rows=len(common),
            excluded_duplicate_rows=int(duplicates.sum()),
            excluded_identity_rows=sum(identity_rows),
            duplicate_timestamps=sorted(set(map(str,live.index[duplicates]))),
            start=str(common.min()) if len(common) else None,end=str(common.max()) if len(common) else None),
        source_versions=dict(store=store_version,live=sorted(versions)),
        archetypes=archetypes,feature_corrections=feature_summary,
        dependency_inventory=DEPENDENCIES,
        boundary=dict(cases=len(boundary['cases']),mismatches=len(boundary['mismatches']),
                      parity_passed=boundary['parity_passed'],details=boundary),
        hashes=hashes,limitations=['not_trade_counts','Current gates on historical snapshots, not historical decisions.',
            'Reference preserves raw features; candidate is finite-FVG/OI-liquidity only.',
            'Funding, regime context, full detector state and execution remain uncertified.',
            'Archived input pairing is not a current-server golden master.']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store',type=Path,required=True)
    p.add_argument('--live-jsonl',type=Path,nargs='+',required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--require-certification',action='store_true')
    args=p.parse_args()
    configs,paths=load_configs(args.config)
    records=[json.loads(line) for path in args.live_jsonl for line in path.read_text().splitlines() if line.strip()]
    code=[Path(__file__),SOURCE,ROOT/'engine/archetypes/archetype_instance.py',
          ROOT/'scripts/research/gate_observability.py',ROOT/'scripts/research/replay_clock.py',
          ROOT/'scripts/research/replay_features.py',ROOT/'scripts/research/decision_boundary_parity.py',
          ROOT/'bin/live/v11_shadow_runner.py',ROOT/'bin/backtest_v11_standalone.py']
    hashes={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*paths,*args.live_jsonl,args.store,*code]}
    result=build_report(pd.read_parquet(args.store),records,configs,hashes)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'replay_contract_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('certified','blockers','coverage','feature_corrections')},indent=2))
    if args.require_certification and not result['certified']:
        raise SystemExit(2)


if __name__=='__main__': main()
