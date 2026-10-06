"""Offline LC practice commands. `owner` is a persistent JSON-lines host bridge.

No command launches models. The host reserves, delivers one complete request to
one fresh role, records its identity, and returns exact bytes to the same owner.
"""
import argparse
import json
from pathlib import Path
import sys

from scripts.research.lc_judgment_runner import _load, _sha
from scripts.research.lc_practice_runtime import PracticeRun, prepare_run


EVIDENCE=Path('results/lc_consolidated_2026_09_15/judgment_v1/evidence')
ARCHIVE=Path('data/recovered_binance_minute_2026_09_14/btc_1m_2021_2026.parquet')
ARCHIVE_SHA='5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035'
ROSTER=['2026-01-20T06:00:00+00:00','2026-01-25T09:00:00+00:00','2026-01-29T16:00:00+00:00',
    '2026-01-31T15:00:00+00:00','2026-02-23T02:00:00+00:00','2026-02-25T02:00:00+00:00',
    '2026-02-28T07:00:00+00:00','2026-03-07T20:00:00+00:00','2026-03-08T23:00:00+00:00',
    '2026-03-22T22:00:00+00:00','2026-05-02T22:00:00+00:00','2026-05-03T23:00:00+00:00']


def _emit(value):
    print(json.dumps(value,sort_keys=True,allow_nan=False,default=str),flush=True)


def owner_bridge(run):
    _emit(dict(owner_ready=True,status=run.status()))
    for line in sys.stdin:
        try:
            command=json.loads(line); op=command['op']
            if op=='status': result=run.status()
            elif op=='authorize': result=run.authorize(command['user_instruction'],command['manifest_sha256'])
            elif op=='reserve':
                result=run.reserve(command['case_id'])
                # Large payload already exists as a sealed, read-only request file.
                # Host can read it once or request its exact hex explicitly.
                if not command.get('include_request_hex',False): result.pop('request_hex')
            elif op=='attach': result=run.attach(command['case_id'],command['agent_id'])
            elif op=='capture':
                raw=Path(command['response_path']).read_bytes()
                delivered=Path(command['delivered_path']).read_bytes()
                result=run.capture(command['case_id'],raw,command['metadata'],delivered)
            elif op=='fail': result=run.fail(command['case_id'],command['reason'])
            elif op=='not_run': result=run.mark_not_run(command['case_id'],command['reason'])
            elif op=='lock': result=run.lock_terminals()
            else: raise ValueError('unknown owner command')
            _emit(dict(op=op,result=result))
        except (ValueError,KeyError,TypeError,OSError) as exc:
            _emit(dict(error=type(exc).__name__+': '+str(exc)))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','status','owner','lock','score','report'])
    parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args(argv)
    if args.command=='prepare':
        roster=_load(EVIDENCE/'roster.json')
        if len(roster)!=20 or sorted(roster)[:12]!=['hourly-lc:'+d for d in ROSTER]:
            raise ValueError('original twenty / selected twelve roster changed')
        if _sha(ARCHIVE)!=ARCHIVE_SHA: raise ValueError('original minute archive changed')
        m=prepare_run(args.run,[EVIDENCE/(cid+'_source_request.json') for cid in roster],
                      input_lock=EVIDENCE/'evidence_lock.json',archive=ARCHIVE)
        _emit(dict(manifest_sha256=m['sha256'],cases=m['cases'],role_policy=m['role_policy']))
        return
    with PracticeRun(args.run) as run:
        if args.command=='owner': owner_bridge(run)
        elif args.command=='status': _emit(run.status())
        elif args.command=='lock': _emit(run.lock_terminals())
        elif args.command=='score':
            from scripts.research.lc_practice_replay import score_run
            result=score_run(run);_emit(dict(result_sha256=result['sha256'],summary=result['summary']))
        elif args.command=='report':
            from scripts.research.lc_practice_report import render_report
            _emit(render_report(run))


if __name__=='__main__': main()
