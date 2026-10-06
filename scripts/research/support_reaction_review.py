"""Lossless, source-only presentation and explicit human/controller adjudication.

No annotation schema check is represented as independent semantic agreement.
"""
from collections import defaultdict
from copy import deepcopy
import argparse
import json
from pathlib import Path

from scripts.research.support_reaction_study import CLAIMS, validate_annotation
from scripts.research.thesis_contract import clock, seal, signed, verify
from scripts.research.thesis_source import ROOT, sha, verify_files

SOURCE_DIR = ROOT/'results/support_reaction_2026_10_03/source_v1'
SOURCE_RECEIPT_SHA = '5c261f850fc125d7523a92c9e319594ef259d4cd3d4cc607236cb10fac28d71b'


def _replace(value, aliases):
    if isinstance(value, str): return aliases.get(value, value)
    if isinstance(value, list): return [_replace(v, aliases) for v in value]
    if isinstance(value, dict): return {k: _replace(v, aliases) for k, v in value.items()}
    return value


def compact(packet):
    verify(packet)
    observations = packet['observations']
    if any(clock(o['available_at']) > clock(packet['cutoff']) for o in observations.values()):
        raise ValueError('future observation')
    ordered = sorted(observations, key=lambda k: (clock(observations[k]['available_at']), k))
    aliases = {f'O{i:03d}': k for i, k in enumerate(ordered, 1)}
    aliases['STREAM'] = packet['stream_id']
    reverse = {v: k for k, v in aliases.items()}
    if len(reverse) != len(aliases): raise ValueError('duplicate source identity')
    metadata = _replace({k: v for k, v in packet.items() if k not in ('observations', 'seal')}, reverse)
    groups = defaultdict(list)
    for cid in ordered:
        o = _replace(deepcopy(observations[cid]), reverse)
        if o['id'] != reverse[cid]: raise ValueError('observation key/id mismatch')
        payload = o.pop('payload', None)
        # Keep explicit unknown/empty payloads distinct from absent payload keys.
        if 'payload' in observations[cid]:
            if isinstance(payload, dict) and payload:
                o.update({'payload.'+k: v for k, v in payload.items()})
            elif payload is None or payload == {}: o['payload'] = payload
            else: raise ValueError('unsupported payload')
        columns = tuple(sorted(o))
        groups[columns].append([o[k] for k in columns])
    mapping = signed(dict(packet_seal=packet['seal'], aliases=aliases))
    view = signed(dict(schema='support-compact-source-v1', packet_seal=packet['seal'],
                       mapping_seal=mapping['seal'], metadata=metadata,
                       tables=[dict(columns=list(k), rows=v) for k, v in groups.items()]))
    if restore(view, mapping) != packet: raise ValueError('non-lossless source view')
    return view, mapping


def restore(view, mapping):
    verify(view); verify(mapping)
    if view['packet_seal'] != mapping['packet_seal'] or view['mapping_seal'] != mapping['seal']:
        raise ValueError('view/map binding mismatch')
    observations = {}
    for table in view['tables']:
        for row in table['rows']:
            if len(row) != len(table['columns']): raise ValueError('malformed source row')
            o = dict(zip(table['columns'], row))
            fields = [k for k in o if k.startswith('payload.')]
            if fields: o['payload'] = {k[8:]: o.pop(k) for k in fields}
            o = _replace(o, mapping['aliases'])
            if o['id'] in observations: raise ValueError('duplicate source row')
            observations[o['id']] = o
    packet = signed(dict(_replace(view['metadata'], mapping['aliases']), observations=observations))
    if packet['seal'] != view['packet_seal']: raise ValueError('source presentation mismatch')
    return packet


def translate(packet, view, mapping, annotation):
    if restore(view, mapping) != packet: raise ValueError('foreign source packet')
    out = deepcopy(annotation)
    for name in (*CLAIMS, None):
        obj = out if name is None else out[name]
        try: obj['citations'] = [mapping['aliases'][c] for c in obj['citations']]
        except KeyError as exc: raise ValueError('foreign citation alias') from exc
    validate_annotation(packet, out)
    return out


def review_receipt(packets, annotations, findings, *, source_receipt_sha, files):
    ids = {p['seal'] for p in packets}
    if (not packets or len(ids) != len(packets) or len(annotations) != len(ids)
            or len(findings) != len(ids) or {a['packet_seal'] for a in annotations} != ids
            or {f['packet_seal'] for f in findings} != ids):
        raise ValueError('fixed review roster incomplete or duplicated')
    by_id = {p['seal']: p for p in packets}
    for a in annotations: validate_annotation(by_id[a['packet_seal']], a)
    for f in findings:
        if (set(f) != {'packet_seal', 'classification', 'notes', 'citations'}
                or f['classification'] not in ('no_material_issue', 'judgment_difference', 'material_blocker')
                or not isinstance(f['notes'], str) or not f['notes'].strip()
                or not isinstance(f['citations'], list) or not f['citations']
                or any(c not in by_id[f['packet_seal']]['observations'] for c in f['citations'])):
            raise ValueError('explicit source-cited semantic adjudication required')
    blocked = any(f['classification'] == 'material_blocker' for f in findings)
    return signed(dict(schema='support-semantic-review-v1', source_receipt_sha=source_receipt_sha,
                       packet_seals=sorted(ids), annotations_seal=seal(annotations), findings=findings,
                       files=files, status='blocked' if blocked else 'exploratory_economics_cleared',
                       review_scope='one_model_source_fidelity_audit_not_predictive_trial',
                       trader_certified=False, edge_demonstrated=False, execution_authorized=False))


def verify_source():
    path = SOURCE_DIR/'receipt.json'
    if sha(path) != SOURCE_RECEIPT_SHA: raise ValueError('frozen support receipt changed')
    receipt = json.loads(path.read_text()); verify(receipt)
    files = {**receipt['files'], str(path): SOURCE_RECEIPT_SHA,
             **{str(SOURCE_DIR/k): v for k, v in receipt['artifacts'].items()}}
    verify_files(files)
    return files


def _write(path, value):
    with Path(path).open('x') as f: json.dump(value, f, sort_keys=True, separators=(',', ':'), allow_nan=False)


def export(output):
    files = verify_source()
    benchmark = json.loads((SOURCE_DIR/'benchmark.json').read_text()); verify(benchmark)
    packets = sorted(benchmark['packets'], key=lambda p: (clock(p['cutoff']), p['episode_id']))
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    (out/'answers').mkdir(); (out/'locked').mkdir()
    cases = []
    for i, p in enumerate(packets, 1):
        view, mapping = compact(p); name = f'{i:02d}'
        _write(out/(name+'.json'), view); _write(out/(name+'.mapping.json'), mapping)
        cases.append(dict(case=name, packet_seal=p['seal'], cutoff=p['cutoff'],
                          view_sha=sha(out/(name+'.json')), mapping_sha=sha(out/(name+'.mapping.json'))))
    verify_files(files)
    _write(out/'manifest.json', signed(dict(schema='support-review-export-v1', cases=cases,
                 source_receipt_sha=SOURCE_RECEIPT_SHA, files=files, independent_annotations_complete=False)))
    return cases


def read_case(directory, number):
    out = Path(directory); manifest = json.loads((out/'manifest.json').read_text()); verify(manifest)
    item = manifest['cases'][int(number)-1]
    if item['case'] != number: raise ValueError('case number mismatch')
    viewpath, mappath = out/(number+'.json'), out/(number+'.mapping.json')
    if sha(viewpath) != item['view_sha'] or sha(mappath) != item['mapping_sha']:
        raise ValueError('export modified')
    view, mapping = json.loads(viewpath.read_text()), json.loads(mappath.read_text())
    return restore(view, mapping), view, mapping


def lock(directory, number, answer_path):
    out = Path(directory)
    previous = None
    for i in range(1, int(number)):
        receipt = json.loads((out/'locked'/f'{i:02d}.receipt.json').read_text()); verify(receipt)
        if receipt['previous'] != previous or sha(out/'locked'/f'{i:02d}.json') != receipt['annotation_sha']:
            raise ValueError('prior annotation chain changed')
        previous = receipt['seal']
    packet, view, mapping = read_case(out, number)
    annotation = translate(packet, view, mapping, json.loads(Path(answer_path).read_text()))
    path = out/'locked'/(number+'.json')
    _write(path, annotation)
    receipt = signed(dict(case=number, packet_seal=packet['seal'], annotation_sha=sha(path),
                          previous=previous, status='schema_valid_judgment_not_adjudicated'))
    _write(out/'locked'/(number+'.receipt.json'), receipt)
    return receipt


def collect_locked(directory):
    out=Path(directory).resolve(); manifest=json.loads((out/'manifest.json').read_text()); verify(manifest)
    packets,annotations,files=[],[],{str(out/'manifest.json'):sha(out/'manifest.json')}
    previous=None
    for item in manifest['cases']:
        n=item['case']; packet,view,mapping=read_case(out,n)
        rp,ap=out/'locked'/(n+'.receipt.json'),out/'locked'/(n+'.json')
        receipt=json.loads(rp.read_text()); verify(receipt)
        if (receipt['case']!=n or receipt['packet_seal']!=packet['seal']
                or receipt['previous']!=previous or sha(ap)!=receipt['annotation_sha']):
            raise ValueError('annotation chain changed')
        annotation=json.loads(ap.read_text()); validate_annotation(packet,annotation)
        draft=out/'answers'/(n+'.json')
        if draft.exists() and translate(packet,view,mapping,json.loads(draft.read_text()))!=annotation:
            raise ValueError('draft differs from locked annotation')
        packets.append(packet); annotations.append(annotation); previous=receipt['seal']
        paths=[rp,ap,out/(n+'.json'),out/(n+'.mapping.json')]+([draft] if draft.exists() else [])
        files.update({str(p):sha(p) for p in paths})
    return packets,annotations,files


def adjudicate(directory, findings_path):
    # This consumes explicit controller findings; it never manufactures agreement.
    from scripts.research.support_reaction_economics import implementation_files
    out=Path(directory).resolve(); path=Path(findings_path).resolve()
    findings=json.loads(path.read_text())
    if not findings.get('reviewer') or not findings.get('independence_statement'):
        raise ValueError('reviewer identity and independence statement required')
    files=verify_source()
    packets,annotations,locked_files=collect_locked(out)
    benchmark=json.loads((SOURCE_DIR/'benchmark.json').read_text()); verify(benchmark)
    if len(packets)!=12 or {p['seal'] for p in packets}!={p['seal'] for p in benchmark['packets']}:
        raise ValueError('not the frozen twelve-case benchmark')
    files.update(locked_files); files.update(implementation_files()); files[str(path)]=sha(path)
    receipt=review_receipt(packets,annotations,findings['cases'],source_receipt_sha=SOURCE_RECEIPT_SHA,files=files)
    receipt=signed(dict(receipt,reviewer=findings['reviewer'],independence_statement=findings['independence_statement'],
                   limitations=findings.get('limitations',[]),review_directory=str(out)))
    verify_files(files); _write(out/'semantic_receipt.json',receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('export', 'show', 'lock', 'adjudicate'))
    parser.add_argument('--directory', required=True)
    parser.add_argument('--case'); parser.add_argument('--answer'); parser.add_argument('--findings')
    args = parser.parse_args()
    if args.action == 'export': print(json.dumps(export(args.directory)))
    elif args.action == 'lock': print(json.dumps(lock(args.directory, args.case, args.answer)))
    elif args.action == 'adjudicate':
        receipt=adjudicate(args.directory,args.findings)
        print(json.dumps({k:receipt[k] for k in ('status','seal','packet_seals')}))
    else:
        _, v, _ = read_case(args.directory, args.case)
        print(json.dumps({k: v[k] for k in ('packet_seal', 'metadata')}, separators=(',', ':')))
        for t in v['tables']:
            print(json.dumps(t['columns']))
            for row in t['rows']: print(json.dumps(row, separators=(',', ':')))


if __name__ == '__main__': main()
