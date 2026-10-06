"""Deterministic teaching-derived candles; never imports a detector.

Prices/timing/volumes are illustrative project choices, not recovered X charts.
Row timestamps are completed hourly bar starts. Checkpoint n includes rows [:n].
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'wyckoff-recognition-input-v1'
COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _interpolate(anchors, i):
    for (a, x), (b, y) in zip(anchors, anchors[1:]):
        if a <= i <= b:
            return x + (y-x) * (i-a) / (b-a)
    raise ValueError('uncovered anchor')


def _candles(kind='spring'):
    # Decline -> stopping action -> repeated bounds -> test -> escape -> retest.
    anchors = [(0,150), (480,125), (576,114), (579,112), (580,102),
               (586,109.5), (592,105), (600,102), (607,108.5), (614,102.5),
               (621,109), (628,103), (635,106), (640,102.5), (644,102.2),
               (648,102), (652,109), (656,112), (660,111), (665,115.5)]
    special = {580:[112,113,99,102,900], 586:[108,110,107.5,109.5,260],
               600:[103,103.5,100.5,102,140], 607:[107.8,109.5,107.5,108.5,130],
               614:[103,103.4,101,102.5,100], 621:[108.5,110,108,109,110],
               628:[103.5,104,101.7,103,85], 644:[102.5,103.2,98.8,102.2,250],
               648:[102.5,102.8,100.5,102,70], 652:[104,110,103.8,109,340],
               656:[109,113,108.8,112,500], 660:[111.5,111.8,110.2,111,65],
               665:[112.8,116,112.5,115.5,300]}
    if kind == 'no_spring':
        anchors = [(i, 103 if i == 644 else 103.5 if i == 648 else v) for i,v in anchors]
        special[644] = [103.5,104,101.8,103,70]
        special[648] = [104,104.3,102.5,103.5,60]
    if kind == 'missing_range':
        anchors = [(0,150),(576,120),(620,112),(639,104),(644,102.2),(652,97),(665,94)]
        special = {644:[103,103.5,98.8,102.2,250]}
    if kind == 'missing_recovery':
        anchors = [(i,v) for i,v in anchors if i < 644] + [(644,99),(648,97),(665,94)]
        special = {i:v for i,v in special.items() if i < 644}
        special[644] = [102.5,102.8,98,99,400]
    if kind == 'failed_test':
        anchors = [(i,v) for i,v in anchors if i < 648] + [(648,99),(656,96),(665,94)]
        special = {i:v for i,v in special.items() if i < 648}
        special[648] = [103,105,97,99,700]
    if kind == 'invalidated_parent':
        anchors = [(i,v) for i,v in anchors if i < 635] + [
            (635,101),(638,95),(641,94),(644,96),(648,95),(665,93)]
        special = {i:v for i,v in special.items() if i < 635}
        special.update({638:[101,101.3,94,95,700],644:[94.5,96.5,92.5,96,220]})
    if kind == 'climax_continuation':
        anchors = [(0,150),(576,114),(579,112),(580,102),(586,96),(665,90)]
        special = {580:special[580]}
    if kind == 'local_rebound':
        anchors = [(i,v) for i,v in anchors if i < 652] + [
            (652,105),(656,106),(660,104.5),(665,105)]
        special = {i:v for i,v in special.items() if i < 652}
        special[656] = [104.8,106.5,104.5,106,180]
    if kind == 'conflicting_tf':
        anchors = [(0,170),(100,135),(140,155),(250,110),(300,140),
                   (400,104),(480,130),(550,120)] + [(i,v) for i,v in anchors if i >= 576]
    start = datetime(2024,1,1,tzinfo=timezone.utc)
    rows = []
    for i in range(666):
        close = _interpolate(anchors, i)
        previous = _interpolate(anchors, max(0,i-1))
        op = previous
        spread = .3 + .05*(i % 5)
        vol = 100 + 8*(i % 7)
        if i >= 600:
            vol = 85 + 5*(i % 5)
        vals = special.get(i, [op,max(op,close)+spread,min(op,close)-spread,close,vol])
        rows.append([(start+timedelta(hours=i)).isoformat()] + [round(v,6) for v in vals])
    return rows


def _card(num, name, category, kind, direction, interpretation, required, prohibited,
          checkpoints, range_status='established'):
    rows = _candles(kind)
    if direction == 'short':
        rows = [[t,round(210-o,6),round(210-l,6),round(210-h,6),round(210-c,6),v]
                for t,o,h,l,c,v in rows]
    cps = [{'n':n,'stage':stage} for n,stage in checkpoints]
    # Post-cutoff candles are retained solely for explicit future-tail witnesses.
    return {'id':f'W{num:02d}', 'name':name, 'category':category,
            'direction':direction, 'synthetic':True,
            'source_ids':['WA','WI_MIRROR','ZI_MIRROR'], 'candles':rows,
            'range':{'low':99.0,'high':110.0,'established_by_n':636,
                     'status':range_status,'timeframe':'1h'},
            'landmarks':{'stopping_action':580,'reaction':586,'secondary_test':600,
                         'candidate':644,'candidate_test':648,'escape':656,'retest':660},
            'checkpoints':cps,'interpretation':interpretation,
            'required_milestones':required, 'prohibited_claims':prohibited,
            'uncertainty':'Illustrative hourly interpretation; no guaranteed phase, intent, trade or profit. '
                          'Higher timeframes are independent interpretations, not automatically confirmed.'}


def build_packet():
    cp = [(636,'range'),(645,'candidate'),(649,'test'),(657,'escape'),(666,'retest_hold')]
    cases = [
        _card(1,'spring_accumulation','positive','spring','long',
              'Established hourly range, undercut/re-entry, quieter higher test, demand escape and support retest.',
              ['spring','sos','lps'],[],cp),
        _card(2,'no_spring_accumulation','positive','no_spring','long',
              'Supply contracts above established support; no spring required before demand escape and support retest.',
              ['sos','lps'],['spring'],cp),
        _card(3,'upthrust_distribution','positive','spring','short',
              'Mirrored supply structure: established range, upside excursion/rejection, quieter test, weakness and underside retest.',
              ['upthrust','sow','lpsy'],[],cp),
        _card(4,'no_upthrust_distribution','positive','no_spring','short',
              'Demand contracts below resistance; no upthrust required before weakness and underside retest.',
              ['sow','lpsy'],['upthrust'],cp),
        _card(5,'missing_range','near_miss','missing_range','long',
              'A downside wick during a decline is not a spring of an established range.',[],
              ['spring','sos','lps','confirmed_C_or_D'],[(641,'decline'),(645,'wick'),(653,'continuation')],'absent'),
        _card(6,'missing_recovery','near_miss','missing_recovery','long',
              'Range support breaks; no recovery inside it. Do not claim a completed spring.',[],
              ['spring','sos','lps','confirmed_C_or_D'],[(636,'range'),(645,'break_without_reclaim'),(649,'continuation')]),
        _card(7,'failed_test','near_miss','failed_test','long',
              'Initial reclaim is plausible; later wide high-volume test breaks the spring low. The long thesis is invalidated.',[],
              ['active_long_after_invalidation'],[(645,'candidate'),(649,'failed_test'),(657,'continuation')]),
        _card(8,'invalidated_parent','near_miss','invalidated_parent','long',
              'The original range breaks before the later wick. That wick cannot inherit the invalidated parent.',[],
              ['spring','sos','lps','confirmed_C_or_D'],[(636,'range'),(642,'parent_broken'),(645,'new_wick'),(649,'below_old_range')],'invalidated_at_639'),
        _card(9,'climax_or_continuation','ambiguous','climax_continuation','long',
              'Stopping-looking candle is provisional; subsequent lower prices do not confirm a completed accumulation.',[],
              ['spring','sos','lps','confirmed_C_or_D'],[(581,'possible_climax'),(587,'continued_decline')],'unestablished'),
        _card(10,'developing_range','ambiguous','spring','long',
              'Stopping action, reaction and initial test permit a developing range, not a confirmed directional phase C/D.',[],
              ['spring','sos','lps','confirmed_C_or_D'],[(581,'possible_climax'),(587,'reaction'),(601,'initial_test')],'developing'),
        _card(11,'local_rebound_not_range_escape','ambiguous','local_rebound','long',
              'A plausible spring/test produces only local recovery below resistance. Do not promote this to parent escape or LPS after escape.',[],
              ['parent_range_escape','lps_after_parent_escape'],[(636,'range'),(645,'candidate'),(649,'test'),(657,'local_recovery'),(666,'still_inside')]),
        _card(12,'conflicting_timeframes','ambiguous','conflicting_tf','long',
              'Local spring and escape coexist with earlier larger lower highs/lows. A local turn does not prove higher-timeframe accumulation or lineage.',[],
              ['confirmed_nested_accumulation'],cp),
    ]
    # Reflection transforms the explanatory parent geometry too.
    for c in cases:
        if c['direction'] == 'short':
            c['range'].update(low=100.0,high=111.0)
    # Source-only review corrections, made BEFORE any detector run. Landmarks
    # assert only observed events, not every neutral point on the shared grid.
    landmark_overrides = {
        'W05':{'wick':644},
        'W06':{'stopping_action':580,'reaction':586,'secondary_test':600,'break':644},
        'W07':{'stopping_action':580,'reaction':586,'secondary_test':600,'candidate':644,'failed_test':648},
        'W08':{'stopping_action':580,'reaction':586,'secondary_test':600,'invalidation':638,'later_wick':644},
        'W09':{'possible_climax':580,'continuation':586},
        'W10':{'stopping_action':580,'reaction':586,'initial_test':600},
        'W11':{'stopping_action':580,'reaction':586,'secondary_test':600,'candidate':644,
               'candidate_test':648,'local_recovery':656,'local_pullback':660},
    }
    for c in cases:
        if c['id'] in landmark_overrides:
            c['landmarks'] = landmark_overrides[c['id']]
        if c['id'] in ('W05','W09'):
            c['range'].update(low=None,high=None,established_by_n=None)
        if c['id'] == 'W10':
            c['range']['established_by_n'] = None
    return {'schema':SCHEMA,'columns':COLUMNS,'provenance':'synthetic_not_reconstructed_X_charts',
            'sources':{
                'WA':{'url':'https://www.wyckoffanalytics.com/wyckoff-method/',
                      'access':'fresh_primary_text','role':'qualitative Wyckoff grammar; no numeric thresholds'},
                'WI_MIRROR':{'url':'https://x.com/Wyckoff_Insider/status/2097647324776239397',
                             'readable_url':'https://twstalker.com/Wyckoff_Insider',
                             'access':'fresh_third_party_text_no_image','role':'context and staged confirmation'},
                'ZI_MIRROR':{'url':'https://x.com/IamZeroIka/status/1779921797434982462',
                             'readable_url':'https://threadreaderapp.com/thread/1779921797434982462.html',
                             'access':'fresh_third_party_text_no_image','role':'location, closes, timeframe roles'}},
            'grading_scope':'Native hourly milestones and conditional consumer effects; provisional scores are not entries.',
            'cases':cases}


def validate_packet(p):
    if set(p) != {'schema','columns','provenance','sources','grading_scope','cases'} or p['schema'] != SCHEMA or p['columns'] != COLUMNS:
        raise ValueError('unsupported packet schema')
    cases = p['cases']
    if len(cases) != 12 or {c['id'] for c in cases} != {f'W{i:02d}' for i in range(1,13)}:
        raise ValueError('exactly twelve unique cases required')
    if Counter(c['category'] for c in cases) != {'positive':4,'near_miss':4,'ambiguous':4}:
        raise ValueError('unbalanced categories')
    for c in cases:
        if c['synthetic'] is not True or c['direction'] not in ('long','short'):
            raise ValueError('bad case direction/provenance')
        if not c['source_ids'] or any(s not in p['sources'] for s in c['source_ids']):
            raise ValueError('unknown source')
        previous = None
        if len(c['candles']) < 24*21:
            raise ValueError('insufficient daily warmup')
        for row in c['candles']:
            if len(row) != 6:
                raise ValueError('only raw OHLCV accepted')
            dt = datetime.fromisoformat(row[0])
            if dt.utcoffset() != timedelta(0) or dt.minute or dt.second or dt.microsecond:
                raise ValueError('UTC hourly starts required')
            if previous is not None and dt-previous != timedelta(hours=1):
                raise ValueError('non-contiguous candles')
            previous = dt
            o,h,l,cl,v = row[1:]
            if not all(isinstance(x,(int,float)) and math.isfinite(x) and x>0 for x in row[1:]):
                raise ValueError('invalid numeric candle')
            if l > min(o,cl) or h < max(o,cl) or l >= h:
                raise ValueError('invalid OHLC geometry')
        ns = [cp['n'] for cp in c['checkpoints']]
        if not ns or ns != sorted(set(ns)) or any(type(n) is not int or not 504 <= n <= len(c['candles']) for n in ns):
            raise ValueError('invalid checkpoints')
        low, high = c['range']['low'], c['range']['high']
        if low is None and high is None and c['range']['status'] in ('absent','unestablished'):
            continue
        if low is None or high is None or not low < high:
            raise ValueError('invalid parent range')
    canonical(p)


def write_packet(path, p):
    validate_packet(p)
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        f.write(canonical(p))


def read_packet(path):
    p = json.loads(Path(path).read_text())
    validate_packet(p)
    return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    packet = build_packet()
    write_packet(args.output,packet)
    print(json.dumps({'cases':12,'packet_sha256':digest(packet),'output':args.output}))
