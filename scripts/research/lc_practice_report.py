"""Self-contained owner-only report; never used as outcome-hidden model input."""
import base64
import hashlib
from html import escape
import io
import json
import math
import os
from pathlib import Path
import tempfile

import pandas as pd

from scripts.research.lc_practice_replay import score_run
from scripts.research.lc_structure_packet import resolve


def _money(value):
    return 'unavailable' if value is None else f'${value:+,.2f}'


def _write_once(path, raw):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != raw: raise ValueError('immutable report differs: '+str(path))
        return path
    fd, temporary = tempfile.mkstemp(prefix='.report-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        try: os.link(temporary,path)
        except FileExistsError:
            if path.read_bytes()!=raw: raise ValueError('concurrent report differs')
        fd=os.open(str(path.parent),os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally: os.unlink(temporary)
    return path


def _anchor(cid,eid):
    return 'e-'+hashlib.sha256((cid+'\0'+eid).encode()).hexdigest()[:24]


def _chart(case):
    packet=case['packet']
    if not packet: return '<p>Source unavailable; no chart invented.</p>'
    with tempfile.TemporaryDirectory(prefix='lc-matplotlib-') as cache:
        prior=os.environ.get('MPLCONFIGDIR');os.environ['MPLCONFIGDIR']=cache
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.dates as dates
            fig=plt.figure(figsize=(14,12),layout='constrained')
            grid=fig.add_gridspec(4,2)
            for n,tf in enumerate(('1d','4h','1h','15m','5m','1m')):
                ax=fig.add_subplot(grid[n//2,n%2]); rows=packet['candles'][tf]
                if rows:
                    times=pd.to_datetime([r['observation_end'] for r in rows],utc=True)
                    ax.vlines(times,[r['low'] for r in rows],[r['high'] for r in rows],color='#68778d',lw=1)
                    ax.plot(times,[r['close'] for r in rows],'.-',color='#087f8c',lw=1)
                    for parent in ('parent_1d','parent_4h'):
                        bound=packet['context'][parent]['bound']
                        if bound:
                            for key in ('range_low','range_high'):
                                ax.axhline(bound[key],ls=':',alpha=.3,color='#b46d23')
                else: ax.text(.5,.5,'Missing evidence',ha='center',transform=ax.transAxes)
                ax.set_title(tf+' pre-decision: OHLC ranges / closes')
                ax.xaxis.set_major_formatter(dates.DateFormatter('%m-%d %H:%M',tz=__import__('datetime').timezone.utc))
                ax.tick_params(axis='x',rotation=20,labelsize=7);ax.grid(alpha=.15)
            ax=fig.add_subplot(grid[3,:]); decision=pd.Timestamp(case['decision_time'])
            prefix=[dict(r,open_time=r['open_time']) for r in packet['candles']['1m']]
            rows=prefix+case['future_minutes']
            values={}
            for r in rows:
                try:
                    vals=[float(r[k]) for k in ('open','high','low','close','volume')]
                    valid=all(math.isfinite(v) for v in vals)
                    at=pd.Timestamp(r['open_time'])+pd.Timedelta(minutes=1)
                    values[at]=vals[3] if valid else float('nan')
                except (KeyError,TypeError,ValueError): continue
            if values:
                series=pd.Series(values).sort_index()
                series=series.reindex(pd.date_range(series.index.min(),series.index.max(),freq='min'))
                ax.plot(series.index,series.values,color='#176b83',lw=.8,label='Minute close (gaps preserved)')
            ax.axvline(decision,color='black',ls='--',label='Decision / future starts')
            if case['elapsed_seconds'] is not None:
                ax.axvline(decision+pd.Timedelta(seconds=case['elapsed_seconds']),color='#ad3d8f',ls=':',label='Response available')
            checked=case['validation']; proposal=checked.get('proposal') if checked else None
            if proposal and proposal['plan']:
                plan=proposal['plan']
                for field,color in [('stop_level_id','#c62828'),('destination_level_id','#267343')]:
                    ax.axhline(packet['levels'][plan[field]]['price'],color=color,ls='--',label=field.replace('_level_id',''))
                if plan['trigger']['kind']=='close_above':
                    ax.axhline(packet['levels'][plan['trigger']['level_id']]['price'],color='#926a00',ls=':',label='Trigger')
                for eid in plan['obstacle_level_ids']:
                    ax.axhline(packet['levels'][eid]['price'],color='#777777',alpha=.06,lw=.5)
            out=case['arms']['agent'].get('outcome')
            if out:
                ax.scatter([pd.Timestamp(out['entry_time'])],[out['entry_price']],marker='^',color='#267343',s=55,label='Agent fill')
                ax.scatter([pd.Timestamp(out['exit_observed_at'])],[out['exit_price']],marker='x',color='#c62828',s=55,label='Exit observed (intrabar time may be unknown)')
            ax.set_title('Owner-only future replay — minute closes; no downsampling')
            ax.legend(fontsize=7,loc='best');ax.grid(alpha=.15)
            ax.xaxis.set_major_formatter(dates.DateFormatter('%m-%d %H:%M',tz=__import__('datetime').timezone.utc))
            buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=100);plt.close(fig)
            payload=base64.b64encode(buffer.getvalue()).decode('ascii')
            return '<img alt="Nested timeframe evidence and future minute replay" src="data:image/png;base64,'+payload+'">'
        finally:
            if prior is None: os.environ.pop('MPLCONFIGDIR',None)
            else: os.environ['MPLCONFIGDIR']=prior


def render_report(run):
    result=score_run(run); cases=result['cases']; summary=result['summary']
    rows=[]; cards=[]
    for c in cases:
        arms=c['arms'];cid=c['case_id']
        rows.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in
            [cid,c['subtype'],c['terminal_status'],arms['agent']['status'],_money(arms['agent']['net_pnl']),
             _money(arms['mechanical_matched']['net_pnl']),_money(arms['mechanical_90s']['net_pnl']),
             _money(arms['legacy_immediate']['net_pnl'])])+'</tr>')
        rationale=[]; checked=c['validation']; proposal=checked.get('proposal') if checked else None
        if proposal:
            for group in ('parent_child','sequence','supporting','opposing','competing_explanation','unknowns'):
                claims=proposal[group] if isinstance(proposal[group],list) else [proposal[group]]
                for claim in claims:
                    links=' '.join('<a href="#'+_anchor(cid,eid)+'">'+escape(eid)+'</a>' for eid in claim['evidence_ids'])
                    rationale.append('<p><b>'+escape(group)+':</b> '+escape(claim['text'])+' '+links+'</p>')
        catalog=[]
        if c['packet']:
            for eid in c['packet']['citation_catalog']:
                catalog.append('<dt id="'+_anchor(cid,eid)+'">'+escape(eid)+'</dt><dd><pre>'+escape(
                    json.dumps(resolve(c['packet'],eid),indent=2,ensure_ascii=False))+'</pre></dd>')
        out=arms['agent'].get('outcome')
        explanation=(f"Filled at {out['entry_price']:,.2f}; {out['exit_reason']} at {out['exit_price']:,.2f}; "
                     f"modeled net {_money(out['net_pnl'])}." if out else
                     f"{arms['agent']['status']}: {arms['agent']['reason']}; modeled net {_money(arms['agent']['net_pnl'])}.")
        cards.append('<section><h2>'+escape(cid)+'</h2><p>'+escape(explanation)+'</p><p>Source subtype: '+
            escape(c['subtype'])+'; terminal: '+escape(c['terminal_status'])+'; elapsed seconds: '+escape(str(c['elapsed_seconds']))+
            '. Interpretation is unreviewed; citation/schema checks are not proof of the trading thesis.</p>'+_chart(c)+
            ''.join(rationale)+'<details><summary>Exact original response and recorded model metadata</summary><pre>'+escape(c['raw_response'] or 'No response')+
            '</pre><pre>'+escape(json.dumps(c['metadata'],indent=2))+'</pre></details>'+
            '<details><summary>Validation and per-arm outcome records</summary><pre>'+escape(json.dumps(dict(validation=checked,arms=arms),indent=2))+
            '</pre></details><details><summary>All source citations and observed barriers</summary><dl>'+''.join(catalog)+'</dl></details></section>')
    overview='<table><thead><tr>'+''.join('<th>'+s+'</th>' for s in
        ['Case (UTC)','Source subtype','Capture','Agent action','Agent net','Matched rule net','Rule at 90s','Legacy reference*'])+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    scorecards=[]
    for subtype,group in summary['by_subtype'].items():
        matched=group['matched']
        scorecards.append('<h3>'+escape(subtype)+'</h3><p>Matched coverage: '+str(matched['count'])+'/'+
            str(matched['roster_count'])+'; known delta: '+_money(matched['known_delta'])+
            '; full subtype delta: '+_money(matched['total_delta'])+'</p><table><tr><th>Comparison</th><th>Cases</th></tr>'+''.join(
                '<tr><td>'+escape(label)+'</td><td>'+str(count)+'</td></tr>' for label,count in sorted(group['attribution_counts'].items()))+'</table>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LC practice report</title><style>body{font:15px system-ui;background:#101620;color:#e6edf7;margin:30px auto;max-width:1400px;padding:0 20px}h1,h2{color:#7cdddb}section{background:#192332;padding:22px;margin:24px 0;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border-bottom:1px solid #465164;padding:10px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}img{width:100%;background:white;border-radius:6px}a{color:#7cdddb}details{margin:12px 0}dd{margin-left:10px}</style>
<h1>LC practice: agent decisions versus code</h1>'''+\
        '<p>'+escape(result['exposure'])+'. Independent case replays; not a funded portfolio or certification of profitability. '+escape(result['costs'])+'.</p>'+\
        '<p>This entire report contains outcomes and must NEVER be supplied to an outcome-hidden assessor. No actual orders.</p>'+\
        '<p>*Legacy uses different sizing, stop/2R target and deadline; its dollars do not measure incremental agent value. Rejected losses are avoided counterfactuals, not earned profits.</p>'+overview+\
        '<h2>Coverage, separate subtypes and matched comparisons</h2>'+''.join(scorecards)+\
        '<details><summary>Full accounting and coverage</summary><pre>'+escape(json.dumps(summary,indent=2))+'</pre></details>'+\
        '<p>Overlap pair counts (no case suppresses another): '+escape(json.dumps(result['overlap_pair_counts']))+'</p>'+''.join(cards)+'</html>'
    markdown=['# LC practice result',result['exposure'],result['costs'],
        'Independent case results, not account return. Interpretations unreviewed.']
    table=['| Case | Agent action | Agent net | Matched rule net |','|---|---|---:|---:|']
    for c in cases:
        table.append('| '+c['case_id']+' | '+c['arms']['agent']['status']+' | '+_money(c['arms']['agent']['net_pnl'])+' | '+_money(c['arms']['mechanical_matched']['net_pnl'])+' |')
    markdown.append('\n'.join(table))
    markdown.extend(['','Full agent total: '+_money(summary['overall']['agent']['total_net_pnl']),
        'Known subtotal: '+_money(summary['overall']['agent']['known_subtotal'])+' across '+str(summary['overall']['agent']['known_count'])+' known cases.',
        'Matched-complete delta: '+_money(summary['matched']['known_delta'])+' across '+str(summary['matched']['count'])+' pairs.',
        'Full-roster matched delta: '+_money(summary['matched']['total_delta']),
        'No live promotion. This exposed batch cannot establish a dependable edge.'])
    return dict(html=_write_once(run.root/'report.html',page.encode('utf-8')),
                markdown=_write_once(run.root/'summary.md',('\n\n'.join(markdown)+'\n').encode('utf-8')),
                json=run.root/'case_results.json')
