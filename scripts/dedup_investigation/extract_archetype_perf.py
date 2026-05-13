"""
Parse the per-archetype performance breakdown from each ablation backtest.log
and emit a CSV table:
    mode,phase,archetype,direction,trades,wr,pf,pnl

Plus a wide-form CSV with trade counts per (mode,phase) -> archetype.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ABLATION = ROOT / 'results' / 'dedup_investigation' / 'ablation'

MODES = ['status_quo', 'normalized', 'unique_sl_zone',
         'round_robin', 'hybrid_rr_fusion', 'pass_through']
PHASES = ['train', 'test']

ROW_RE = re.compile(
    r'^\s+(?P<arch>[a-z_]+)\s+\|\s+(?P<dir>\w+)\s+\|\s+Trades:\s+(?P<n>\d+)'
    r'.*?WR:\s+(?P<wr>[0-9.]+)%.*?PF:\s+(?P<pf>[0-9.inf]+).*?PnL:\s+\$\s*(?P<pnl>[\-0-9.,]+)'
)


def parse_log(log_path: Path):
    rows = []
    if not log_path.exists():
        return rows
    capture = False
    saw_sep = False
    for line in log_path.read_text().splitlines():
        if 'ARCHETYPE PERFORMANCE BREAKDOWN' in line:
            capture = True
            saw_sep = False
            continue
        if not capture:
            continue
        if line.startswith('==='):
            if saw_sep:
                # second separator = end of section
                capture = False
                continue
            saw_sep = True
            continue
        m = ROW_RE.match(line)
        if m:
            rows.append({
                'archetype': m.group('arch'),
                'direction': m.group('dir'),
                'trades': int(m.group('n')),
                'win_rate': float(m.group('wr')),
                'profit_factor': m.group('pf'),
                'pnl': float(m.group('pnl').replace(',', '')),
            })
    return rows


def main():
    long_rows = []
    trade_counts = {}  # (mode,phase) -> {arch: trades}
    for mode in MODES:
        for phase in PHASES:
            log = ABLATION / mode / phase / 'backtest.log'
            parsed = parse_log(log)
            tc = {}
            for r in parsed:
                long_rows.append({'mode': mode, 'phase': phase, **r})
                tc[r['archetype']] = r['trades']
            trade_counts[(mode, phase)] = tc

    # Write long form.
    long_csv = ABLATION / 'ARCHETYPE_PERF_LONG.csv'
    with long_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'mode', 'phase', 'archetype', 'direction',
            'trades', 'win_rate', 'profit_factor', 'pnl',
        ])
        writer.writeheader()
        writer.writerows(long_rows)

    # Write wide form (trades per archetype across modes, OOS only).
    all_archs = sorted({r['archetype'] for r in long_rows})
    wide_csv = ABLATION / 'TRADE_COUNTS_BY_MODE.csv'
    with wide_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        for phase in PHASES:
            writer.writerow([f'== {phase.upper()} ==', '', '', '', '', '', ''])
            writer.writerow(['archetype'] + MODES)
            for a in all_archs:
                row = [a]
                for m in MODES:
                    row.append(trade_counts.get((m, phase), {}).get(a, 0))
                writer.writerow(row)
            writer.writerow([])

    print(f'wrote {long_csv}')
    print(f'wrote {wide_csv}')

    # Pretty-print OOS table.
    print('\n=== OOS (TEST) — TRADES BY ARCHETYPE × MODE ===')
    hdr = f'{"archetype":<22}' + ''.join(f'{m[:10]:>11}' for m in MODES)
    print(hdr)
    print('-' * len(hdr))
    for a in all_archs:
        row = f'{a:<22}'
        for m in MODES:
            row += f'{trade_counts.get((m, "test"), {}).get(a, 0):>11}'
        print(row)
    # totals.
    totals = []
    for m in MODES:
        s = sum(trade_counts.get((m, 'test'), {}).values())
        totals.append(s)
    print('-' * len(hdr))
    print(f'{"TOTAL":<22}' + ''.join(f'{t:>11}' for t in totals))


if __name__ == '__main__':
    main()
