"""
Phase 3 dedup mode ablation — WFO backtests across dedup modes.

Train window: 2018-01-01 → 2022-12-31
Test  window: 2023-01-01 → 2024-12-31

For each mode, runs both train and test backtests and records:
  total_trades, profit_factor, net_pnl, max_drawdown_pct, sharpe_ratio

Per-archetype trade share is parsed from the backtester's "Per Archetype"
stdout section.

Modes tested:
    status_quo, normalized, unique_sl_zone, round_robin, hybrid_rr_fusion,
    pass_through

Output: results/dedup_investigation/ablation/<mode>/<phase>/{summary.json, log}
        results/dedup_investigation/ablation/SUMMARY.csv
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

MODES = [
    'status_quo',
    'normalized',
    'unique_sl_zone',
    'round_robin',
    'hybrid_rr_fusion',
    'pass_through',
]

PHASES = {
    'train': ('2020-01-01', '2022-12-31'),
    'test':  ('2023-01-01', '2024-12-31'),
}


def run_one(mode: str, phase: str, start: str, end: str) -> dict:
    out_dir = ROOT / 'results' / 'dedup_investigation' / 'ablation' / mode / phase
    out_dir.mkdir(parents=True, exist_ok=True)
    dedup_csv = out_dir / 'dedup_events.csv'
    if dedup_csv.exists():
        dedup_csv.unlink()
    log_path = out_dir / 'backtest.log'
    summary_path = out_dir / 'summary.json'

    env = os.environ.copy()
    env['DEDUP_MODE'] = mode
    env['DEDUP_LOG_PATH'] = str(dedup_csv)

    backtest = ROOT / 'bin' / 'backtest_v11_standalone.py'
    code = (
        'import sys; sys.path.insert(0, r"' + str(ROOT) + '");\n'
        'import scripts.dedup_investigation.dedup_patch as p; p.apply_patch();\n'
        'import runpy; sys.argv = ' + repr([
            str(backtest),
            '--config', 'configs/bull_machine_isolated_v11_fixed.json',
            '--start-date', start,
            '--end-date', end,
            '--commission-rate', '0.0002',
            '--slippage-bps', '3',
            '--initial-cash', '100000',
        ]) + '; runpy.run_path(sys.argv[0], run_name="__main__")'
    )

    t0 = time.time()
    proc = subprocess.run(
        ['python3', '-c', code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    elapsed = time.time() - t0
    log_path.write_text(proc.stdout + '\n--- STDERR ---\n' + proc.stderr)
    summary = parse_summary(proc.stdout)
    summary['mode'] = mode
    summary['phase'] = phase
    summary['elapsed_sec'] = round(elapsed, 1)
    summary['returncode'] = proc.returncode

    # Trade share by archetype.
    summary['archetype_trades'] = parse_archetype_trades(proc.stdout)

    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def parse_summary(stdout: str) -> dict:
    out = {}
    patterns = {
        'total_trades':       r'Total Trades:\s+(\d+)',
        'profit_factor':      r'Profit Factor:\s+([0-9.]+|inf)',
        'net_pnl':            r'(?:Net PnL|Net P&L|Total PnL):\s+\$?([\-0-9.,]+)',
        'max_drawdown_pct':   r'Max Drawdown:\s+([\-0-9.,]+)%',
        'max_drawdown_usd':   r'Max Drawdown \$:\s+\$?([\-0-9.,]+)',
        'sharpe_ratio':       r'Sharpe Ratio:\s+([\-0-9.,]+)',
        'win_rate':           r'Win Rate:\s+([0-9.]+)%',
    }
    for key, pat in patterns.items():
        m = re.search(pat, stdout)
        if m:
            v = m.group(1).replace(',', '').replace('$', '')
            try:
                out[key] = float(v)
            except ValueError:
                out[key] = v
    return out


def parse_archetype_trades(stdout: str) -> dict:
    """
    Parse per-archetype trade counts from a block like:
        Per-Archetype Performance:
        ------------------------------------------------------------
          archetype_name      ...     trades=N  pf=X  pnl=Y
    """
    out = {}
    # Try several known formats.
    for line in stdout.splitlines():
        m = re.match(
            r'\s*(?P<name>[a-z_]+)\s*[:|]?\s*(?:trades?=)?(?P<n>\d+)\s+(?:trades?\b|pf=|wr=)',
            line,
        )
        if m and m.group('name') in {
            'spring', 'wick_trap', 'liquidity_sweep', 'retest_cluster',
            'trap_within_trend', 'failed_continuation', 'order_block_retest',
            'liquidity_vacuum', 'long_squeeze', 'funding_divergence',
            'fvg_continuation', 'volume_fade_chop', 'whipsaw',
            'liquidity_compression', 'exhaustion_reversal',
            'confluence_breakout', 'oi_divergence',
        }:
            out[m.group('name')] = int(m.group('n'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', nargs='*', default=MODES)
    ap.add_argument('--phases', nargs='*', default=list(PHASES.keys()))
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args()

    rows = []
    for mode in args.modes:
        for phase in args.phases:
            start, end = PHASES[phase]
            sp = ROOT / 'results' / 'dedup_investigation' / 'ablation' / mode / phase / 'summary.json'
            if args.skip_existing and sp.exists():
                print(f'[ablation] {mode}/{phase}: skipping (already exists)')
                summary = json.loads(sp.read_text())
            else:
                print(f'[ablation] {mode}/{phase} ({start}..{end})')
                summary = run_one(mode, phase, start, end)
                print(
                    f'[ablation]   PF={summary.get("profit_factor"):.3f} '
                    f'PnL={summary.get("net_pnl")} '
                    f'trades={summary.get("total_trades")} '
                    f'dd={summary.get("max_drawdown_pct")} '
                    f'({summary.get("elapsed_sec")}s)'
                ) if summary.get('profit_factor') is not None else print(f'[ablation]   FAILED rc={summary.get("returncode")}')
            rows.append(summary)

    # Write summary CSV.
    out_csv = ROOT / 'results' / 'dedup_investigation' / 'ablation' / 'SUMMARY.csv'
    with out_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'mode', 'phase', 'total_trades', 'profit_factor', 'net_pnl',
            'max_drawdown_pct', 'sharpe_ratio', 'win_rate', 'elapsed_sec',
        ])
        for r in rows:
            writer.writerow([
                r.get('mode'), r.get('phase'), r.get('total_trades'),
                r.get('profit_factor'), r.get('net_pnl'),
                r.get('max_drawdown_pct'), r.get('sharpe_ratio'),
                r.get('win_rate'), r.get('elapsed_sec'),
            ])
    print(f'[ablation] summary written to {out_csv}')


if __name__ == '__main__':
    main()
