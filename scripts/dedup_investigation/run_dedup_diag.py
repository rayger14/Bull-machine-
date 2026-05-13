"""
Phase 1 diagnostic: run the backtester with the dedup-logging patch active
on the configured mode (default: status_quo / best_per_direction) and emit
a CSV of every (winner, loser) pair at every dedup event.

Output:
    results/dedup_investigation/<run_id>/dedup_events.csv  (raw winner/loser pairs)
    results/dedup_investigation/<run_id>/dedup_matrix.csv  (per-archetype matrix)
    results/dedup_investigation/<run_id>/backtest_summary.json (PF, PnL, etc.)
    results/dedup_investigation/<run_id>/backtest.log

Usage:
    python3 scripts/dedup_investigation/run_dedup_diag.py \
        --start-date 2020-01-01 --end-date 2024-12-31 \
        --mode status_quo --run-id phase1_full

If --mode is omitted, no DEDUP_MODE env override is applied and the engine's
config-driven mode (best_per_direction) is used.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True, help='subdir under results/dedup_investigation/')
    ap.add_argument('--start-date', default='2020-01-01')
    ap.add_argument('--end-date', default='2024-12-31')
    ap.add_argument('--mode', default='', help='Override DEDUP_MODE (default: empty = use config)')
    ap.add_argument('--config', default='configs/bull_machine_isolated_v11_fixed.json')
    ap.add_argument('--initial-cash', default='100000')
    ap.add_argument('--no-build-matrix', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = ROOT / 'results' / 'dedup_investigation' / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dedup_csv = out_dir / 'dedup_events.csv'
    if dedup_csv.exists():
        dedup_csv.unlink()
    summary_json = out_dir / 'backtest_summary.json'
    log_path = out_dir / 'backtest.log'

    env = os.environ.copy()
    env['DEDUP_LOG_PATH'] = str(dedup_csv)
    if args.mode:
        env['DEDUP_MODE'] = args.mode
    else:
        env.pop('DEDUP_MODE', None)

    # PYTHONSTARTUP trick: ensure the patch module is imported before the
    # backtester runs. Easiest approach: use a sitecustomize.py-like loader.
    # We achieve this with -c shim that imports the patch first.
    backtest = ROOT / 'bin' / 'backtest_v11_standalone.py'
    code = (
        'import sys; sys.path.insert(0, r"' + str(ROOT) + '");\n'
        'import scripts.dedup_investigation.dedup_patch as p; p.apply_patch();\n'
        'import runpy; sys.argv = ' + repr([
            str(backtest),
            '--config', args.config,
            '--start-date', args.start_date,
            '--end-date', args.end_date,
            '--commission-rate', '0.0002',
            '--slippage-bps', '3',
            '--initial-cash', args.initial_cash,
        ]) + '; runpy.run_path(sys.argv[0], run_name="__main__")'
    )

    print(f'[diag] running backtest {args.start_date}..{args.end_date} mode={args.mode or "(default)"} -> {out_dir}')
    proc = subprocess.run(
        ['python3', '-c', code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    log_path.write_text(proc.stdout + '\n--- STDERR ---\n' + proc.stderr)

    # Parse backtest summary from stdout.
    out = proc.stdout
    summary = parse_backtest_summary(out)
    summary_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f'[diag] PF={summary.get("profit_factor")} PnL={summary.get("net_pnl")} '
          f'trades={summary.get("total_trades")} dd={summary.get("max_drawdown_pct")}')

    if proc.returncode != 0:
        print('[diag] backtester returned non-zero; see backtest.log for details', file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        sys.exit(proc.returncode)

    if not args.no_build_matrix:
        build_matrix(dedup_csv, out_dir / 'dedup_matrix.csv', out_dir / 'dedup_summary.json')


def parse_backtest_summary(stdout: str) -> dict:
    """Extract key metrics from the backtester stdout (line-based)."""
    summary = {}
    for line in stdout.splitlines():
        line = line.strip()
        # Match lines like "Profit Factor:       1.66" or "Total Trades: 755"
        for key, label in [
            ('total_trades', 'Total Trades'),
            ('profit_factor', 'Profit Factor'),
            ('net_pnl', 'Net PnL'),
            ('net_pnl', 'Net P&L'),
            ('max_drawdown_pct', 'Max Drawdown'),
            ('sharpe_ratio', 'Sharpe Ratio'),
            ('win_rate', 'Win Rate'),
        ]:
            if label.lower() in line.lower():
                parts = line.split(':', 1)
                if len(parts) == 2:
                    raw = parts[1].strip().split()[0]
                    raw = raw.replace('$', '').replace(',', '').replace('%', '')
                    try:
                        summary[key] = float(raw)
                    except ValueError:
                        pass
    return summary


def build_matrix(events_csv: Path, matrix_csv: Path, summary_json: Path):
    """Build the per-archetype dedup matrix and summary from the event log."""
    import csv
    if not events_csv.exists() or events_csv.stat().st_size < 50:
        print(f'[diag] no events in {events_csv}; skipping matrix build')
        return

    # ============================================================
    # Per archetype:
    #   fires_total        : how many distinct (timestamp, archetype) signals
    #   wins_dedup         : how many of those events the archetype WON
    #   losses             : (wins_dedup - fires) — won as winner, lost as loser
    #   blocked_by[X]      : how many times this archetype lost to X
    # ============================================================
    fires_total = defaultdict(int)
    wins_dedup = defaultdict(int)
    losses_total = defaultdict(int)
    blocked_by = defaultdict(lambda: defaultdict(int))

    # Recover "fires" by combining winners and losers per (timestamp, direction).
    # Each row is one (winner, loser) edge. Same archetype-id appearing as
    # winner on a timestamp is counted once; as loser is counted once.

    seen_winners = set()  # (ts, dir, archetype)
    seen_losers = set()   # (ts, dir, archetype)

    with events_csv.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row['timestamp']
            wid = row['winner_id']
            lid = row['loser_id']
            wdir = row['winner_dir']
            ldir = row['loser_dir']
            key_w = (ts, wdir, wid)
            key_l = (ts, ldir, lid)
            if key_w not in seen_winners:
                seen_winners.add(key_w)
                wins_dedup[wid] += 1
                fires_total[wid] += 1
            if key_l not in seen_losers:
                seen_losers.add(key_l)
                losses_total[lid] += 1
                fires_total[lid] += 1
            blocked_by[lid][wid] += 1

    # All archetypes seen.
    all_archetypes = sorted(set(list(fires_total.keys())))

    # Top 5 blockers as columns.
    top_blockers = sorted(
        {b for d in blocked_by.values() for b in d.keys()},
        key=lambda x: -sum(blocked_by[a].get(x, 0) for a in blocked_by),
    )[:5]

    # Write matrix.
    with matrix_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        header = (
            ['archetype', 'fires_total', 'wins_dedup', 'win_rate', 'losses_total']
            + [f'blocked_by_{b}' for b in top_blockers]
            + ['blocked_by_other']
        )
        writer.writerow(header)
        for a in all_archetypes:
            ft = fires_total[a]
            wins = wins_dedup[a]
            losses = losses_total[a]
            wr = (wins / ft) if ft else 0.0
            row = [a, ft, wins, f'{wr:.4f}', losses]
            blocker_sum = 0
            for b in top_blockers:
                cnt = blocked_by[a].get(b, 0)
                row.append(cnt)
                blocker_sum += cnt
            other = losses - blocker_sum
            row.append(other)
            writer.writerow(row)

    summary = {
        'total_dedup_pairs': sum(losses_total.values()),
        'unique_archetypes_involved': len(all_archetypes),
        'top_blockers': top_blockers,
        'wins_by_archetype': dict(wins_dedup),
        'losses_by_archetype': dict(losses_total),
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f'[diag] dedup matrix written to {matrix_csv}')
    print(f'[diag] total dedup loss events: {summary["total_dedup_pairs"]}')


if __name__ == '__main__':
    main()
