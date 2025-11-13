#!/usr/bin/env python3
"""
Monitor Optuna optimization progress in real-time.

Usage:
    python3 bin/monitor_optuna.py results/optuna_step5_full/optuna_study.db
"""

import argparse
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

def get_study_stats(db_path):
    """Extract study statistics from Optuna SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get study info
    cursor.execute("SELECT study_name FROM studies")
    study_name = cursor.fetchone()[0]

    # Get trial stats
    cursor.execute("""
        SELECT
            COUNT(*) as total_trials,
            COUNT(CASE WHEN state = 'COMPLETE' THEN 1 END) as complete,
            COUNT(CASE WHEN state = 'RUNNING' THEN 1 END) as running,
            COUNT(CASE WHEN state = 'PRUNED' THEN 1 END) as pruned,
            COUNT(CASE WHEN state = 'FAIL' THEN 1 END) as failed
        FROM trials
    """)
    total, complete, running, pruned, failed = cursor.fetchone()

    # Get best trial
    cursor.execute("""
        SELECT trial_id, value, datetime_complete
        FROM trials
        WHERE state = 'COMPLETE' AND value IS NOT NULL
        ORDER BY value DESC
        LIMIT 1
    """)
    best_result = cursor.fetchone()

    # Get recent trials (last 5)
    cursor.execute("""
        SELECT trial_id, value, state, datetime_complete
        FROM trials
        ORDER BY trial_id DESC
        LIMIT 5
    """)
    recent_trials = cursor.fetchall()

    # Get parameter distributions for successful trials (value > 0)
    cursor.execute("""
        SELECT tp.param_name, AVG(tp.param_value) as avg_val,
               MIN(tp.param_value) as min_val, MAX(tp.param_value) as max_val
        FROM trial_params tp
        JOIN trials t ON tp.trial_id = t.trial_id
        WHERE t.state = 'COMPLETE' AND t.value > 0
        GROUP BY tp.param_name
    """)
    param_stats = cursor.fetchall()

    conn.close()

    return {
        'study_name': study_name,
        'total': total,
        'complete': complete,
        'running': running,
        'pruned': pruned,
        'failed': failed,
        'best_trial': best_result,
        'recent_trials': recent_trials,
        'param_stats': param_stats
    }

def print_stats(stats):
    """Print formatted statistics."""
    print("=" * 80)
    print(f"OPTUNA STUDY: {stats['study_name']}")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print(f"Progress: {stats['complete']}/{stats['total']} trials complete")
    print(f"  Complete: {stats['complete']}")
    print(f"  Running:  {stats['running']}")
    print(f"  Pruned:   {stats['pruned']}")
    print(f"  Failed:   {stats['failed']}")
    print()

    if stats['best_trial']:
        trial_id, value, dt_complete = stats['best_trial']
        print(f"Best Trial: #{trial_id}")
        print(f"  Objective: {value:.3f}")
        print(f"  Completed: {dt_complete}")

        if value > 0:
            print(f"  Status: ✅ PASSED CONSTRAINTS (PF ≥ 1.2)")
        else:
            implied_pf = abs(value)
            if implied_pf > 1.0:
                print(f"  Status: ⚠️  CONSTRAINT FAILED (estimated PF ~{implied_pf:.2f})")
            else:
                print(f"  Status: ❌ CONSTRAINT FAILED (PF < 1.2)")
    else:
        print("Best Trial: None yet")
    print()

    if stats['recent_trials']:
        print("Recent Trials (last 5):")
        for trial_id, value, state, dt_complete in stats['recent_trials']:
            status_icon = "✅" if value and value > 0 else "❌" if state == 'COMPLETE' else "🏃" if state == 'RUNNING' else "⏭️"
            val_str = f"{value:.3f}" if value else "N/A"
            print(f"  {status_icon} Trial #{trial_id}: {val_str} ({state})")
    print()

    if stats['param_stats']:
        print("Parameter Ranges for Successful Trials (objective > 0):")
        print("-" * 80)
        for param_name, avg_val, min_val, max_val in stats['param_stats']:
            print(f"  {param_name:30s}: avg={avg_val:6.3f}, range=[{min_val:.3f}, {max_val:.3f}]")
        print()
    else:
        print("No successful trials yet (all failed constraints)")
        print()

def monitor_loop(db_path, interval=60):
    """Monitor study in a loop."""
    print("Monitoring Optuna study (Ctrl+C to exit)...")
    print()

    try:
        while True:
            if Path(db_path).exists():
                stats = get_study_stats(db_path)
                print_stats(stats)

                if stats['complete'] > 0:
                    success_rate = (stats['complete'] - stats['failed']) / stats['complete'] * 100
                    print(f"Success Rate: {success_rate:.1f}% (trials passing constraints)")
                    print()
            else:
                print(f"Waiting for database: {db_path}")

            print(f"Next update in {interval}s...")
            print("=" * 80)
            print()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Monitor Optuna optimization progress')
    parser.add_argument('db_path', help='Path to Optuna SQLite database')
    parser.add_argument('--interval', type=int, default=60,
                       help='Update interval in seconds (default: 60)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (no loop)')
    args = parser.parse_args()

    if args.once:
        if Path(args.db_path).exists():
            stats = get_study_stats(args.db_path)
            print_stats(stats)
        else:
            print(f"Database not found: {args.db_path}")
            sys.exit(1)
    else:
        monitor_loop(args.db_path, args.interval)

if __name__ == '__main__':
    main()
