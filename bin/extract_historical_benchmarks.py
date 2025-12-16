#!/usr/bin/env python3
"""
Extract Historical Benchmarks from Optuna Databases

Purpose: Find the EXACT conditions that produced historical PF 2.22, PF 1.86 claims
"""

import sqlite3
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

def extract_optuna_study_details(db_path: str) -> Dict:
    """Extract all trials and best parameters from Optuna database."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all studies
    studies = cursor.execute("SELECT study_id, study_name FROM studies").fetchall()

    results = {}

    for study_id, study_name in studies:
        print(f"\n{'='*80}")
        print(f"Study: {study_name} (ID: {study_id})")
        print(f"{'='*80}")

        # Get ALL completed trials (not just best)
        # Optuna stores values in separate table
        trials = cursor.execute("""
            SELECT t.trial_id, t.number, tv.value, t.datetime_start, t.datetime_complete, t.state
            FROM trials t
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id = ? AND t.state = 'COMPLETE'
            ORDER BY tv.value DESC
        """, (study_id,)).fetchall()

        print(f"Total completed trials: {len(trials)}")

        trial_details = []

        for trial_id, trial_num, value, start_time, end_time, state in trials[:10]:  # Top 10 trials
            # Get parameters
            params = cursor.execute("""
                SELECT param_name, param_value
                FROM trial_params
                WHERE trial_id = ?
            """, (trial_id,)).fetchall()

            param_dict = dict(params)

            # Get user attributes (may contain metadata)
            user_attrs = cursor.execute("""
                SELECT key, value_json
                FROM trial_user_attributes
                WHERE trial_id = ?
            """, (trial_id,)).fetchall()

            user_attr_dict = {k: v for k, v in user_attrs}

            # Get system attributes (may contain test period info)
            sys_attrs = cursor.execute("""
                SELECT key, value_json
                FROM trial_system_attributes
                WHERE trial_id = ?
            """, (trial_id,)).fetchall()

            sys_attr_dict = {k: v for k, v in sys_attrs}

            trial_info = {
                "trial_number": trial_num,
                "profit_factor": value,
                "start_time": start_time,
                "end_time": end_time,
                "parameters": param_dict,
                "user_attributes": user_attr_dict,
                "system_attributes": sys_attr_dict
            }

            trial_details.append(trial_info)

            # Print summary
            print(f"\n  Trial {trial_num}: PF = {value:.4f}")
            print(f"    Time: {start_time} to {end_time}")
            print(f"    Parameters: {json.dumps(param_dict, indent=6)}")
            if user_attr_dict:
                print(f"    User Attrs: {json.dumps(user_attr_dict, indent=6)}")
            if sys_attr_dict:
                print(f"    System Attrs: {json.dumps(sys_attr_dict, indent=6)}")

        results[study_name] = {
            "total_trials": len(trials),
            "best_pf": trials[0][2] if trials else None,
            "top_10_trials": trial_details
        }

    conn.close()
    return results


def find_pf_2_22_trials(db_path: str) -> List[Dict]:
    """Find any trials with PF near 2.22 (±0.05)."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Search for trials with PF between 2.17 and 2.27
    trials = cursor.execute("""
        SELECT t.trial_id, t.number, tv.value, s.study_name, t.datetime_start
        FROM trials t
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        JOIN studies s ON t.study_id = s.study_id
        WHERE t.state = 'COMPLETE' AND tv.value >= 2.17 AND tv.value <= 2.27
        ORDER BY tv.value DESC
    """).fetchall()

    matching_trials = []

    for trial_id, trial_num, value, study_name, start_time in trials:
        # Get parameters
        params = cursor.execute("""
            SELECT param_name, param_value
            FROM trial_params
            WHERE trial_id = ?
        """, (trial_id,)).fetchall()

        matching_trials.append({
            "study_name": study_name,
            "trial_number": trial_num,
            "profit_factor": value,
            "start_time": start_time,
            "parameters": dict(params)
        })

    conn.close()
    return matching_trials


def find_pf_1_86_trials(db_path: str) -> List[Dict]:
    """Find any trials with PF near 1.86 (±0.05)."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Search for trials with PF between 1.81 and 1.91
    trials = cursor.execute("""
        SELECT t.trial_id, t.number, tv.value, s.study_name, t.datetime_start
        FROM trials t
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        JOIN studies s ON t.study_id = s.study_id
        WHERE t.state = 'COMPLETE' AND tv.value >= 1.81 AND tv.value <= 1.91
        ORDER BY tv.value DESC
    """).fetchall()

    matching_trials = []

    for trial_id, trial_num, value, study_name, start_time in trials:
        # Get parameters
        params = cursor.execute("""
            SELECT param_name, param_value
            FROM trial_params
            WHERE trial_id = ?
        """, (trial_id,)).fetchall()

        matching_trials.append({
            "study_name": study_name,
            "trial_number": trial_num,
            "profit_factor": value,
            "start_time": start_time,
            "parameters": dict(params)
        })

    conn.close()
    return matching_trials


def main():
    print("="*100)
    print("HISTORICAL BENCHMARK EXTRACTION")
    print("="*100)

    # Define databases to search
    databases = {
        "S4 Calibration": "./results/s4_calibration/optuna_s4_calibration.db",
        "S2 Calibration": "./results/s2_calibration/optuna_s2_calibration.db",
        "Liquidity Vacuum": "./results/liquidity_vacuum_calibration/optuna_liquidity_vacuum.db"
    }

    all_results = {}

    # Extract from each database
    for name, db_path in databases.items():
        if not Path(db_path).exists():
            print(f"\n⚠️  Database not found: {db_path}")
            continue

        print(f"\n\n{'#'*100}")
        print(f"# DATABASE: {name}")
        print(f"# Path: {db_path}")
        print(f"{'#'*100}")

        # Extract all study details
        study_results = extract_optuna_study_details(db_path)
        all_results[name] = study_results

        # Search for specific PF values
        print(f"\n\n--- Searching for PF ≈ 2.22 ---")
        pf_2_22_matches = find_pf_2_22_trials(db_path)
        if pf_2_22_matches:
            print(f"Found {len(pf_2_22_matches)} trials with PF near 2.22:")
            for match in pf_2_22_matches:
                print(f"  - {match['study_name']}: Trial {match['trial_number']}, PF {match['profit_factor']:.4f}")
                print(f"    Parameters: {json.dumps(match['parameters'], indent=6)}")
        else:
            print("No trials found with PF near 2.22")

        print(f"\n--- Searching for PF ≈ 1.86 ---")
        pf_1_86_matches = find_pf_1_86_trials(db_path)
        if pf_1_86_matches:
            print(f"Found {len(pf_1_86_matches)} trials with PF near 1.86:")
            for match in pf_1_86_matches:
                print(f"  - {match['study_name']}: Trial {match['trial_number']}, PF {match['profit_factor']:.4f}")
                print(f"    Parameters: {json.dumps(match['parameters'], indent=6)}")
        else:
            print("No trials found with PF near 1.86")

    # Save consolidated results
    output_path = Path("results/historical_benchmarks_extraction.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*100}")
    print(f"✅ Results saved to: {output_path}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
