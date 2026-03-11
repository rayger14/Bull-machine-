#!/usr/bin/env python3
"""
Merge Optuna Parallel Group Results into Production Configs

Loads results.json from each of the 6 optimization groups and merges
the best parameters into the production config and archetype YAMLs.

Dry-run by default. Pass --apply to write changes.

Groups:
  1: Top earners base_threshold (wick_trap, retest_cluster, liquidity_sweep, trap_within_trend)
  2: Mid-tier base_threshold (spring, failed_continuation, order_block_retest, liquidity_vacuum)
  3: New archetypes base_threshold (funding_divergence, long_squeeze, fvg_continuation, exhaustion_reversal)
  4: Global CMI params (temp_range, instab_range, crisis_coefficient)
  5: Structural gate thresholds (wick_pct, vol_z, RSI, BOS proximity, funding_z)
  6: ATR stop/TP multipliers (6 active archetypes x 2 params)

Usage:
    python3 bin/merge_optuna_results.py                # dry-run, show diff
    python3 bin/merge_optuna_results.py --apply        # write changes (backups created)
    python3 bin/merge_optuna_results.py --groups 1 4   # only merge groups 1 and 4

Author: Claude Code
Date: 2026-03-10
"""

import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent

PRODUCTION_CONFIG = PROJECT_ROOT / "configs" / "bull_machine_isolated_v11_fixed.json"
ARCHETYPE_CONFIG_DIR = PROJECT_ROOT / "configs" / "archetypes"

# Group result directories (must match optuna_parallel_group.py output)
GROUP_DIRS = {
    1: PROJECT_ROOT / "results" / "optuna_group_1_top_earners",
    2: PROJECT_ROOT / "results" / "optuna_group_2_mid_tier",
    3: PROJECT_ROOT / "results" / "optuna_group_3_new_archetypes",
    4: PROJECT_ROOT / "results" / "optuna_group_4_global_cmi",
    5: PROJECT_ROOT / "results" / "optuna_group_5_structural_gates",
    6: PROJECT_ROOT / "results" / "optuna_group_6_atr_multipliers",
}

GROUP_NAMES = {
    1: "Top Earners (base_threshold)",
    2: "Mid-Tier (base_threshold)",
    3: "New Archetypes (base_threshold)",
    4: "Global CMI Params",
    5: "Structural Gate Thresholds",
    6: "ATR Stop/TP Multipliers",
}

# Baseline values from optuna_parallel_group.py for comparison
BASELINE_THRESHOLDS = {
    'wick_trap': 0.10,
    'retest_cluster': 0.06,
    'liquidity_sweep': 0.08,
    'trap_within_trend': 0.08,
    'spring': 0.08,
    'failed_continuation': 0.08,
    'order_block_retest': 0.10,
    'liquidity_vacuum': 0.08,
    'funding_divergence': 0.18,
    'long_squeeze': 0.18,
    'fvg_continuation': 0.18,
    'exhaustion_reversal': 0.18,
}

BASELINE_GLOBALS = {
    'temp_range': 0.38,
    'instab_range': 0.15,
    'crisis_coefficient': 0.50,
}

BASELINE_GATES = {
    'wick_pct_K': 0.35,
    'wick_pct_G': 0.35,
    'vol_z_L': 1.0,
    'rsi_upper_L': 70.0,
    'rsi_lower_L': 30.0,
    'rsi_upper_F': 78.0,
    'rsi_lower_F': 22.0,
    'bos_atr_B': 1.5,
    'funding_z_S4': -1.0,
    'funding_z_S5': 1.0,
}

BASELINE_ATR = {
    'atr_stop_wick_trap': 3.4,
    'atr_tp_wick_trap': 4.0,
    'atr_stop_retest_cluster': 3.4,
    'atr_tp_retest_cluster': 5.4,
    'atr_stop_liquidity_sweep': 3.4,
    'atr_tp_liquidity_sweep': 5.7,
    'atr_stop_trap_within_trend': 2.9,
    'atr_tp_trap_within_trend': 3.8,
    'atr_stop_spring': 1.3,
    'atr_tp_spring': 3.1,
    'atr_stop_failed_continuation': 1.8,
    'atr_tp_failed_continuation': 6.0,
}

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def load_group_results(group_num: int) -> Optional[Dict[str, Any]]:
    """Load results.json for a group. Returns None if not found."""
    results_path = GROUP_DIRS[group_num] / "results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        return json.load(f)


def load_production_config() -> Dict[str, Any]:
    """Load the production JSON config."""
    with open(PRODUCTION_CONFIG) as f:
        return json.load(f)


def load_archetype_yaml(archetype_name: str) -> Optional[Dict[str, Any]]:
    """Load an archetype YAML config. Returns None if not found."""
    yaml_path = ARCHETYPE_CONFIG_DIR / f"{archetype_name}.yaml"
    if not yaml_path.exists():
        return None
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def save_production_config(config: Dict[str, Any]) -> None:
    """Write the production JSON config."""
    with open(PRODUCTION_CONFIG, 'w') as f:
        json.dump(config, f, indent=2)
        f.write('\n')


def save_archetype_yaml(archetype_name: str, data: Dict[str, Any]) -> None:
    """Write an archetype YAML config."""
    yaml_path = ARCHETYPE_CONFIG_DIR / f"{archetype_name}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def backup_file(path: Path) -> Path:
    """Create a .bak backup of a file. Returns backup path."""
    bak_path = path.with_suffix(path.suffix + '.bak')
    shutil.copy2(path, bak_path)
    return bak_path


def fmt_val(val, precision=4) -> str:
    """Format a numeric value for display."""
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)


def delta_arrow(old_val: float, new_val: float) -> str:
    """Return a directional indicator for the change."""
    diff = new_val - old_val
    if abs(diff) < 1e-6:
        return "  (unchanged)"
    pct = (diff / abs(old_val) * 100) if old_val != 0 else float('inf')
    direction = "+" if diff > 0 else ""
    return f"  ({direction}{diff:.4f}, {direction}{pct:.1f}%)"


def print_separator(char='=', width=90):
    print(char * width)


def print_header(text: str, char='=', width=90):
    print_separator(char, width)
    print(f"  {text}")
    print_separator(char, width)


# ─────────────────────────────────────────────────────────────────────
# Change tracking
# ─────────────────────────────────────────────────────────────────────

class ChangeTracker:
    """Tracks all proposed changes for summary and application."""

    def __init__(self):
        self.json_changes: List[Dict[str, Any]] = []  # Changes to production JSON
        self.yaml_changes: Dict[str, List[Dict[str, Any]]] = {}  # archetype -> changes
        self.group_summaries: Dict[int, Dict[str, Any]] = {}

    def add_json_change(self, group: int, section: str, key: str,
                        old_val: Any, new_val: Any):
        self.json_changes.append({
            'group': group,
            'section': section,
            'key': key,
            'old': old_val,
            'new': new_val,
        })

    def add_yaml_change(self, group: int, archetype: str, key: str,
                        old_val: Any, new_val: Any):
        if archetype not in self.yaml_changes:
            self.yaml_changes[archetype] = []
        self.yaml_changes[archetype].append({
            'group': group,
            'key': key,
            'old': old_val,
            'new': new_val,
        })

    def add_group_summary(self, group: int, results: Dict[str, Any]):
        self.group_summaries[group] = results

    @property
    def total_changes(self) -> int:
        yaml_count = sum(len(v) for v in self.yaml_changes.values())
        return len(self.json_changes) + yaml_count


# ─────────────────────────────────────────────────────────────────────
# Group processors
# ─────────────────────────────────────────────────────────────────────

def process_base_threshold_group(group_num: int, results: Dict[str, Any],
                                 config: Dict[str, Any],
                                 tracker: ChangeTracker) -> None:
    """Process Groups 1-3: per-archetype base_threshold updates."""
    best_params = results['best_params']
    af = config.get('adaptive_fusion', {})
    per_arch = af.get('per_archetype_base_threshold', {})

    for param_key, new_val in sorted(best_params.items()):
        # Parameter names are like "bt_wick_trap" -> archetype "wick_trap"
        if not param_key.startswith('bt_'):
            continue
        arch_name = param_key[3:]  # strip "bt_" prefix
        old_val = per_arch.get(arch_name, BASELINE_THRESHOLDS.get(arch_name, 0.18))
        tracker.add_json_change(
            group=group_num,
            section='adaptive_fusion.per_archetype_base_threshold',
            key=arch_name,
            old_val=old_val,
            new_val=new_val,
        )


def process_global_cmi_group(results: Dict[str, Any],
                             config: Dict[str, Any],
                             tracker: ChangeTracker) -> None:
    """Process Group 4: global CMI parameter updates."""
    best_params = results['best_params']
    af = config.get('adaptive_fusion', {})

    for param_key in ['temp_range', 'instab_range', 'crisis_coefficient']:
        if param_key in best_params:
            old_val = af.get(param_key, BASELINE_GLOBALS.get(param_key))
            new_val = best_params[param_key]
            tracker.add_json_change(
                group=4,
                section='adaptive_fusion',
                key=param_key,
                old_val=old_val,
                new_val=new_val,
            )


def process_structural_gates_group(results: Dict[str, Any],
                                   config: Dict[str, Any],
                                   tracker: ChangeTracker) -> None:
    """Process Group 5: structural gate threshold updates."""
    best_params = results['best_params']
    sc = config.get('structural_checks', {})
    gate_params = sc.get('gate_params', {})

    for param_key in sorted(BASELINE_GATES.keys()):
        if param_key in best_params:
            old_val = gate_params.get(param_key, BASELINE_GATES.get(param_key))
            new_val = best_params[param_key]
            tracker.add_json_change(
                group=5,
                section='structural_checks.gate_params',
                key=param_key,
                old_val=old_val,
                new_val=new_val,
            )


def process_atr_multipliers_group(results: Dict[str, Any],
                                  tracker: ChangeTracker) -> None:
    """Process Group 6: ATR stop/TP multiplier updates to archetype YAMLs."""
    best_params = results['best_params']

    for param_key, new_val in sorted(best_params.items()):
        # Parse "atr_stop_wick_trap" -> field_type="stop", arch_name="wick_trap"
        # Parse "atr_tp_wick_trap" -> field_type="tp", arch_name="wick_trap"
        parts = param_key.split('_', 2)  # ['atr', 'stop'/'tp', 'archetype_name']
        if len(parts) < 3:
            continue
        field_type = parts[1]  # 'stop' or 'tp'
        arch_name = param_key.replace(f'atr_{field_type}_', '')
        yaml_key = f'atr_{field_type}_mult'

        # Get old value from YAML or baseline
        arch_yaml = load_archetype_yaml(arch_name)
        if arch_yaml and 'position_sizing' in arch_yaml:
            old_val = arch_yaml['position_sizing'].get(yaml_key,
                      BASELINE_ATR.get(param_key, '?'))
        else:
            old_val = BASELINE_ATR.get(param_key, '?')

        tracker.add_yaml_change(
            group=6,
            archetype=arch_name,
            key=yaml_key,
            old_val=old_val,
            new_val=new_val,
        )


# ─────────────────────────────────────────────────────────────────────
# Apply changes
# ─────────────────────────────────────────────────────────────────────

def apply_json_changes(tracker: ChangeTracker) -> List[str]:
    """Apply all JSON changes to the production config. Returns list of actions."""
    actions = []
    bak = backup_file(PRODUCTION_CONFIG)
    actions.append(f"Backed up: {PRODUCTION_CONFIG.name} -> {bak.name}")

    config = load_production_config()
    af = config.setdefault('adaptive_fusion', {})
    per_arch = af.setdefault('per_archetype_base_threshold', {})

    for change in tracker.json_changes:
        section = change['section']
        key = change['key']
        new_val = change['new']

        if section == 'adaptive_fusion.per_archetype_base_threshold':
            per_arch[key] = new_val
            actions.append(f"  JSON: {section}.{key} = {new_val}")

        elif section == 'adaptive_fusion':
            af[key] = new_val
            actions.append(f"  JSON: {section}.{key} = {new_val}")

        elif section == 'structural_checks.gate_params':
            sc = config.setdefault('structural_checks', {})
            sc['enabled'] = True
            gp = sc.setdefault('gate_params', {})
            gp[key] = new_val
            actions.append(f"  JSON: {section}.{key} = {new_val}")

    save_production_config(config)
    actions.append(f"Saved: {PRODUCTION_CONFIG.name}")
    return actions


def apply_yaml_changes(tracker: ChangeTracker) -> List[str]:
    """Apply all YAML changes to archetype configs. Returns list of actions."""
    actions = []
    backed_up = set()

    for arch_name, changes in tracker.yaml_changes.items():
        yaml_path = ARCHETYPE_CONFIG_DIR / f"{arch_name}.yaml"

        if not yaml_path.exists():
            actions.append(f"  SKIP: {yaml_path.name} does not exist")
            continue

        # Backup once per file
        if arch_name not in backed_up:
            bak = backup_file(yaml_path)
            actions.append(f"Backed up: {yaml_path.name} -> {bak.name}")
            backed_up.add(arch_name)

        arch_data = load_archetype_yaml(arch_name)
        if arch_data is None:
            actions.append(f"  SKIP: Could not load {yaml_path.name}")
            continue

        ps = arch_data.setdefault('position_sizing', {})
        for change in changes:
            key = change['key']
            new_val = change['new']
            # Round to 1 decimal for ATR multipliers (matches optuna step=0.1)
            ps[key] = round(new_val, 1)
            actions.append(f"  YAML: {arch_name}.position_sizing.{key} = {round(new_val, 1)}")

        save_archetype_yaml(arch_name, arch_data)
        actions.append(f"Saved: {yaml_path.name}")

    return actions


# ─────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────

def print_group_metrics(group_num: int, results: Dict[str, Any]):
    """Print train/OOS metrics and WFE for a group."""
    train = results.get('train_metrics', {})
    oos = results.get('oos_metrics', {})
    wfe = results.get('wfe', 0)
    best_trial = results.get('best_trial', '?')
    best_score = results.get('best_score', 0)
    total_time = results.get('total_time_s', 0)
    n_trials = results.get('trials', 0)

    print(f"  Best Trial: #{best_trial} | Score: {best_score:.4f} | "
          f"Time: {total_time:.0f}s ({n_trials} trials, {total_time/max(n_trials,1):.0f}s/trial)")

    train_pf = train.get('profit_factor', 0)
    train_trades = train.get('total_trades', 0)
    train_wr = train.get('win_rate', 0)
    train_dd = abs(train.get('max_drawdown', 0))
    train_sharpe = train.get('sharpe_ratio', 0)
    train_pnl = train.get('total_pnl', 0)

    oos_pf = oos.get('profit_factor', 0)
    oos_trades = oos.get('total_trades', 0)
    oos_wr = oos.get('win_rate', 0)
    oos_dd = abs(oos.get('max_drawdown', 0))
    oos_sharpe = oos.get('sharpe_ratio', 0)
    oos_pnl = oos.get('total_pnl', 0)

    print(f"  Train: PF={train_pf:.3f} | Trades={train_trades:>4d} | WR={train_wr:.1f}% | "
          f"MaxDD={train_dd:.1f}% | Sharpe={train_sharpe:.2f} | PnL=${train_pnl:,.0f}")
    print(f"  OOS:   PF={oos_pf:.3f} | Trades={oos_trades:>4d} | WR={oos_wr:.1f}% | "
          f"MaxDD={oos_dd:.1f}% | Sharpe={oos_sharpe:.2f} | PnL=${oos_pnl:,.0f}")

    wfe_label = "PASS" if wfe >= 70 else ("WARN" if wfe >= 50 else "FAIL")
    print(f"  WFE:   {wfe:.0f}% {wfe_label}")


def print_change_table(tracker: ChangeTracker):
    """Print a formatted table of all proposed changes."""
    if not tracker.json_changes and not tracker.yaml_changes:
        print("\n  No changes to apply.")
        return

    # JSON changes (groups 1-5)
    if tracker.json_changes:
        print(f"\n  {'Parameter':<50s} {'Old':>10s} {'New':>10s} {'Delta':>20s}")
        print(f"  {'-'*50} {'-'*10} {'-'*10} {'-'*20}")

        current_section = None
        for change in sorted(tracker.json_changes, key=lambda c: (c['section'], c['key'])):
            section = change['section']
            if section != current_section:
                current_section = section
                print(f"\n  [{section}]")

            key = change['key']
            old_v = change['old']
            new_v = change['new']
            delta = delta_arrow(float(old_v), float(new_v)) if isinstance(old_v, (int, float)) else ""
            print(f"    {key:<48s} {fmt_val(old_v):>10s} {fmt_val(new_v):>10s} {delta}")

    # YAML changes (group 6)
    if tracker.yaml_changes:
        print(f"\n  [YAML: configs/archetypes/*.yaml -> position_sizing]")
        print(f"  {'Archetype.Key':<50s} {'Old':>10s} {'New':>10s} {'Delta':>20s}")
        print(f"  {'-'*50} {'-'*10} {'-'*10} {'-'*20}")

        for arch_name in sorted(tracker.yaml_changes.keys()):
            for change in tracker.yaml_changes[arch_name]:
                key = change['key']
                old_v = change['old']
                new_v = change['new']
                label = f"{arch_name}.{key}"
                delta = delta_arrow(float(old_v), float(new_v)) if isinstance(old_v, (int, float)) else ""
                print(f"    {label:<48s} {fmt_val(old_v, 1):>10s} {fmt_val(new_v, 1):>10s} {delta}")


def print_final_summary(tracker: ChangeTracker, applied: bool):
    """Print final summary of all changes across all groups."""
    print_header("FINAL SUMMARY")

    # WFE table
    print(f"\n  {'Group':<8s} {'Name':<35s} {'WFE':>6s} {'Status':>8s} {'Train PF':>10s} {'OOS PF':>10s}")
    print(f"  {'-'*8} {'-'*35} {'-'*6} {'-'*8} {'-'*10} {'-'*10}")

    for gnum in sorted(tracker.group_summaries.keys()):
        res = tracker.group_summaries[gnum]
        wfe = res.get('wfe', 0)
        wfe_label = "PASS" if wfe >= 70 else ("WARN" if wfe >= 50 else "FAIL")
        train_pf = res.get('train_metrics', {}).get('profit_factor', 0)
        oos_pf = res.get('oos_metrics', {}).get('profit_factor', 0)
        name = GROUP_NAMES.get(gnum, f"Group {gnum}")
        print(f"  {gnum:<8d} {name:<35s} {wfe:>5.0f}% {wfe_label:>8s} {train_pf:>10.3f} {oos_pf:>10.3f}")

    # Change counts
    json_count = len(tracker.json_changes)
    yaml_count = sum(len(v) for v in tracker.yaml_changes.values())
    total = json_count + yaml_count

    print(f"\n  Total changes: {total}")
    print(f"    JSON config changes: {json_count}")
    print(f"    YAML config changes: {yaml_count}")

    # Files modified
    files_modified = set()
    if json_count > 0:
        files_modified.add(str(PRODUCTION_CONFIG))
    for arch_name in tracker.yaml_changes:
        files_modified.add(str(ARCHETYPE_CONFIG_DIR / f"{arch_name}.yaml"))

    if files_modified:
        print(f"\n  Files {'modified' if applied else 'to be modified'}:")
        for f in sorted(files_modified):
            print(f"    {f}")

    if applied:
        print(f"\n  STATUS: All changes APPLIED. Backup files created with .bak suffix.")
        print(f"  NEXT: Run a validation backtest to confirm results:")
        print(f"    python3 bin/backtest_v11_standalone.py --start-date 2023-01-01 --end-date 2024-12-31")
    else:
        print(f"\n  STATUS: DRY RUN. No files modified.")
        print(f"  To apply: python3 bin/merge_optuna_results.py --apply")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    global PRODUCTION_CONFIG

    parser = argparse.ArgumentParser(
        description="Merge Optuna parallel group results into production configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 bin/merge_optuna_results.py                # Dry-run: show all changes
  python3 bin/merge_optuna_results.py --apply        # Apply all changes (with backups)
  python3 bin/merge_optuna_results.py --groups 1 4   # Only merge groups 1 and 4
  python3 bin/merge_optuna_results.py --min-wfe 60   # Only merge groups with WFE >= 60%
""",
    )
    parser.add_argument('--apply', action='store_true',
                        help='Apply changes to config files (default: dry-run)')
    parser.add_argument('--groups', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                        choices=[1, 2, 3, 4, 5, 6],
                        help='Which groups to merge (default: all)')
    parser.add_argument('--min-wfe', type=float, default=0.0,
                        help='Minimum WFE %% to accept a group result (default: 0, accept all)')
    parser.add_argument('--config', type=str, default=None,
                        help=f'Production config path (default: {PRODUCTION_CONFIG})')
    args = parser.parse_args()

    # Allow overriding config path
    if args.config is not None:
        PRODUCTION_CONFIG = Path(args.config)

    print_header("OPTUNA PARALLEL GROUP RESULTS MERGER")
    print(f"  Mode:   {'APPLY (will write files)' if args.apply else 'DRY RUN (no files modified)'}")
    print(f"  Config: {PRODUCTION_CONFIG}")
    print(f"  Groups: {args.groups}")
    if args.min_wfe > 0:
        print(f"  Min WFE: {args.min_wfe:.0f}%")
    print()

    # Load production config for current values
    if not PRODUCTION_CONFIG.exists():
        print(f"ERROR: Production config not found: {PRODUCTION_CONFIG}")
        sys.exit(1)

    config = load_production_config()
    tracker = ChangeTracker()

    # ── Process each group ──────────────────────────────────────────
    groups_loaded = 0
    groups_skipped_missing = []
    groups_skipped_wfe = []

    for group_num in sorted(args.groups):
        print_header(f"GROUP {group_num}: {GROUP_NAMES[group_num]}", char='-')

        results = load_group_results(group_num)
        if results is None:
            print(f"  SKIPPED: results.json not found at {GROUP_DIRS[group_num]}/")
            groups_skipped_missing.append(group_num)
            print()
            continue

        groups_loaded += 1
        tracker.add_group_summary(group_num, results)

        # Print metrics
        print_group_metrics(group_num, results)

        # Check WFE threshold
        wfe = results.get('wfe', 0)
        if wfe < args.min_wfe:
            print(f"\n  SKIPPED: WFE {wfe:.0f}% < minimum {args.min_wfe:.0f}%")
            groups_skipped_wfe.append(group_num)
            print()
            continue

        # Process based on group type
        if group_num in (1, 2, 3):
            process_base_threshold_group(group_num, results, config, tracker)
        elif group_num == 4:
            process_global_cmi_group(results, config, tracker)
        elif group_num == 5:
            process_structural_gates_group(results, config, tracker)
        elif group_num == 6:
            process_atr_multipliers_group(results, tracker)

        print()

    # ── Display all changes ─────────────────────────────────────────
    if groups_loaded == 0:
        print("\nNo group results found. Nothing to merge.")
        print("Expected result locations:")
        for gnum in sorted(args.groups):
            print(f"  Group {gnum}: {GROUP_DIRS[gnum]}/results.json")
        sys.exit(0)

    print_header("PROPOSED CHANGES")
    print_change_table(tracker)

    # ── Status of skipped groups ────────────────────────────────────
    if groups_skipped_missing:
        print(f"\n  Groups skipped (no results): {groups_skipped_missing}")
    if groups_skipped_wfe:
        print(f"\n  Groups skipped (below WFE threshold): {groups_skipped_wfe}")

    # ── Apply if requested ──────────────────────────────────────────
    if args.apply and tracker.total_changes > 0:
        print_header("APPLYING CHANGES")

        actions = []
        if tracker.json_changes:
            actions.extend(apply_json_changes(tracker))
        if tracker.yaml_changes:
            actions.extend(apply_yaml_changes(tracker))

        for action in actions:
            print(f"  {action}")

        print_final_summary(tracker, applied=True)

    elif tracker.total_changes > 0:
        print_final_summary(tracker, applied=False)
    else:
        print("\n  No changes to apply (all values unchanged or no valid groups).")


if __name__ == '__main__':
    main()
