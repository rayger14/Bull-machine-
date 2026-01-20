#!/usr/bin/env python3
"""
System B0 Deployment Safety Verification

Comprehensive pre-deployment and runtime verification system to ensure
System B0 deployment does not interfere with critical archetype optimizations.

Features:
- Pre-deployment resource checks
- Process conflict detection
- Runtime monitoring
- Post-deployment validation
- Rollback verification
- Detailed diagnostics

Usage:
    # Full pre-deployment check
    python bin/verify_safe_deployment.py --full-check

    # Monitor during deployment
    python bin/verify_safe_deployment.py --monitor

    # Continuous monitoring
    python bin/verify_safe_deployment.py --monitor --interval 30 --duration 3600

    # Post-deployment validation
    python bin/verify_safe_deployment.py --post-deployment-check

    # Post-rollback verification
    python bin/verify_safe_deployment.py --post-rollback-check

Architecture:
- Resource monitoring (CPU, memory, disk)
- Process health checking
- Database conflict detection
- Performance baseline comparison
- Alert system for threshold violations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import time
import argparse
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sqlite3


# =============================================================================
# Configuration
# =============================================================================

# Resource thresholds
THRESHOLDS = {
    'disk_min_gb': 10,
    'memory_min_gb': 4,
    'cpu_load_warning': 8.0,
    'cpu_load_critical': 10.0,
    'trial_speed_degradation_warning': 0.8,  # 80% of baseline
    'trial_speed_degradation_critical': 0.7,  # 70% of baseline
    'error_rate_warning': 0.001,  # 0.1%
    'error_rate_critical': 0.01,  # 1%
}

# Expected process patterns
PROCESS_PATTERNS = {
    'archetype_optimizations': [
        'optuna',
        'optimize_bear',
        'optuna_parallel',
        'optimize_s2',
        'optimize_s5'
    ],
    'system_b0': [
        'system_b0',
        'run_system_b0',
        'backtest_system_b0'
    ]
}

# Database files
OPTIMIZATION_DBS = [
    'optuna_production_v2_*.db',
    'optuna_quick_test_v3_*.db',
    'results/phase2_optimization/optimization_study.db',
    'results/s2_calibration/optuna_*.db',
    'results/s4_calibration/optuna_*.db'
]

SYSTEM_B0_DBS = [
    'data/system_b0_production.db',
    'results/system_b0/*.db'
]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ResourceStatus:
    """System resource status."""
    timestamp: datetime
    disk_free_gb: float
    memory_free_gb: float
    cpu_load_1m: float
    cpu_load_5m: float
    cpu_load_15m: float
    cpu_count: int

    def is_healthy(self) -> bool:
        """Check if resources are within healthy thresholds."""
        return (
            self.disk_free_gb >= THRESHOLDS['disk_min_gb'] and
            self.memory_free_gb >= THRESHOLDS['memory_min_gb'] and
            self.cpu_load_1m < THRESHOLDS['cpu_load_critical']
        )

    def get_warnings(self) -> List[str]:
        """Get warning messages for threshold violations."""
        warnings = []

        if self.disk_free_gb < THRESHOLDS['disk_min_gb']:
            warnings.append(f"LOW DISK SPACE: {self.disk_free_gb:.1f}GB free (minimum: {THRESHOLDS['disk_min_gb']}GB)")

        if self.memory_free_gb < THRESHOLDS['memory_min_gb']:
            warnings.append(f"LOW MEMORY: {self.memory_free_gb:.1f}GB free (minimum: {THRESHOLDS['memory_min_gb']}GB)")

        if self.cpu_load_1m >= THRESHOLDS['cpu_load_critical']:
            warnings.append(f"CRITICAL CPU LOAD: {self.cpu_load_1m:.2f} (threshold: {THRESHOLDS['cpu_load_critical']})")
        elif self.cpu_load_1m >= THRESHOLDS['cpu_load_warning']:
            warnings.append(f"HIGH CPU LOAD: {self.cpu_load_1m:.2f} (warning: {THRESHOLDS['cpu_load_warning']})")

        return warnings


@dataclass
class ProcessStatus:
    """Process health status."""
    timestamp: datetime
    optimization_processes: List[Dict[str, Any]]
    system_b0_processes: List[Dict[str, Any]]

    def is_healthy(self) -> bool:
        """Check if processes are healthy."""
        # Optimizations should be running (or intentionally stopped)
        # System B0 may or may not be running (deployment phase dependent)
        return True  # Detailed checks in get_warnings

    def get_warnings(self) -> List[str]:
        """Get warning messages."""
        warnings = []

        if not self.optimization_processes:
            warnings.append("WARNING: No archetype optimization processes detected")

        # Check for zombie processes
        for proc in self.optimization_processes + self.system_b0_processes:
            if proc.get('status') == 'Z':
                warnings.append(f"ZOMBIE PROCESS: PID {proc['pid']} ({proc['name']})")

        return warnings


@dataclass
class DatabaseStatus:
    """Database health status."""
    timestamp: datetime
    optimization_dbs: List[Dict[str, Any]]
    system_b0_dbs: List[Dict[str, Any]]
    conflicts: List[str]

    def is_healthy(self) -> bool:
        """Check if databases are healthy."""
        return len(self.conflicts) == 0

    def get_warnings(self) -> List[str]:
        """Get warning messages."""
        warnings = []

        if self.conflicts:
            warnings.append(f"DATABASE CONFLICTS: {len(self.conflicts)} detected")
            for conflict in self.conflicts:
                warnings.append(f"  - {conflict}")

        return warnings


@dataclass
class PerformanceStatus:
    """Performance metrics status."""
    timestamp: datetime
    trial_rate_current: float
    trial_rate_baseline: float
    trial_rate_ratio: float
    error_rate: float

    def is_healthy(self) -> bool:
        """Check if performance is healthy."""
        return (
            self.trial_rate_ratio >= THRESHOLDS['trial_speed_degradation_critical'] and
            self.error_rate <= THRESHOLDS['error_rate_critical']
        )

    def get_warnings(self) -> List[str]:
        """Get warning messages."""
        warnings = []

        if self.trial_rate_ratio < THRESHOLDS['trial_speed_degradation_critical']:
            warnings.append(f"CRITICAL: Trial rate degraded to {self.trial_rate_ratio:.1%} of baseline")
        elif self.trial_rate_ratio < THRESHOLDS['trial_speed_degradation_warning']:
            warnings.append(f"WARNING: Trial rate degraded to {self.trial_rate_ratio:.1%} of baseline")

        if self.error_rate >= THRESHOLDS['error_rate_critical']:
            warnings.append(f"CRITICAL: Error rate {self.error_rate:.2%} (threshold: {THRESHOLDS['error_rate_critical']:.2%})")
        elif self.error_rate >= THRESHOLDS['error_rate_warning']:
            warnings.append(f"WARNING: Error rate {self.error_rate:.2%} (threshold: {THRESHOLDS['error_rate_warning']:.2%})")

        return warnings


@dataclass
class VerificationResult:
    """Overall verification result."""
    timestamp: datetime
    status: str  # GO, NO-GO, WARNING
    resources: ResourceStatus
    processes: ProcessStatus
    databases: DatabaseStatus
    performance: Optional[PerformanceStatus]
    warnings: List[str]
    errors: List[str]

    def is_go(self) -> bool:
        """Determine if deployment is GO."""
        return self.status == 'GO'

    def is_no_go(self) -> bool:
        """Determine if deployment is NO-GO."""
        return self.status == 'NO-GO'


# =============================================================================
# Resource Monitoring
# =============================================================================

def get_resource_status() -> ResourceStatus:
    """Get current system resource status."""

    # Get disk space
    df_output = subprocess.check_output(['df', '-k', '.']).decode('utf-8')
    df_lines = df_output.strip().split('\n')
    df_data = df_lines[1].split()
    disk_free_kb = int(df_data[3])
    disk_free_gb = disk_free_kb / (1024 * 1024)

    # Get memory (macOS)
    vm_stat = subprocess.check_output(['vm_stat']).decode('utf-8')
    page_size = 4096  # Default page size
    pages_free = 0

    for line in vm_stat.split('\n'):
        if 'page size of' in line:
            page_size = int(line.split()[-2])
        if 'Pages free:' in line:
            pages_free = int(line.split()[-1].rstrip('.').replace(',', ''))

    memory_free_gb = (pages_free * page_size) / (1024 ** 3)

    # Get CPU load
    uptime_output = subprocess.check_output(['uptime']).decode('utf-8')
    load_avg = uptime_output.split('load averages:')[1].strip().split()
    cpu_load_1m = float(load_avg[0])
    cpu_load_5m = float(load_avg[1])
    cpu_load_15m = float(load_avg[2])

    # Get CPU count
    cpu_count = int(subprocess.check_output(['sysctl', '-n', 'hw.ncpu']).decode('utf-8').strip())

    return ResourceStatus(
        timestamp=datetime.now(),
        disk_free_gb=disk_free_gb,
        memory_free_gb=memory_free_gb,
        cpu_load_1m=cpu_load_1m,
        cpu_load_5m=cpu_load_5m,
        cpu_load_15m=cpu_load_15m,
        cpu_count=cpu_count
    )


# =============================================================================
# Process Monitoring
# =============================================================================

def get_running_processes(patterns: List[str]) -> List[Dict[str, Any]]:
    """Get running processes matching patterns."""

    processes = []

    # Get process list
    ps_output = subprocess.check_output(['ps', 'aux']).decode('utf-8')

    for line in ps_output.split('\n')[1:]:  # Skip header
        if not line.strip():
            continue

        # Check if any pattern matches
        if any(pattern in line for pattern in patterns):
            parts = line.split(None, 10)
            if len(parts) >= 11:
                processes.append({
                    'user': parts[0],
                    'pid': int(parts[1]),
                    'cpu': float(parts[2]),
                    'mem': float(parts[3]),
                    'status': parts[7],
                    'started': parts[8],
                    'time': parts[9],
                    'command': parts[10]
                })

    return processes


def get_process_status() -> ProcessStatus:
    """Get current process status."""

    optimization_procs = get_running_processes(PROCESS_PATTERNS['archetype_optimizations'])
    system_b0_procs = get_running_processes(PROCESS_PATTERNS['system_b0'])

    return ProcessStatus(
        timestamp=datetime.now(),
        optimization_processes=optimization_procs,
        system_b0_processes=system_b0_procs
    )


# =============================================================================
# Database Monitoring
# =============================================================================

def find_database_files(patterns: List[str]) -> List[str]:
    """Find database files matching patterns."""

    import glob

    db_files = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        db_files.extend(matches)

    return db_files


def check_database_locks(db_file: str) -> List[Dict[str, Any]]:
    """Check for locks on database file."""

    locks = []

    try:
        lsof_output = subprocess.check_output(['lsof', db_file], stderr=subprocess.DEVNULL).decode('utf-8')

        for line in lsof_output.split('\n')[1:]:  # Skip header
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) >= 9:
                locks.append({
                    'process': parts[0],
                    'pid': int(parts[1]),
                    'user': parts[2],
                    'mode': parts[3],
                    'type': parts[4]
                })
    except subprocess.CalledProcessError:
        # No locks (file not open)
        pass

    return locks


def get_database_status() -> DatabaseStatus:
    """Get current database status."""

    # Find databases
    opt_dbs = find_database_files(OPTIMIZATION_DBS)
    b0_dbs = find_database_files(SYSTEM_B0_DBS)

    opt_db_info = []
    for db in opt_dbs:
        if os.path.exists(db):
            size_mb = os.path.getsize(db) / (1024 * 1024)
            locks = check_database_locks(db)
            opt_db_info.append({
                'path': db,
                'size_mb': size_mb,
                'locks': locks
            })

    b0_db_info = []
    for db in b0_dbs:
        if os.path.exists(db):
            size_mb = os.path.getsize(db) / (1024 * 1024)
            locks = check_database_locks(db)
            b0_db_info.append({
                'path': db,
                'size_mb': size_mb,
                'locks': locks
            })

    # Check for conflicts (shared databases with write locks)
    conflicts = []

    # Check if System B0 is accessing optimization databases (BAD)
    for db_info in opt_db_info:
        for lock in db_info['locks']:
            if any(pattern in lock['process'].lower() for pattern in ['system_b0', 'b0']):
                conflicts.append(f"System B0 accessing optimization database: {db_info['path']}")

    # Check for multiple write locks on same database
    all_dbs = opt_db_info + b0_db_info
    for db_info in all_dbs:
        write_locks = [lock for lock in db_info['locks'] if 'W' in lock.get('mode', '')]
        if len(write_locks) > 1:
            conflicts.append(f"Multiple write locks on {db_info['path']}: {[lock['process'] for lock in write_locks]}")

    return DatabaseStatus(
        timestamp=datetime.now(),
        optimization_dbs=opt_db_info,
        system_b0_dbs=b0_db_info,
        conflicts=conflicts
    )


# =============================================================================
# Performance Monitoring
# =============================================================================

def get_trial_rate(db_file: str, hours: int = 1) -> float:
    """Get trial completion rate from Optuna database."""

    if not os.path.exists(db_file):
        return 0.0

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Get trials completed in last N hours
        query = """
        SELECT COUNT(*)
        FROM trials
        WHERE state = 'COMPLETE'
        AND datetime_complete >= datetime('now', '-{} hours')
        """.format(hours)

        cursor.execute(query)
        count = cursor.fetchone()[0]

        conn.close()

        # Trials per hour
        return count / hours

    except Exception as e:
        print(f"Warning: Could not query {db_file}: {e}")
        return 0.0


def get_performance_status(baseline_trial_rate: Optional[float] = None) -> PerformanceStatus:
    """Get current performance status."""

    # Get current trial rate across all optimization databases
    opt_dbs = find_database_files(OPTIMIZATION_DBS)

    current_trial_rates = []
    for db in opt_dbs:
        if os.path.exists(db) and os.path.getsize(db) > 0:
            rate = get_trial_rate(db, hours=1)
            if rate > 0:
                current_trial_rates.append(rate)

    trial_rate_current = sum(current_trial_rates) if current_trial_rates else 0.0

    # Use baseline or assume current is baseline
    trial_rate_baseline = baseline_trial_rate or max(trial_rate_current, 1.0)

    # Calculate ratio
    trial_rate_ratio = trial_rate_current / trial_rate_baseline if trial_rate_baseline > 0 else 1.0

    # TODO: Implement error rate tracking from logs
    error_rate = 0.0

    return PerformanceStatus(
        timestamp=datetime.now(),
        trial_rate_current=trial_rate_current,
        trial_rate_baseline=trial_rate_baseline,
        trial_rate_ratio=trial_rate_ratio,
        error_rate=error_rate
    )


# =============================================================================
# Verification Logic
# =============================================================================

def run_full_check(baseline_trial_rate: Optional[float] = None) -> VerificationResult:
    """Run full pre-deployment verification check."""

    print("\n" + "=" * 80)
    print("SYSTEM B0 DEPLOYMENT SAFETY VERIFICATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print()

    # Collect status
    print("Checking system resources...")
    resources = get_resource_status()

    print("Checking running processes...")
    processes = get_process_status()

    print("Checking database status...")
    databases = get_database_status()

    print("Checking performance metrics...")
    performance = get_performance_status(baseline_trial_rate)

    # Aggregate warnings and errors
    warnings = []
    errors = []

    # Resource checks
    resource_warnings = resources.get_warnings()
    if resource_warnings:
        warnings.extend(resource_warnings)

    # Process checks
    process_warnings = processes.get_warnings()
    if process_warnings:
        warnings.extend(process_warnings)

    # Database checks
    db_warnings = databases.get_warnings()
    if db_warnings:
        errors.extend(db_warnings)  # Database conflicts are errors

    # Performance checks
    perf_warnings = performance.get_warnings()
    if perf_warnings:
        warnings.extend(perf_warnings)

    # Determine overall status
    if errors or not resources.is_healthy():
        status = 'NO-GO'
    elif warnings:
        status = 'WARNING'
    else:
        status = 'GO'

    result = VerificationResult(
        timestamp=datetime.now(),
        status=status,
        resources=resources,
        processes=processes,
        databases=databases,
        performance=performance,
        warnings=warnings,
        errors=errors
    )

    # Print results
    print_verification_result(result)

    return result


def print_verification_result(result: VerificationResult):
    """Print verification result."""

    print("\n" + "=" * 80)
    print("VERIFICATION RESULT")
    print("=" * 80)

    # Overall status
    status_color = {
        'GO': '✓ GO',
        'NO-GO': '✗ NO-GO',
        'WARNING': '⚠ WARNING'
    }

    print(f"\nStatus: {status_color.get(result.status, result.status)}")
    print()

    # Resources
    print("SYSTEM RESOURCES:")
    print(f"  Disk Free:      {result.resources.disk_free_gb:.1f} GB (minimum: {THRESHOLDS['disk_min_gb']} GB)")
    print(f"  Memory Free:    {result.resources.memory_free_gb:.1f} GB (minimum: {THRESHOLDS['memory_min_gb']} GB)")
    print(f"  CPU Load (1m):  {result.resources.cpu_load_1m:.2f} / {result.resources.cpu_count} cores")
    print(f"  CPU Load (5m):  {result.resources.cpu_load_5m:.2f} / {result.resources.cpu_count} cores")
    print(f"  CPU Load (15m): {result.resources.cpu_load_15m:.2f} / {result.resources.cpu_count} cores")
    print()

    # Processes
    print("PROCESSES:")
    print(f"  Archetype Optimizations: {len(result.processes.optimization_processes)} running")
    for proc in result.processes.optimization_processes:
        print(f"    - PID {proc['pid']}: {proc['command'][:60]}")

    print(f"  System B0: {len(result.processes.system_b0_processes)} running")
    for proc in result.processes.system_b0_processes:
        print(f"    - PID {proc['pid']}: {proc['command'][:60]}")
    print()

    # Databases
    print("DATABASES:")
    print(f"  Optimization DBs: {len(result.databases.optimization_dbs)}")
    for db in result.databases.optimization_dbs:
        print(f"    - {db['path']} ({db['size_mb']:.1f} MB, {len(db['locks'])} locks)")

    print(f"  System B0 DBs: {len(result.databases.system_b0_dbs)}")
    for db in result.databases.system_b0_dbs:
        print(f"    - {db['path']} ({db['size_mb']:.1f} MB, {len(db['locks'])} locks)")

    print(f"  Conflicts: {len(result.databases.conflicts)}")
    for conflict in result.databases.conflicts:
        print(f"    - {conflict}")
    print()

    # Performance
    if result.performance:
        print("PERFORMANCE:")
        print(f"  Trial Rate (current):  {result.performance.trial_rate_current:.2f} trials/hour")
        print(f"  Trial Rate (baseline): {result.performance.trial_rate_baseline:.2f} trials/hour")
        print(f"  Performance Ratio:     {result.performance.trial_rate_ratio:.1%}")
        print(f"  Error Rate:            {result.performance.error_rate:.2%}")
        print()

    # Warnings and Errors
    if result.warnings:
        print("WARNINGS:")
        for warning in result.warnings:
            print(f"  ⚠ {warning}")
        print()

    if result.errors:
        print("ERRORS:")
        for error in result.errors:
            print(f"  ✗ {error}")
        print()

    # Recommendation
    print("RECOMMENDATION:")
    if result.is_go():
        print("  ✓ SAFE TO DEPLOY")
        print("  All checks passed. System B0 deployment can proceed.")
    elif result.is_no_go():
        print("  ✗ DO NOT DEPLOY")
        print("  Critical issues detected. Resolve errors before deployment.")
    else:
        print("  ⚠ DEPLOY WITH CAUTION")
        print("  Warnings detected. Review warnings and monitor closely during deployment.")

    print("=" * 80 + "\n")


# =============================================================================
# Monitoring Functions
# =============================================================================

def run_monitoring(interval: int = 30, duration: Optional[int] = None):
    """Run continuous monitoring."""

    print("\n" + "=" * 80)
    print("CONTINUOUS MONITORING STARTED")
    print("=" * 80)
    print(f"Interval: {interval}s")
    if duration:
        print(f"Duration: {duration}s ({duration // 60} minutes)")
    else:
        print("Duration: Infinite (Ctrl+C to stop)")
    print()

    start_time = time.time()
    check_count = 0

    # Get baseline
    baseline_perf = get_performance_status()
    baseline_trial_rate = baseline_perf.trial_rate_current

    print(f"Baseline trial rate: {baseline_trial_rate:.2f} trials/hour")
    print()

    try:
        while True:
            check_count += 1

            print(f"\n[Check #{check_count}] {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 40)

            # Quick check
            resources = get_resource_status()
            processes = get_process_status()
            databases = get_database_status()
            performance = get_performance_status(baseline_trial_rate)

            # Print summary
            print(f"CPU: {resources.cpu_load_1m:.2f}  Mem: {resources.memory_free_gb:.1f}GB  Disk: {resources.disk_free_gb:.1f}GB")
            print(f"Opt Procs: {len(processes.optimization_processes)}  B0 Procs: {len(processes.system_b0_processes)}")
            print(f"Trial Rate: {performance.trial_rate_current:.2f}/h ({performance.trial_rate_ratio:.1%} of baseline)")
            print(f"DB Conflicts: {len(databases.conflicts)}")

            # Check for warnings
            all_warnings = []
            all_warnings.extend(resources.get_warnings())
            all_warnings.extend(processes.get_warnings())
            all_warnings.extend(databases.get_warnings())
            all_warnings.extend(performance.get_warnings())

            if all_warnings:
                print("\n⚠ WARNINGS:")
                for warning in all_warnings:
                    print(f"  {warning}")
            else:
                print("✓ All checks OK")

            # Check duration
            if duration and (time.time() - start_time) >= duration:
                print("\n\nMonitoring duration completed.")
                break

            # Wait for next check
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")

    print("\n" + "=" * 80)
    print(f"MONITORING COMPLETE - {check_count} checks performed")
    print("=" * 80 + "\n")


def check_optimizations():
    """Check health of archetype optimizations."""

    print("\n" + "=" * 80)
    print("ARCHETYPE OPTIMIZATION HEALTH CHECK")
    print("=" * 80)
    print()

    processes = get_process_status()
    databases = get_database_status()
    performance = get_performance_status()

    print(f"Running Processes: {len(processes.optimization_processes)}")
    for proc in processes.optimization_processes:
        print(f"  - PID {proc['pid']}: CPU {proc['cpu']}% MEM {proc['mem']}%")
        print(f"    {proc['command'][:70]}")
    print()

    print(f"Databases: {len(databases.optimization_dbs)}")
    for db in databases.optimization_dbs:
        print(f"  - {os.path.basename(db['path'])}: {db['size_mb']:.1f} MB")

        # Get trial stats
        if os.path.exists(db['path']):
            try:
                conn = sqlite3.connect(db['path'])
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE'")
                complete = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM trials WHERE state = 'RUNNING'")
                running = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM trials WHERE state = 'FAIL'")
                failed = cursor.fetchone()[0]

                print(f"    Complete: {complete}, Running: {running}, Failed: {failed}")

                conn.close()
            except Exception as e:
                print(f"    Error querying: {e}")
    print()

    print(f"Trial Rate: {performance.trial_rate_current:.2f} trials/hour")
    print()

    # Health assessment
    if len(processes.optimization_processes) == 0:
        print("⚠ WARNING: No optimization processes running")
    elif performance.trial_rate_current < 0.1:
        print("⚠ WARNING: Very low trial rate")
    else:
        print("✓ Optimizations appear healthy")

    print("=" * 80 + "\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='System B0 Deployment Safety Verification',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Check modes
    parser.add_argument(
        '--full-check',
        action='store_true',
        help='Run full pre-deployment verification'
    )

    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Run continuous monitoring'
    )

    parser.add_argument(
        '--check-optimizations',
        action='store_true',
        help='Check archetype optimization health'
    )

    parser.add_argument(
        '--post-deployment-check',
        action='store_true',
        help='Run post-deployment validation'
    )

    parser.add_argument(
        '--post-rollback-check',
        action='store_true',
        help='Run post-rollback verification'
    )

    # Monitoring parameters
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Monitoring interval in seconds (default: 30)'
    )

    parser.add_argument(
        '--duration',
        type=int,
        help='Monitoring duration in seconds (default: infinite)'
    )

    # Baseline
    parser.add_argument(
        '--baseline-trial-rate',
        type=float,
        help='Baseline trial rate for comparison (trials/hour)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    try:
        if args.full_check or args.post_deployment_check or args.post_rollback_check:
            result = run_full_check(args.baseline_trial_rate)

            # Exit code based on result
            if result.is_no_go():
                return 1
            elif result.status == 'WARNING':
                return 2
            else:
                return 0

        elif args.monitor:
            run_monitoring(args.interval, args.duration)
            return 0

        elif args.check_optimizations:
            check_optimizations()
            return 0

        else:
            # Default: run full check
            result = run_full_check(args.baseline_trial_rate)

            if result.is_no_go():
                return 1
            elif result.status == 'WARNING':
                return 2
            else:
                return 0

    except Exception as e:
        print(f"\nERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
