# DEPLOYMENT TESTING GUIDE - SYSTEM B0

## VERIFICATION SCRIPT USAGE

The `bin/verify_safe_deployment.py` script provides comprehensive deployment safety verification with multiple modes.

### Quick Start

```bash
# Pre-deployment check (required before deploying)
python3 bin/verify_safe_deployment.py --full-check

# Monitor during deployment (run in separate terminal)
python3 bin/verify_safe_deployment.py --monitor

# Check optimization health
python3 bin/verify_safe_deployment.py --check-optimizations

# Post-deployment validation
python3 bin/verify_safe_deployment.py --post-deployment-check

# Post-rollback verification
python3 bin/verify_safe_deployment.py --post-rollback-check
```

### Exit Codes

The script returns different exit codes for automation:

| Exit Code | Meaning | Action |
|-----------|---------|--------|
| 0 | GO - Safe to deploy | Proceed with deployment |
| 1 | NO-GO - Critical issues | Do NOT deploy, fix issues |
| 2 | WARNING - Non-critical issues | Deploy with caution, monitor closely |

### Usage in Scripts

```bash
#!/bin/bash

# Automated deployment with safety check
python3 bin/verify_safe_deployment.py --full-check
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Safety check passed - deploying..."
    python3 bin/backtest_system_b0.py --config configs/system_b0_production.json
elif [ $EXIT_CODE -eq 2 ]; then
    echo "⚠ Warnings detected - deploy with caution? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        python3 bin/backtest_system_b0.py --config configs/system_b0_production.json
    fi
else
    echo "✗ Critical issues detected - aborting deployment"
    exit 1
fi
```

## TESTING SCENARIOS

### Scenario 1: Pre-Deployment Check (No Optimizations Running)

**Setup**: No optimization processes running (idle system)

```bash
python3 bin/verify_safe_deployment.py --full-check
```

**Expected Result**:
- Status: WARNING or NO-GO (depends on resources)
- Warnings: "No archetype optimization processes detected"
- Resources: Should pass (disk, memory within limits)
- Recommendation: May deploy if only optimization warning

**Interpretation**:
- If only warning is "no optimization processes", it's safe to deploy
- If other critical issues (low disk, low memory, CPU overload), fix first

### Scenario 2: Pre-Deployment Check (Optimizations Running)

**Setup**: Archetype optimizations actively running

```bash
# Start an optimization (in background)
# python3 bin/optuna_parallel_archetypes_v2.py &

# Run check
python3 bin/verify_safe_deployment.py --full-check
```

**Expected Result**:
- Status: GO (if resources OK) or WARNING (if resources tight)
- Processes: 1-4 optimization processes detected
- Trial Rate: >0 trials/hour
- Resources: Within acceptable limits

**Interpretation**:
- GO status: Safe to deploy System B0
- WARNING status: Review warnings, ensure enough resources
- NO-GO status: Critical issues, do NOT deploy

### Scenario 3: During Deployment Monitoring

**Setup**: System B0 backtest running, optimizations may or may not be running

```bash
# Terminal 1: Run System B0 backtest
python3 bin/backtest_system_b0.py --config configs/system_b0_production.json &

# Terminal 2: Monitor
python3 bin/verify_safe_deployment.py --monitor --interval 30
```

**Expected Behavior**:
- Every 30 seconds, checks resources, processes, databases
- Reports CPU, memory, disk usage
- Alerts on threshold violations
- Continues until Ctrl+C

**What to Watch**:
- CPU load should stay reasonable (<8 warning, <10 critical)
- Memory should not decrease dramatically
- No database conflicts
- If optimizations running, trial rate should stay >70% baseline

### Scenario 4: Continuous Monitoring (1 Hour)

**Setup**: Monitor for extended period to detect trends

```bash
python3 bin/verify_safe_deployment.py --monitor --interval 30 --duration 3600
```

**Expected Behavior**:
- Runs for 1 hour (3600 seconds)
- Checks every 30 seconds (120 total checks)
- Logs all metrics
- Stops automatically after 1 hour

**What to Watch**:
- Memory leaks (gradual memory decrease)
- CPU saturation (sustained high load)
- Performance degradation (trial rate declining)

### Scenario 5: Post-Deployment Validation

**Setup**: System B0 deployed and running

```bash
python3 bin/verify_safe_deployment.py --post-deployment-check
```

**Expected Result**:
- Status: GO or WARNING (not NO-GO if deployed successfully)
- System B0 processes: 1+ running
- Resources: Stable
- No critical issues

**Interpretation**:
- GO: Deployment successful, system healthy
- WARNING: Deployment successful but monitor closely
- NO-GO: Something went wrong, investigate immediately

### Scenario 6: Post-Rollback Verification

**Setup**: System B0 stopped, optimizations should still be running

```bash
# Stop System B0
pkill -f system_b0

# Verify rollback
python3 bin/verify_safe_deployment.py --post-rollback-check
```

**Expected Result**:
- Status: GO or WARNING
- System B0 processes: 0 running (stopped successfully)
- Optimization processes: Still running (if were running before)
- Resources: Back to pre-deployment levels
- No database corruption

**Interpretation**:
- GO: Rollback successful, system restored
- WARNING: Rollback successful but some warnings
- NO-GO: Critical issue, optimizations may be affected

### Scenario 7: Check Optimization Health

**Setup**: Want to verify optimizations are running correctly

```bash
python3 bin/verify_safe_deployment.py --check-optimizations
```

**Expected Output**:
- Running Processes: List of optimization processes (or 0 if none)
- Databases: Trial counts (Complete, Running, Failed)
- Trial Rate: Trials per hour
- Health Assessment: HEALTHY or WARNING

**Example Output**:
```
Running Processes: 4
  - PID 1234: CPU 25.0% MEM 2.5%
    optuna_parallel_archetypes_v2.py

Databases: 12
  - optuna_production_v2_trap_within_trend.db: 0.3 MB
    Complete: 94, Running: 2, Failed: 0
  ...

Trial Rate: 12.50 trials/hour

✓ Optimizations appear healthy
```

## TROUBLESHOOTING

### Issue: Memory Shows as 0.0 GB

**Cause**: vm_stat parsing issue on some macOS versions

**Workaround**: The script defaults to 0 if it can't parse. Check manually:
```bash
vm_stat | grep "Pages free"
# Calculate: pages * 4096 / 1024^3 = GB
```

**Fix**: Script attempts to handle this with comma removal and better parsing.

### Issue: False "NO-GO" Due to No Optimizations

**Cause**: Script detects no optimization processes as a warning

**Solution**: This is expected if optimizations aren't running. As long as other resources are OK, you can deploy.

**Override**: If you're intentionally deploying without optimizations running:
```bash
# Check resources only
python3 bin/verify_safe_deployment.py --full-check | grep "SYSTEM RESOURCES"
```

### Issue: High CPU Load Warning

**Cause**: System under heavy load (optimizations, other processes)

**Solution**:
1. Check if load is from optimizations (expected) or other processes
2. Wait for load to decrease
3. If deploying anyway, reduce System B0 resource usage:
   - Use smaller batch sizes
   - Run backtest-only (no live trading)
   - Lower priority: `nice -n 10 python3 bin/backtest_system_b0.py ...`

### Issue: Database Lock Conflicts

**Cause**: Multiple processes writing to same database

**Solution**:
1. Stop both processes
2. Verify database integrity:
   ```bash
   sqlite3 problematic.db "PRAGMA integrity_check"
   ```
3. Ensure System B0 uses separate database
4. Check configs for correct database paths

### Issue: Trial Rate Shows 0.0%

**Cause**: No recent trials completed, or optimizations not running

**Solution**:
1. Check if optimizations are actually running
2. Look at database to see last trial timestamp:
   ```bash
   sqlite3 optuna_production_v2_*.db "SELECT MAX(datetime_complete) FROM trials"
   ```
3. If trials are old, optimizations may have stalled

## INTEGRATION WITH DEPLOYMENT

### Complete Deployment Workflow

```bash
#!/bin/bash
# safe_deploy_system_b0.sh

set -e  # Exit on error

echo "Step 1: Pre-deployment verification..."
python3 bin/verify_safe_deployment.py --full-check
EXIT_CODE=$?

if [ $EXIT_CODE -eq 1 ]; then
    echo "✗ Pre-deployment check failed - aborting"
    exit 1
elif [ $EXIT_CODE -eq 2 ]; then
    echo "⚠ Warnings detected - review before continuing"
    python3 bin/verify_safe_deployment.py --full-check | grep "WARNINGS:" -A 20
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✓ Pre-deployment check passed"
echo

# Record baseline
echo "Step 2: Recording baseline metrics..."
python3 bin/verify_safe_deployment.py --check-optimizations > /tmp/baseline_before_deploy.txt
BASELINE_TRIAL_RATE=$(grep "Trial Rate:" /tmp/baseline_before_deploy.txt | awk '{print $3}')
echo "Baseline trial rate: $BASELINE_TRIAL_RATE trials/hour"
echo

# Deploy
echo "Step 3: Deploying System B0 (backtest mode)..."
python3 bin/backtest_system_b0.py \
    --config configs/system_b0_production.json \
    --output results/system_b0/backtest_$(date +%Y%m%d_%H%M%S).json &

DEPLOY_PID=$!
echo "System B0 PID: $DEPLOY_PID"
echo

# Monitor
echo "Step 4: Monitoring deployment (5 minutes)..."
python3 bin/verify_safe_deployment.py \
    --monitor \
    --interval 30 \
    --duration 300 \
    --baseline-trial-rate $BASELINE_TRIAL_RATE

# Check if deployment still running
if ps -p $DEPLOY_PID > /dev/null; then
    echo "✓ System B0 still running - deployment successful so far"
else
    wait $DEPLOY_PID
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ System B0 completed successfully"
    else
        echo "✗ System B0 failed with exit code $EXIT_CODE"
        exit 1
    fi
fi

# Post-deployment validation
echo
echo "Step 5: Post-deployment validation..."
python3 bin/verify_safe_deployment.py --post-deployment-check
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Deployment completed successfully"
    exit 0
elif [ $EXIT_CODE -eq 2 ]; then
    echo "⚠ Deployment completed with warnings - monitor closely"
    exit 0
else
    echo "✗ Deployment validation failed - consider rollback"
    exit 1
fi
```

### Rollback Workflow

```bash
#!/bin/bash
# emergency_rollback_system_b0.sh

set -e

echo "EMERGENCY ROLLBACK - SYSTEM B0"
echo "=============================="
echo

echo "Step 1: Stopping System B0..."
pkill -f system_b0 || echo "No System B0 processes found"
sleep 2

echo "Step 2: Verifying System B0 stopped..."
if ps aux | grep -v grep | grep system_b0 > /dev/null; then
    echo "✗ System B0 still running - force kill"
    pkill -9 -f system_b0
    sleep 2
fi

if ps aux | grep -v grep | grep system_b0 > /dev/null; then
    echo "✗ CRITICAL: Cannot stop System B0 - manual intervention required"
    exit 1
else
    echo "✓ System B0 stopped"
fi

echo
echo "Step 3: Checking optimization health..."
python3 bin/verify_safe_deployment.py --check-optimizations

echo
echo "Step 4: Post-rollback verification..."
python3 bin/verify_safe_deployment.py --post-rollback-check
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Rollback successful - system restored"
    exit 0
else
    echo "✗ Rollback issues detected - review output above"
    exit 1
fi
```

## AUTOMATION AND CI/CD

### GitHub Actions Integration

```yaml
# .github/workflows/deploy_system_b0.yml
name: Deploy System B0

on:
  workflow_dispatch:
    inputs:
      mode:
        description: 'Deployment mode'
        required: true
        default: 'backtest'
        type: choice
        options:
          - backtest
          - paper
          - live

jobs:
  deploy:
    runs-on: self-hosted  # Your trading server

    steps:
      - uses: actions/checkout@v3

      - name: Pre-deployment check
        id: precheck
        run: |
          python3 bin/verify_safe_deployment.py --full-check
          echo "exit_code=$?" >> $GITHUB_OUTPUT
        continue-on-error: true

      - name: Evaluate pre-check
        if: steps.precheck.outputs.exit_code == '1'
        run: |
          echo "❌ Pre-deployment check failed - aborting"
          exit 1

      - name: Deploy System B0
        run: |
          python3 bin/backtest_system_b0.py \
            --config configs/system_b0_production.json \
            --output results/system_b0/backtest_$(date +%Y%m%d_%H%M%S).json

      - name: Post-deployment validation
        run: |
          python3 bin/verify_safe_deployment.py --post-deployment-check
```

## PERFORMANCE BENCHMARKS

### Expected Resource Usage (Idle System)

| Metric | Value |
|--------|-------|
| Disk Free | 355 GB |
| Memory Free | 4-8 GB |
| CPU Load (1m) | 1-3 |
| Optimization Processes | 0 |
| Trial Rate | 0 trials/hour |

### Expected Resource Usage (Optimizations Running)

| Metric | Value |
|--------|-------|
| Disk Free | 350+ GB |
| Memory Free | 2-6 GB |
| CPU Load (1m) | 4-8 |
| Optimization Processes | 1-4 |
| Trial Rate | 5-20 trials/hour |

### Expected Resource Usage (Optimizations + System B0 Backtest)

| Metric | Value |
|--------|-------|
| Disk Free | 350+ GB |
| Memory Free | 1-5 GB |
| CPU Load (1m) | 5-10 |
| Optimization Processes | 1-4 |
| System B0 Processes | 1 |
| Trial Rate | 4-18 trials/hour (80-90% of baseline) |

## LOGGING AND DIAGNOSTICS

### Log Locations

| Log Type | Location |
|----------|----------|
| Verification Script | `logs/deployment_monitor.log` (if using --log option) |
| System B0 | `logs/system_b0/*.log` |
| Optimizations | `logs/archetype_optimization.log` |
| Alerts | `logs/system_b0/alerts.jsonl` |

### Manual Diagnostics

```bash
# Check system resources
df -h .
vm_stat | grep "Pages free"
uptime

# Check running processes
ps aux | grep -E "(optuna|system_b0)"

# Check database locks
lsof *.db

# Check recent errors
tail -100 logs/system_b0/*.log | grep -i error

# Check trial completion
sqlite3 optuna_production_v2_trap_within_trend.db \
  "SELECT COUNT(*) FROM trials WHERE state='COMPLETE' AND datetime(datetime_complete) > datetime('now', '-1 hour')"

# Check database integrity
sqlite3 data/bull_machine.db "PRAGMA integrity_check"
```

## CHECKLISTS

### Pre-Deployment Checklist

- [ ] Run `python3 bin/verify_safe_deployment.py --full-check`
- [ ] Status is GO or WARNING (not NO-GO)
- [ ] Disk space >10GB free
- [ ] Memory >2GB free (4GB+ ideal)
- [ ] CPU load <10
- [ ] No database conflicts
- [ ] Git status clean (or changes committed)

### During Deployment Checklist

- [ ] Monitoring running: `python3 bin/verify_safe_deployment.py --monitor`
- [ ] CPU load not spiking >10 sustained
- [ ] Memory stable (not decreasing rapidly)
- [ ] No error alerts
- [ ] If optimizations running, trial rate >70% baseline

### Post-Deployment Checklist

- [ ] Run `python3 bin/verify_safe_deployment.py --post-deployment-check`
- [ ] System B0 process running (if continuous) or completed successfully
- [ ] Optimizations still running (if were running before)
- [ ] Resources returned to normal
- [ ] No database corruption
- [ ] Backtest results look reasonable

### Rollback Checklist

- [ ] Stop System B0: `pkill -f system_b0`
- [ ] Verify stopped: `ps aux | grep system_b0` (no results)
- [ ] Check optimizations: `python3 bin/verify_safe_deployment.py --check-optimizations`
- [ ] Post-rollback check: `python3 bin/verify_safe_deployment.py --post-rollback-check`
- [ ] Status is GO (system restored)

---

**END OF DEPLOYMENT TESTING GUIDE**

**Version**: 1.0
**Last Updated**: 2025-12-04
**Owner**: Backend Architect
**Status**: Production Ready
