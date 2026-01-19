# DEPLOYMENT SAFETY CHECKLIST - SYSTEM B0

**CRITICAL**: This checklist ensures System B0 deployment does not interfere with ongoing archetype optimizations.

## TABLE OF CONTENTS
1. [Pre-Flight Checklist](#pre-flight-checklist)
2. [Deployment Procedure](#deployment-procedure)
3. [Monitoring During Deployment](#monitoring-during-deployment)
4. [Emergency Rollback](#emergency-rollback)
5. [Post-Deployment Validation](#post-deployment-validation)

---

## PRE-FLIGHT CHECKLIST

**DO NOT PROCEED UNTIL ALL ITEMS ARE VERIFIED**

### System Resources
- [ ] **Disk Space**: Minimum 10GB free (System B0 logs/results)
  - Check: `df -h .`
  - Current available: _____GB
  - PASS THRESHOLD: >10GB

- [ ] **Memory Available**: Minimum 4GB free
  - Check: `vm_stat | grep "Pages free"`
  - Current free: _____GB
  - PASS THRESHOLD: >4GB

- [ ] **CPU Load**: Load average <8 (12 cores available)
  - Check: `uptime`
  - Current load: _____
  - PASS THRESHOLD: <8.0

### Process Verification
- [ ] **Archetype Optimizations Running**
  - Check: `python bin/verify_safe_deployment.py --check-processes`
  - Processes found: _____
  - EXPECTED: 1-4 optimization processes (optuna/parallel)

- [ ] **No Database Locks**
  - Check: `lsof *.db | grep -v "READ"`
  - Write locks detected: _____
  - PASS THRESHOLD: 0 (System B0 uses separate DB)

### Data Integrity
- [ ] **Feature Store Available (READ-ONLY)**
  - Check: `ls -lh data/bull_machine.db`
  - Size: _____MB
  - Permissions: Should be readable by all processes

- [ ] **Historical Data Complete**
  - Check: `ls -lh data/*.csv | wc -l`
  - Files found: _____
  - EXPECTED: 15-20 CSV files (BTC, ETH, SOL, etc.)

- [ ] **No Shared Database Files with Optimizations**
  - System B0 uses: `data/system_b0_production.db` (NEW)
  - Optimizations use: `optuna_*.db` (SEPARATE)
  - Conflict risk: **NONE**

### Configuration Validation
- [ ] **System B0 Config Exists**
  - Check: `ls -lh configs/system_b0_production.json`
  - Status: _____

- [ ] **Config Uses Correct Data Sources**
  - Check: `python bin/verify_safe_deployment.py --validate-config`
  - Data sources: READ-ONLY from feature store
  - No writes to optimization databases: CONFIRMED

- [ ] **Port Availability (if live trading)**
  - Check: `lsof -i :8000-8100`
  - Ports in use: _____
  - System B0 needs: None (backtest mode) OR port 8080 (live mode)

### Backup Readiness
- [ ] **Git Status Clean**
  - Check: `git status --porcelain | wc -l`
  - Uncommitted changes: _____
  - RECOMMENDATION: Commit before deployment

- [ ] **Can Quickly Revert**
  - System B0 files can be removed without affecting optimizations
  - No shared state between systems
  - RISK LEVEL: **LOW**

### Final GO/NO-GO Decision

Run automated verification:
```bash
python bin/verify_safe_deployment.py --full-check
```

**RESULT**: ____ (GO / NO-GO)

If **NO-GO**, see diagnostic output and resolve issues before proceeding.

---

## DEPLOYMENT PROCEDURE

### Phase 1: Minimal Impact Deployment (SAFEST)

**GOAL**: Deploy System B0 in backtest-only mode with zero runtime interference.

#### Step 1: Create Isolated Environment
```bash
# Create deployment workspace
mkdir -p logs/system_b0
mkdir -p results/system_b0

# Verify no conflicts
python bin/verify_safe_deployment.py --check-conflicts
```

**Expected**: No conflicts detected, all paths isolated.

#### Step 2: Validate Configuration
```bash
# Validate System B0 config
python bin/validate_system_b0.py --config configs/system_b0_production.json

# Check resource requirements
python bin/verify_safe_deployment.py --estimate-resources
```

**Expected**:
- Config valid
- Estimated CPU: 10-20% of 1 core
- Estimated Memory: 200-500MB
- Estimated Disk I/O: Minimal (read-only from cached data)

#### Step 3: Dry Run (No Execution)
```bash
# Dry run - validate without execution
python bin/validate_system_b0.py --dry-run --config configs/system_b0_production.json
```

**Expected**: All checks pass, no actual backtests run.

#### Step 4: Single Backtest Test
```bash
# Run single backtest on small dataset (2023 only)
python bin/backtest_system_b0.py \
  --config configs/system_b0_production.json \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --output results/system_b0/test_backtest.json
```

**MONITOR**:
- Check process doesn't interfere: `python bin/verify_safe_deployment.py --monitor`
- Expected runtime: 2-5 minutes
- Memory usage: <500MB
- CPU usage: <25%

**CHECKPOINT 1**: If test backtest completes successfully with no interference, proceed.

#### Step 5: Full Historical Backtest
```bash
# Run complete historical backtest
python bin/backtest_system_b0.py \
  --config configs/system_b0_production.json \
  --output results/system_b0/full_backtest_$(date +%Y%m%d).json \
  --verbose
```

**MONITOR**:
```bash
# In separate terminal, monitor continuously
watch -n 30 'python bin/verify_safe_deployment.py --monitor'
```

**Expected**:
- Runtime: 10-30 minutes
- Archetype optimizations continue unaffected
- CPU load increase: <15%
- Memory increase: <1GB

**CHECKPOINT 2**: If full backtest completes with no interference, System B0 is safely deployed in backtest mode.

---

### Phase 2: Paper Trading Deployment (MEDIUM RISK)

**PREREQUISITES**: Phase 1 completed successfully.

**GOAL**: Deploy System B0 in paper trading mode (simulated orders, no real money).

#### Step 1: Paper Trading Configuration
```bash
# Create paper trading config (copy from production, set paper_trading=true)
cp configs/system_b0_production.json configs/system_b0_paper.json

# Edit config to enable paper trading
# Set: "mode": "paper_trading"
```

#### Step 2: Validate Paper Trading Setup
```bash
# Validate paper trading config
python bin/validate_system_b0.py --config configs/system_b0_paper.json --mode paper

# Check exchange connectivity (if needed)
python bin/test_exchange_connection.py --dry-run
```

**Expected**: Paper trading mode configured, no real order execution possible.

#### Step 3: Start Paper Trading (Background)
```bash
# Start in background with monitoring
nohup python bin/run_system_b0.py \
  --config configs/system_b0_paper.json \
  --mode paper \
  > logs/system_b0/paper_trading.log 2>&1 &

# Save process ID
echo $! > logs/system_b0/paper_trading.pid
```

**MONITOR**:
```bash
# Monitor for first 10 minutes
for i in {1..20}; do
  python bin/verify_safe_deployment.py --monitor
  sleep 30
done
```

#### Step 4: Verify No Interference
```bash
# Check archetype optimizations still healthy
python bin/verify_safe_deployment.py --check-optimizations

# Check System B0 running correctly
python bin/monitor_system_b0.py --once
```

**CHECKPOINT 3**: If paper trading runs for 1 hour with no issues, proceed to live deployment.

---

### Phase 3: Live Trading Deployment (HIGHEST RISK)

**PREREQUISITES**:
- Phase 1 and 2 completed successfully
- User explicitly authorizes live trading
- Risk management verified
- Exchange API keys configured

**WARNING**: This involves real money. Triple-check all settings.

#### Step 1: Pre-Live Verification
```bash
# Run comprehensive pre-live checks
python bin/verify_safe_deployment.py --pre-live-check

# Validate risk parameters
python bin/validate_system_b0.py --validate-risk --config configs/system_b0_production.json
```

**MANUAL VERIFICATION REQUIRED**:
- [ ] Position sizing correct
- [ ] Stop losses configured
- [ ] Maximum drawdown limits set
- [ ] Emergency shutdown tested
- [ ] User explicitly authorizes live trading

#### Step 2: Start Live Trading (Monitored)
```bash
# Start with minimal position size (10% of normal)
python bin/run_system_b0.py \
  --config configs/system_b0_production.json \
  --mode live \
  --position-size-multiplier 0.1 \
  --max-positions 1
```

**MONITOR CONTINUOUSLY**:
```bash
# Real-time monitoring dashboard
python bin/monitor_system_b0.py --interval 60
```

#### Step 3: Gradual Scale-Up
After 24 hours of successful operation:
- Increase position size to 25%
- After 48 hours: 50%
- After 72 hours: 100%

**CHECKPOINT 4**: System B0 fully deployed in live trading mode.

---

## MONITORING DURING DEPLOYMENT

### Continuous Monitoring (Run in Separate Terminal)

```bash
# Start deployment monitor
watch -n 30 'python bin/verify_safe_deployment.py --monitor'
```

**What This Checks**:
1. Archetype optimization processes still running
2. System B0 process health
3. CPU/Memory/Disk usage
4. No database conflicts
5. No error spikes in logs

### Key Metrics to Watch

#### System Resources
| Metric | Normal Range | Warning Threshold | Critical Threshold |
|--------|--------------|-------------------|-------------------|
| CPU Load | 2-6 | >8 | >10 |
| Memory Free | >4GB | <2GB | <1GB |
| Disk Free | >10GB | <5GB | <2GB |
| I/O Wait | <5% | >10% | >20% |

#### Process Health
| Metric | Expected | Warning | Critical |
|--------|----------|---------|----------|
| Archetype Optimizations | Running | Stalled | Crashed |
| System B0 | Running | Errors in logs | Crashed |
| Database Locks | 0 write conflicts | 1-2 conflicts | >3 conflicts |

#### Performance Impact
| Metric | Baseline | Acceptable | Unacceptable |
|--------|----------|------------|--------------|
| Optimization Speed | 100% | >80% | <70% |
| Memory per Process | Normal | +20% | +50% |
| Trial Completion Rate | Normal | -10% | -25% |

### Red Flags - STOP DEPLOYMENT IMMEDIATELY

**CRITICAL ISSUES** (Rollback immediately):
- Archetype optimization processes crashed
- Memory exhausted (OOM killer activated)
- Disk full
- Database corruption detected
- Trial completion rate drops >30%

**WARNING ISSUES** (Investigate before proceeding):
- CPU load >10 sustained
- Memory free <1GB
- Optimization speed degraded >20%
- Error rate increased in logs
- Database lock contention detected

### Manual Verification Commands

```bash
# Check archetype optimizations
ps aux | grep optuna

# Check System B0
ps aux | grep system_b0

# Check database usage
lsof *.db

# Check disk usage
df -h .

# Check memory
vm_stat | grep "Pages free"

# Check CPU load
uptime

# Check recent errors
tail -50 logs/system_b0/*.log | grep -i error

# Check trial completion (optimizations)
sqlite3 optuna_production_v2_*.db "SELECT COUNT(*) FROM trials WHERE state='COMPLETE' AND datetime(start_datetime, 'unixepoch') > datetime('now', '-1 hour')"
```

### Automated Monitoring Script

```bash
# Run continuous monitoring (logs to file)
python bin/verify_safe_deployment.py --monitor --interval 30 --duration 3600
```

This will:
- Check every 30 seconds
- Run for 1 hour (3600 seconds)
- Log all metrics to `logs/deployment_monitor.log`
- Alert on any threshold violations

---

## EMERGENCY ROLLBACK

### Quick Rollback Procedure (1-Page Guide)

**TIME TO ROLLBACK: <2 MINUTES**

#### Step 1: Stop System B0 (Immediately)
```bash
# Find and kill System B0 process
pkill -f "system_b0" || pkill -f "run_system_b0"

# Verify stopped
ps aux | grep system_b0
# Expected: No processes
```

#### Step 2: Verify Archetype Optimizations Still Running
```bash
# Check optimization processes
ps aux | grep optuna
# Expected: 1-4 processes still running

# If crashed, check why
tail -100 logs/archetype_optimization.log
```

#### Step 3: Clean Up System B0 Resources
```bash
# Remove PID file
rm -f logs/system_b0/*.pid

# Archive logs for post-mortem
mkdir -p logs/system_b0/rollback_$(date +%Y%m%d_%H%M%S)
mv logs/system_b0/*.log logs/system_b0/rollback_$(date +%Y%m%d_%H%M%S)/

# Remove partial results
rm -f results/system_b0/temp_*
```

#### Step 4: Verify System Stability
```bash
# Run full system check
python bin/verify_safe_deployment.py --post-rollback-check
```

**Expected**:
- Archetype optimizations: RUNNING
- System B0: STOPPED
- Resources: NORMAL
- No corruption: CONFIRMED

#### Step 5: Resume Normal Operations
```bash
# Check archetype optimization health
python bin/verify_safe_deployment.py --check-optimizations

# Verify no data corruption
sqlite3 optuna_production_v2_*.db "PRAGMA integrity_check"
# Expected: "ok"
```

### Rollback Decision Matrix

| Symptom | Severity | Action | Rollback? |
|---------|----------|--------|-----------|
| System B0 high CPU | Low | Adjust config | No |
| System B0 errors in logs | Medium | Investigate | Maybe |
| Archetype optimizations slow | High | Monitor closely | Yes if >30% degradation |
| Archetype optimizations crashed | Critical | ROLLBACK NOW | YES |
| Database lock errors | High | ROLLBACK NOW | YES |
| Memory exhausted | Critical | ROLLBACK NOW | YES |
| Disk full | Critical | ROLLBACK NOW | YES |

### Post-Rollback Recovery

#### If Archetype Optimizations Were Affected:

1. **Check for lost trials**:
```bash
sqlite3 optuna_production_v2_*.db "SELECT COUNT(*) FROM trials WHERE state='RUNNING'"
# Expected: 0 (no stuck trials)
```

2. **Resume from last checkpoint**:
```bash
# Optimizations should auto-resume
# Verify with:
python bin/verify_safe_deployment.py --check-optimizations
```

3. **Estimate impact**:
```bash
# Check last successful trial timestamp
sqlite3 optuna_production_v2_*.db "SELECT MAX(datetime_complete) FROM trials WHERE state='COMPLETE'"
# Compare to current time - how much was lost?
```

#### If Data Corruption Detected:

1. **Restore from backup** (if needed):
```bash
# Check if backup exists
ls -lh backups/bull_machine_*.db

# Restore if needed (ONLY if corruption confirmed)
# cp backups/bull_machine_YYYYMMDD.db data/bull_machine.db
```

2. **Re-run integrity checks**:
```bash
python bin/verify_safe_deployment.py --integrity-check
```

### Rollback Checklist

After rollback, verify:
- [ ] System B0 completely stopped (no processes)
- [ ] Archetype optimizations still running
- [ ] Database integrity confirmed
- [ ] Disk space recovered (if needed)
- [ ] Memory usage normal
- [ ] CPU load normal
- [ ] No error spikes in logs
- [ ] Trial completion rate back to normal

**ROLLBACK COMPLETE**: System restored to pre-deployment state.

---

## POST-DEPLOYMENT VALIDATION

### Immediate Validation (First 10 Minutes)

```bash
# Run quick validation
python bin/verify_safe_deployment.py --post-deployment-check
```

**Checks**:
- System B0 started successfully
- Archetype optimizations unaffected
- Resource usage within bounds
- No error spikes

### Short-Term Validation (First Hour)

```bash
# Monitor for 1 hour
python bin/verify_safe_deployment.py --monitor --duration 3600
```

**Watch For**:
- Memory leaks (gradual increase)
- CPU saturation
- Disk space consumption
- Error rate trends

### Medium-Term Validation (First 24 Hours)

**Manual Checks**:

1. **Archetype Optimization Progress**:
```bash
# Compare trials before/after deployment
python bin/analyze_optuna_results.py --compare-periods
```
Expected: No degradation in trial completion rate.

2. **System B0 Performance**:
```bash
# Check System B0 metrics
python bin/monitor_system_b0.py --summary --last 24h
```
Expected: Stable performance, no crashes.

3. **Resource Trends**:
```bash
# Check resource usage trends
python bin/verify_safe_deployment.py --resource-report --last 24h
```
Expected: No concerning trends (memory leaks, disk growth, etc.).

### Long-Term Validation (First Week)

**Weekly Check**:
```bash
# Full system health report
python bin/verify_safe_deployment.py --weekly-report
```

**Review**:
- [ ] System B0 stability (uptime, error rate)
- [ ] Archetype optimization impact (compare to baseline)
- [ ] Resource utilization trends
- [ ] Any unexpected behaviors

### Validation Metrics

#### Success Criteria (All Must Pass)

| Metric | Target | Measurement |
|--------|--------|-------------|
| System B0 Uptime | >99% | No crashes, clean restarts only |
| Archetype Optimization Speed | >95% of baseline | Trial completion rate |
| Memory Usage | Stable | No leaks, <10% increase |
| CPU Impact | <10% average | Load average increase |
| Disk Growth | <1GB/day | Log files, results |
| Error Rate | <0.1% | Errors per operation |

#### Failure Criteria (Any Triggers Rollback)

| Metric | Threshold | Action |
|--------|-----------|--------|
| System B0 Crashes | >1 per day | Rollback |
| Archetype Optimization Speed | <70% baseline | Rollback |
| Memory Leak | >100MB/hour | Investigate, rollback if severe |
| CPU Saturation | >80% sustained | Rollback |
| Error Rate | >1% | Rollback |

### Validation Report Template

```markdown
# System B0 Deployment Validation Report

**Deployment Date**: _______________
**Validation Period**: _______________
**Validation Status**: PASS / FAIL

## Deployment Phase
- Phase completed: Backtest / Paper Trading / Live Trading
- Deployment duration: _____ hours

## System Health

### System B0
- Uptime: _____%
- Total operations: _____
- Errors: _____
- Error rate: _____%

### Archetype Optimizations
- Trials before deployment (24h): _____
- Trials after deployment (24h): _____
- Speed impact: _____%
- Status: HEALTHY / DEGRADED / CRITICAL

### Resource Usage
- CPU: Baseline _____ → Current _____
- Memory: Baseline _____ → Current _____
- Disk: Baseline _____ → Current _____

## Issues Detected
1. _____
2. _____
3. _____

## Recommendations
- [ ] Continue deployment
- [ ] Rollback
- [ ] Investigate and hold

**Validator**: _______________
**Sign-off**: _______________
```

---

## APPENDIX: Troubleshooting

### Common Issues and Solutions

#### Issue: System B0 High Memory Usage
**Symptom**: Memory usage >1GB
**Cause**: Large dataframes in memory
**Solution**:
```bash
# Reduce batch size in config
# Set: "batch_size": 1000 (instead of 10000)
```

#### Issue: Database Lock Contention
**Symptom**: "database is locked" errors
**Cause**: Concurrent writes to same DB
**Solution**:
```bash
# Verify System B0 uses separate database
grep "database" configs/system_b0_production.json
# Should show: "database": "data/system_b0_production.db"
```

#### Issue: Archetype Optimizations Slowed Down
**Symptom**: Trial completion rate decreased
**Cause**: CPU contention
**Solution**:
```bash
# Reduce System B0 priority
renice +10 $(pgrep -f system_b0)
```

#### Issue: Disk Space Running Out
**Symptom**: Disk free <5GB
**Cause**: Excessive logging
**Solution**:
```bash
# Compress old logs
gzip logs/system_b0/*.log.1
gzip logs/system_b0/*.log.2

# Clean up old results
rm -f results/system_b0/temp_*
```

---

## QUICK REFERENCE

### Pre-Deployment Command
```bash
python bin/verify_safe_deployment.py --full-check
```

### Deployment Command (Safest)
```bash
python bin/backtest_system_b0.py --config configs/system_b0_production.json
```

### Monitoring Command
```bash
python bin/verify_safe_deployment.py --monitor
```

### Rollback Command
```bash
pkill -f system_b0 && python bin/verify_safe_deployment.py --post-rollback-check
```

### Validation Command
```bash
python bin/verify_safe_deployment.py --post-deployment-check
```

---

**END OF DEPLOYMENT SAFETY CHECKLIST**

**Version**: 1.0
**Last Updated**: 2025-12-04
**Owner**: Backend Architect
**Status**: Production Ready
