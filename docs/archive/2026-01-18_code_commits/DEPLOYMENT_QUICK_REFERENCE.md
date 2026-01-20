# DEPLOYMENT QUICK REFERENCE - SYSTEM B0

**1-PAGE EMERGENCY GUIDE**

## PRE-DEPLOYMENT (30 SECONDS)

```bash
# Run automated safety check
python bin/verify_safe_deployment.py --full-check

# Expected: "✓ GO - SAFE TO DEPLOY"
# If NO-GO: Fix errors before proceeding
# If WARNING: Review warnings, proceed with caution
```

## SAFEST DEPLOYMENT (5 MINUTES)

```bash
# Step 1: Verify safety
python bin/verify_safe_deployment.py --full-check

# Step 2: Run backtest (minimal impact)
python bin/backtest_system_b0.py \
  --config configs/system_b0_production.json \
  --output results/system_b0/backtest_$(date +%Y%m%d).json

# Step 3: Monitor (in separate terminal)
python bin/verify_safe_deployment.py --monitor --interval 30

# Expected: Backtest completes, no interference with optimizations
```

## MONITORING (CONTINUOUS)

```bash
# Start continuous monitoring
python bin/verify_safe_deployment.py --monitor

# What it checks:
# - CPU/Memory/Disk resources
# - Process health (optimizations + System B0)
# - Database conflicts
# - Performance degradation
```

**Watch For**:
- CPU load >8 (warning) or >10 (critical)
- Memory <4GB free
- Disk <10GB free
- Trial rate <70% of baseline
- Database conflicts

## EMERGENCY ROLLBACK (30 SECONDS)

```bash
# Step 1: Stop System B0 immediately
pkill -f system_b0

# Step 2: Verify stopped
ps aux | grep system_b0
# Expected: No processes

# Step 3: Verify optimizations still running
python bin/verify_safe_deployment.py --check-optimizations
# Expected: Processes running, healthy trial rate

# Step 4: Full post-rollback check
python bin/verify_safe_deployment.py --post-rollback-check
# Expected: ✓ GO - System restored
```

## VALIDATION (POST-DEPLOYMENT)

```bash
# Immediate validation
python bin/verify_safe_deployment.py --post-deployment-check

# 1-hour monitoring
python bin/verify_safe_deployment.py --monitor --duration 3600
```

## KEY COMMANDS

| Task | Command |
|------|---------|
| Pre-flight check | `python bin/verify_safe_deployment.py --full-check` |
| Monitor deployment | `python bin/verify_safe_deployment.py --monitor` |
| Check optimizations | `python bin/verify_safe_deployment.py --check-optimizations` |
| Emergency stop | `pkill -f system_b0` |
| Post-rollback verify | `python bin/verify_safe_deployment.py --post-rollback-check` |

## DECISION TREE

```
START
  |
  v
Run full-check
  |
  +--> NO-GO? --> Fix errors --> Retry
  |
  +--> WARNING? --> Review warnings --> Proceed with caution
  |
  +--> GO? --> Deploy
                  |
                  v
              Monitor
                  |
                  +--> No issues? --> Continue
                  |
                  +--> Warnings? --> Monitor closely
                  |
                  +--> Critical issues? --> ROLLBACK
```

## CRITICAL THRESHOLDS

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| CPU Load | >8 | >10 | Rollback if sustained |
| Memory Free | <4GB | <2GB | Rollback immediately |
| Disk Free | <10GB | <5GB | Rollback immediately |
| Trial Rate | <80% baseline | <70% baseline | Rollback if critical |
| Error Rate | >0.1% | >1% | Rollback if critical |

## ROLLBACK TRIGGERS

**IMMEDIATE ROLLBACK** if any:
- Archetype optimizations crashed
- Memory exhausted
- Disk full
- Database corruption
- Trial rate <70% baseline

**INVESTIGATE** if:
- CPU load >8 sustained
- Memory <4GB free
- Trial rate 70-80% baseline
- Warnings in logs

## FILES CREATED

- `/Users/raymondghandchi/Bull-machine-/Bull-machine-/DEPLOYMENT_SAFETY_CHECKLIST.md` - Full checklist
- `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/verify_safe_deployment.py` - Verification script
- `/Users/raymondghandchi/Bull-machine-/Bull-machine-/DEPLOYMENT_QUICK_REFERENCE.md` - This file

## SUPPORT

If issues arise:
1. Stop System B0: `pkill -f system_b0`
2. Run diagnostics: `python bin/verify_safe_deployment.py --full-check`
3. Check logs: `tail -100 logs/system_b0/*.log`
4. Verify optimizations: `python bin/verify_safe_deployment.py --check-optimizations`

---

**REMEMBER**: When in doubt, ROLLBACK. Archetype optimizations are critical and must not be interrupted.
