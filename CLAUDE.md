# Claude AI Development Guidelines & Known Issues

## Critical Issues & Fixes Log

### Issue #1: Zero Signals Generated (2026-01-29) ✅ FIXED

**Symptom**: Backtest runs complete but generate 0 signals, 0 trades
```
Total signals: 0
Signals taken: 0
Total Trades: 0
```

**Root Cause**: Hard gate checks in archetype logic are too strict, blocking all signals
- ADX threshold requires >= 25.0, but most data has ADX < 25
- Missing archetype parameters in config cause fallback to strict defaults
- Wick ratio, RSI, liquidity checks failing

**Fix Applied** (2026-01-29):
Added archetype-specific parameters to `configs/nautilus_production_full_soul.json`:
- `trap_within_trend`: Added `adx_threshold: 15.0` (was 25.0), `liquidity_threshold: 0.50` (was 0.30)
- `wick_trap`: Added `wick_lower_threshold: 0.60` (was 0.75), `rsi_threshold: 40` (was 35)
- `spring`: Added `pti_score_threshold: 0.25` (was 0.30), `disp_atr_multiplier: 0.4` (was 0.5)
- `order_block_retest`: Added `boms_strength_min: 0.25`, `wyckoff_min: 0.30`
- `long_squeeze`: Added `funding_z_min: 1.0`, `rsi_min: 65`, `liquidity_max: 0.30`
- `funding_divergence`: Added `funding_z_max: -1.0`, `resilience_min: 0.4`, `liquidity_max: 0.35`

**Verification**: ✅ Q1 2023 backtest generated 2 trades, system working correctly

**Prevention**:
- Always run backtest after making archetype logic changes
- Keep a "baseline test" config that generates known number of signals
- Add integration test that verifies signal generation
- Maintain complete parameter configs for all enabled archetypes

---

### Issue #2: Zombie Position Bug (2026-01-29)

**Symptom**: Positions never fully close, blocking all future entries
- Only 2 positions taken in 7 years of data
- 5,879 signals blocked by "existing position" with tiny quantity

**Root Cause**: Exit percentages calculated from CURRENT position instead of ORIGINAL
```python
# WRONG
exit_qty = current_position.quantity * exit_pct  # Compounds: 20% of 80% = 16%, not 20%

# CORRECT
exit_qty = original_position.quantity * exit_pct  # Always relative to original
```

**Fix Applied**:
- `engine/integrations/nautilus_strategy.py:382`: Store `original_quantity` at entry
- `engine/integrations/nautilus_strategy.py:618-627`: Calculate exits from original quantity
- `engine/integrations/nautilus_strategy.py:654-678`: Force-close microscopic remainders

**Status**: ✅ Fixed (verified 2→4 positions, but need signal generation to fully test)

**Prevention**:
- Always track original position size separately from current size
- Add assertion: `sum(all_exit_pcts) == 1.0` to catch incomplete exits
- Add cleanup for positions with `abs(quantity) < 1e-8`

---

### Issue #3: Config Not Being Read (2026-01-29)

**Symptom**: Exit logic uses hardcoded defaults (20%+20%+30%) instead of config values (30%+40%+30%)

**Root Cause**: Code looked for wrong config key
```python
# WRONG
exit_config = self.config.get('exit_rules', {})

# CORRECT
exit_config = self.config.get('archetypes', self.config.get('exit_rules', {}))
```

**Fix Applied**:
- `engine/archetypes/exit_logic.py:125`: Read from 'archetypes' key
- `engine/archetypes/exit_logic.py:433-441`: Parse `profit_targets` array

**Status**: ✅ Fixed (verified by debug output, awaiting integration test)

**Prevention**:
- Add schema validation for config files
- Add debug logging at config load time
- Create config unit tests

---

## Development Best Practices

### 1. Always Run Backtests After Changes

**Critical Paths That Affect Signal Generation**:
- `engine/archetypes/logic_v2_adapter.py`: Archetype signal logic
- `engine/archetypes/threshold_policy.py`: Dynamic thresholds
- `configs/*.json`: Threshold and parameter configs
- `engine/context/regime_service.py`: Regime classification

**Quick Validation Command**:
```bash
# Should generate ~5-10 signals on Q1 2023 data
python3 bin/nautilus_backtest_bull_machine.py \
  --config configs/nautilus_production_full_soul.json \
  --data data/btc_1h_2023_Q1.csv \
  | grep "Total signals"
```

**Baseline Expectations**:
- Q1 2023 (2,134 bars): ~5-20 signals
- Jan 2022 (721 bars): ~2-8 signals
- Full 2018-2024 (61,306 bars): ~5,000-10,000 signals

### 2. Config File Changes

**Always Validate**:
1. JSON syntax (use `jq . < config.json` to validate)
2. Required fields present
3. Percentages sum to 1.0
4. Thresholds in valid ranges

**Config Structure Reference**:
```json
{
  "exit_logic": {
    "archetypes": {
      "archetype_name": {
        "max_hold_hours": 48,
        "profit_targets": [
          {"r_multiple": 0.5, "exit_pct": 0.30},
          {"r_multiple": 1.0, "exit_pct": 0.40},
          {"r_multiple": 1.5, "exit_pct": 0.30}
        ]
      }
    }
  }
}
```

### 3. Git Workflow

**Before Making Changes**:
```bash
# Create a checkpoint
git add -A
git commit -m "checkpoint: working state before X"

# Make your changes

# Test
python3 bin/nautilus_backtest_bull_machine.py --config configs/test.json

# If broken, easy to revert
git reset --hard HEAD~1
```

**After Testing**:
```bash
# Only commit if tests pass
git add -A
git commit -m "feat: description of change"
```

### 4. Debug Logging

**Add Debug Output for Critical Paths**:
```python
# At config loading
print(f"[DEBUG] Config keys: {list(config.keys())}")
print(f"[DEBUG] Archetype config: {config.get('archetypes', {}).get('failed_rally', 'NOT FOUND')}")

# At decision points
logger.info(f"[DECISION] Signal {archetype} - fusion={fusion:.3f}, threshold={threshold:.3f}, pass={fusion >= threshold}")

# At exits
logger.info(f"[EXIT] Closing {exit_pct*100:.1f}% of original={original_qty:.6f}, current={current_qty:.6f}")
```

**Remove Debug Logging**:
- Use `grep -r "print(f\"\[DEBUG\]"` to find all debug prints before committing
- Convert to `logger.debug()` for production code

### 5. Common Gotchas

#### Float Comparison
```python
# WRONG - floating point precision issues
if position.quantity == 0:
    delete_position()

# CORRECT
if abs(position.quantity) < 1e-8:  # Essentially zero
    delete_position()
```

#### Dictionary.get() Default Values
```python
# WRONG - mutable default
def process(rules={}):  # Dangerous!

# CORRECT
def process(rules=None):
    if rules is None:
        rules = {}
```

#### Percentage Calculations
```python
# WRONG - compounds percentages
for pct in [0.2, 0.2, 0.3]:
    exit_qty = current_qty * pct  # Each exit reduces current_qty!
    current_qty -= exit_qty

# CORRECT - use original quantity
original_qty = position.quantity
for pct in [0.2, 0.2, 0.3]:
    exit_qty = original_qty * pct  # Always relative to original
    current_qty -= exit_qty
```

---

## Persistent Memory & Tracking Tools

### Built-in Python Libraries

#### 1. JSON for Simple Tracking
```python
import json
from pathlib import Path

# Save state
state = {
    "last_run": "2026-01-29",
    "known_issues": ["zero_signals", "zombie_positions"],
    "fixes_applied": {"exit_config": True, "original_quantity": True}
}
Path(".claude/session_state.json").write_text(json.dumps(state, indent=2))

# Load state
state = json.loads(Path(".claude/session_state.json").read_text())
```

#### 2. Shelve for Key-Value Persistence
```python
import shelve

# Save
with shelve.open('.claude/memory.db') as db:
    db['backtest_results'] = {"Q1_2023": {"signals": 0, "trades": 0}}
    db['known_bugs'] = ["zombie_position", "config_loading"]

# Load
with shelve.open('.claude/memory.db') as db:
    bugs = db.get('known_bugs', [])
```

#### 3. SQLite for Structured Tracking
```python
import sqlite3

conn = sqlite3.connect('.claude/tracking.db')
c = conn.cursor()

# Create table
c.execute('''
    CREATE TABLE IF NOT EXISTS issues (
        id INTEGER PRIMARY KEY,
        date TEXT,
        issue TEXT,
        status TEXT,
        fix_location TEXT
    )
''')

# Add issue
c.execute('''
    INSERT INTO issues VALUES (NULL, ?, ?, ?, ?)
''', ('2026-01-29', 'zero_signals', 'investigating', 'logic_v2_adapter.py'))

conn.commit()
```

#### 4. Joblib for ML Model Tracking
```python
from joblib import dump, load

# Save
metadata = {
    "model_path": "models/regime_v4.pkl",
    "features": [...],
    "performance": {"accuracy": 0.85}
}
dump(metadata, '.claude/model_metadata.joblib')

# Load
metadata = load('.claude/model_metadata.joblib')
```

### Recommended Approach for This Project

**Use `.claude/` Directory Structure**:
```
.claude/
├── session_state.json          # Current session state
├── known_issues.json           # Issue tracker
├── backtest_baseline.json      # Expected results for validation
└── config_history/             # Config snapshots
    ├── 2026-01-29_working.json
    └── 2026-01-29_broken.json
```

**Simple Tracking Script** (`.claude/track_issue.py`):
```python
#!/usr/bin/env python3
import json
from datetime import datetime
from pathlib import Path

ISSUES_FILE = Path(".claude/known_issues.json")

def load_issues():
    if ISSUES_FILE.exists():
        return json.loads(ISSUES_FILE.read_text())
    return []

def save_issues(issues):
    ISSUES_FILE.parent.mkdir(exist_ok=True)
    ISSUES_FILE.write_text(json.dumps(issues, indent=2))

def add_issue(name, description, location):
    issues = load_issues()
    issues.append({
        "id": len(issues) + 1,
        "date": datetime.now().isoformat(),
        "name": name,
        "description": description,
        "location": location,
        "status": "open"
    })
    save_issues(issues)
    print(f"✓ Issue #{len(issues)} added: {name}")

def resolve_issue(issue_id, fix_description):
    issues = load_issues()
    for issue in issues:
        if issue["id"] == issue_id:
            issue["status"] = "resolved"
            issue["fix"] = fix_description
            issue["resolved_date"] = datetime.now().isoformat()
    save_issues(issues)
    print(f"✓ Issue #{issue_id} resolved")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: track_issue.py add|resolve|list")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "list":
        for issue in load_issues():
            status_icon = "✅" if issue["status"] == "resolved" else "❌"
            print(f"{status_icon} #{issue['id']}: {issue['name']} ({issue['date']})")
```

**Usage**:
```bash
# Add issue
python .claude/track_issue.py add "zero_signals" "No signals generated" "logic_v2_adapter.py"

# List issues
python .claude/track_issue.py list

# Resolve issue
python .claude/track_issue.py resolve 1 "Softened ADX threshold from 25 to 15"
```

---

## External Tools (Optional)

### 1. MLflow (For ML Experiments)
```bash
pip install mlflow

# Track experiments
import mlflow
mlflow.start_run()
mlflow.log_param("fusion_threshold", 0.24)
mlflow.log_metric("total_signals", 5910)
mlflow.log_artifact("configs/config.json")
mlflow.end_run()
```

### 2. Weights & Biases (wandb)
```python
import wandb

wandb.init(project="bull-machine")
wandb.config.update({"threshold": 0.24})
wandb.log({"signals": 5910, "trades": 13})
```

### 3. DVC (Data Version Control)
```bash
# Track data and models
dvc add data/btc_1h_2018_2024.csv
dvc add models/regime_v4.pkl
git add data/.gitignore data/btc_1h_2018_2024.csv.dvc
```

---

## Quick Reference Card

### ❌ Symptoms → 🔍 Check Location

| Symptom | Check These Files |
|---------|------------------|
| 0 signals generated | `logic_v2_adapter.py` gate checks, config thresholds |
| Position never closes | `nautilus_strategy.py` exit calculations |
| Config not loaded | `exit_logic.py` config key names |
| Trades blocked | `nautilus_strategy.py` has_position check |
| Wrong exit percentages | `exit_logic.py` profit_targets parsing |
| Features missing | Feature store CSV, feature_provider.py |

### 🐛 Common Error Messages

| Error | Likely Cause | Fix Location |
|-------|--------------|--------------|
| `NOT FOUND in archetype` | Missing config parameter | Add to `configs/*.json` |
| `VETO: ADX too weak` | Threshold too high | Lower ADX threshold in config |
| `Pattern NOT matched` | Pattern logic too strict | Check pattern conditions in `logic_v2_adapter.py` |
| `blocked by existing position` | Zombie position | Check exit calculations in `nautilus_strategy.py` |

---

## Testing Checklist

Before committing code changes:

- [ ] Run quick backtest: `python3 bin/nautilus_backtest_bull_machine.py --data data/btc_1h_2023_Q1.csv`
- [ ] Check signals generated: Should be > 0
- [ ] Check trades executed: Should be > 0 if signals generated
- [ ] Check exit percentages: Should match config (grep for "[EXIT]" in logs)
- [ ] Check for errors: `grep -i "error\|exception\|traceback" in output`
- [ ] Validate config JSON: `jq . < configs/your_config.json`
- [ ] Check git diff: `git diff --stat` (should be reasonable, not 2,000+ lines)

---

**Last Updated**: 2026-01-29
**Maintainer**: Claude AI Session Tracking
**Location**: `/Users/raymondghandchi/Bull-machine-/Bull-machine-/CLAUDE.md`
