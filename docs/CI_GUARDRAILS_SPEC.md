# CI/CD Guardrails Specification

**Version:** 2.0.0
**Date:** 2025-11-19
**Status:** Design Complete - Ready for Implementation
**Purpose:** Define automated CI/CD guardrails to prevent regressions during Ghost → Live v2 upgrade

---

## Executive Summary

This document specifies 4 critical CI/CD guardrails that will run automatically on every commit to prevent code-feature inconsistencies, dead features, config drift, and performance regressions.

**Guardrails:**
1. **Blueprint vs Code Consistency Test** - Ensures code matches feature inventory
2. **Dead Feature Detector** - Finds features in store but never used in code
3. **Config Consistency Test** - Validates configs match current schema
4. **Backtest Expectations Test** - Ensures gold standard metrics hold

**Trigger:** Every commit, pull request, and merge to integration branch
**Enforcement:** Blocking (CI fails if guardrails fail)

---

## 1. Guardrail #1: Blueprint vs Code Consistency Test

### 1.1 Purpose

Ensure the brain blueprint (module inventory) matches the actual code implementation. Prevents documentation drift.

### 1.2 Implementation

**Script:** `bin/validate_blueprint_consistency.py`

```python
#!/usr/bin/env python3
"""
Validate Brain Blueprint Snapshot vs Actual Code

Ensures:
- All modules listed in blueprint exist in codebase
- All modules in codebase are documented in blueprint
- Module status (LIVE/PARTIAL/IDEA) matches implementation
"""

import os
import json
from pathlib import Path
from typing import Dict, Set, Tuple

BLUEPRINT_PATH = "docs/BRAIN_BLUEPRINT_SNAPSHOT_v2.md"
ENGINE_PATH = "engine/"

def parse_blueprint(blueprint_path: str) -> Dict[str, str]:
    """Parse blueprint markdown to extract module inventory"""
    modules = {}
    current_status = None

    with open(blueprint_path) as f:
        for line in f:
            # Detect status sections
            if "### 1.1 LIVE Modules" in line:
                current_status = "LIVE"
            elif "### 1.2 PARTIAL Modules" in line:
                current_status = "PARTIAL"
            elif "### 1.3 IDEA ONLY Modules" in line:
                current_status = "IDEA"

            # Extract module paths (format: "1. `engine/path/module.py` - Description")
            if current_status and ". `engine/" in line:
                # Extract path between backticks
                start = line.find("`") + 1
                end = line.find("`", start)
                module_path = line[start:end]
                modules[module_path] = current_status

    return modules

def scan_codebase(engine_path: str) -> Set[str]:
    """Scan engine/ directory for all Python modules"""
    modules = set()
    for root, dirs, files in os.walk(engine_path):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != '__pycache__']

        for file in files:
            if file.endswith('.py'):
                rel_path = os.path.relpath(os.path.join(root, file))
                modules.add(rel_path)

    return modules

def check_module_status(module_path: str) -> str:
    """Infer module status from code quality"""
    if not os.path.exists(module_path):
        return "IDEA"

    with open(module_path) as f:
        code = f.read()

    # IDEA: mostly comments, no tests, lots of TODOs
    if code.count("TODO") > 5 or code.count("FIXME") > 3:
        return "IDEA"

    # PARTIAL: some implementation, but incomplete
    if code.count("NotImplementedError") > 0:
        return "PARTIAL"
    if code.count("raise NotImplementedError") > 0:
        return "PARTIAL"
    if code.count("# TODO:") > 2:
        return "PARTIAL"

    # Check for tests
    test_file = module_path.replace("engine/", "tests/unit/test_")
    test_file = test_file.replace(".py", ".py")
    if not os.path.exists(test_file):
        return "PARTIAL"

    # LIVE: implementation complete, tests present
    return "LIVE"

def main():
    print("📋 Validating Brain Blueprint Consistency...\n")

    # Parse blueprint
    blueprint_modules = parse_blueprint(BLUEPRINT_PATH)
    print(f"✓ Parsed blueprint: {len(blueprint_modules)} modules documented")

    # Scan codebase
    codebase_modules = scan_codebase(ENGINE_PATH)
    print(f"✓ Scanned codebase: {len(codebase_modules)} modules found")

    # Check for missing modules in blueprint
    undocumented_modules = codebase_modules - set(blueprint_modules.keys())
    if undocumented_modules:
        print(f"\n❌ UNDOCUMENTED MODULES ({len(undocumented_modules)}):")
        for module in sorted(undocumented_modules):
            print(f"   - {module}")
        print("\n   Action: Add these modules to BRAIN_BLUEPRINT_SNAPSHOT_v2.md")

    # Check for modules in blueprint but not in codebase
    missing_modules = set(blueprint_modules.keys()) - codebase_modules
    if missing_modules:
        print(f"\n❌ MISSING MODULES ({len(missing_modules)}):")
        for module in sorted(missing_modules):
            status = blueprint_modules[module]
            print(f"   - {module} (documented as {status})")
        print("\n   Action: Implement these modules or remove from blueprint")

    # Check module status consistency
    status_mismatches = []
    for module, documented_status in blueprint_modules.items():
        if module in codebase_modules:
            actual_status = check_module_status(module)
            if actual_status != documented_status:
                status_mismatches.append((module, documented_status, actual_status))

    if status_mismatches:
        print(f"\n⚠ STATUS MISMATCHES ({len(status_mismatches)}):")
        for module, documented, actual in status_mismatches:
            print(f"   - {module}:")
            print(f"       Documented: {documented}")
            print(f"       Actual: {actual}")
        print("\n   Action: Update blueprint or fix module implementation")

    # Summary
    print("\n" + "="*60)
    if undocumented_modules or missing_modules or status_mismatches:
        print("❌ BLUEPRINT CONSISTENCY: FAILED")
        print(f"   - Undocumented modules: {len(undocumented_modules)}")
        print(f"   - Missing modules: {len(missing_modules)}")
        print(f"   - Status mismatches: {len(status_mismatches)}")
        return 1
    else:
        print("✓ BLUEPRINT CONSISTENCY: PASSED")
        return 0

if __name__ == "__main__":
    exit(main())
```

### 1.3 CI Integration

**GitHub Actions Workflow:**

```yaml
# .github/workflows/blueprint-consistency.yml
name: Blueprint Consistency

on: [push, pull_request]

jobs:
  validate-blueprint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Validate Blueprint Consistency
        run: python bin/validate_blueprint_consistency.py
```

### 1.4 Expected Output

**Success:**
```
📋 Validating Brain Blueprint Consistency...

✓ Parsed blueprint: 89 modules documented
✓ Scanned codebase: 89 modules found

============================================================
✓ BLUEPRINT CONSISTENCY: PASSED
```

**Failure:**
```
📋 Validating Brain Blueprint Consistency...

✓ Parsed blueprint: 89 modules documented
✓ Scanned codebase: 92 modules found

❌ UNDOCUMENTED MODULES (3):
   - engine/features/new_feature.py
   - engine/psychology/advanced_sentiment.py
   - engine/ml/deep_learning.py

   Action: Add these modules to BRAIN_BLUEPRINT_SNAPSHOT_v2.md

============================================================
❌ BLUEPRINT CONSISTENCY: FAILED
   - Undocumented modules: 3
   - Missing modules: 0
   - Status mismatches: 0
```

---

## 2. Guardrail #2: Dead Feature Detector

### 2.1 Purpose

Find features in the feature store that are never referenced in code. Prevents feature store bloat and identifies unused calculations.

### 2.2 Implementation

**Script:** `bin/detect_dead_features.py`

```python
#!/usr/bin/env python3
"""
Dead Feature Detector

Finds feature store columns that are never used in code.

Checks:
- Feature store schema (all 116+ columns)
- All Python files in engine/ (grep for column references)
- Reports unused features
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import Set, Dict

FEATURE_STORE_PATH = "data/features_mtf/BTC_1H_2022-2024.parquet"
ENGINE_PATH = "engine/"
DOCS_PATH = "docs/"

def get_feature_columns(parquet_path: str) -> Set[str]:
    """Extract all column names from feature store"""
    df = pd.read_parquet(parquet_path)
    return set(df.columns)

def scan_code_for_references(search_path: str, feature_columns: Set[str]) -> Dict[str, Set[str]]:
    """Scan all Python files for feature column references"""
    references = {col: set() for col in feature_columns}

    for root, dirs, files in os.walk(search_path):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != '__pycache__']

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path) as f:
                    content = f.read()

                # Check for each feature column
                for col in feature_columns:
                    # Match patterns:
                    # - df['column_name']
                    # - df["column_name"]
                    # - row['column_name']
                    # - row["column_name"]
                    # - 'column_name' in df.columns
                    patterns = [
                        rf"['\"]{ re.escape(col)}['\"]",
                        rf"\b{re.escape(col)}\b",  # Direct reference
                    ]

                    for pattern in patterns:
                        if re.search(pattern, content):
                            references[col].add(file_path)
                            break

    return references

def main():
    print("🔍 Detecting Dead Features...\n")

    # Load feature store
    if not os.path.exists(FEATURE_STORE_PATH):
        print(f"❌ Feature store not found: {FEATURE_STORE_PATH}")
        return 1

    feature_columns = get_feature_columns(FEATURE_STORE_PATH)
    print(f"✓ Loaded feature store: {len(feature_columns)} columns")

    # Scan code for references
    references = scan_code_for_references(ENGINE_PATH, feature_columns)
    print(f"✓ Scanned codebase: {len([f for f in Path(ENGINE_PATH).rglob('*.py')])} files")

    # Find dead features (no references)
    dead_features = [col for col, refs in references.items() if len(refs) == 0]
    dead_features.sort()

    # Report
    if dead_features:
        print(f"\n⚠ DEAD FEATURES DETECTED ({len(dead_features)}):")
        print("\nThese columns exist in feature store but are never used in code:\n")
        for col in dead_features:
            print(f"   - {col}")

        print("\nRecommended Actions:")
        print("1. Remove dead features from feature store (if truly unused)")
        print("2. Add code to use these features (if planned)")
        print("3. Document as 'reserved for future use' (if experimental)")

        # Categorize by prefix
        print("\n📊 Dead Features by Category:")
        categories = {}
        for col in dead_features:
            prefix = col.split('_')[0]
            if prefix not in categories:
                categories[prefix] = []
            categories[prefix].append(col)

        for prefix, cols in sorted(categories.items()):
            print(f"\n   {prefix.upper()}: {len(cols)} features")
            for col in cols[:5]:  # Show first 5
                print(f"     - {col}")
            if len(cols) > 5:
                print(f"     ... and {len(cols) - 5} more")

    # Summary
    print("\n" + "="*60)
    usage_rate = (len(feature_columns) - len(dead_features)) / len(feature_columns) * 100
    print(f"✓ Feature Usage Rate: {usage_rate:.1f}% ({len(feature_columns) - len(dead_features)}/{len(feature_columns)})")

    if len(dead_features) > 10:  # Allow up to 10 dead features (experimental)
        print(f"❌ DEAD FEATURE CHECK: FAILED ({len(dead_features)} dead features)")
        return 1
    elif len(dead_features) > 0:
        print(f"⚠ DEAD FEATURE CHECK: WARNING ({len(dead_features)} dead features)")
        return 0  # Warning, but don't fail CI
    else:
        print("✓ DEAD FEATURE CHECK: PASSED (all features used)")
        return 0

if __name__ == "__main__":
    exit(main())
```

### 2.3 CI Integration

```yaml
# .github/workflows/dead-features.yml
name: Dead Feature Detection

on: [push, pull_request]

jobs:
  detect-dead-features:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install pandas pyarrow
      - name: Detect Dead Features
        run: python bin/detect_dead_features.py
```

### 2.4 Expected Output

**Success:**
```
🔍 Detecting Dead Features...

✓ Loaded feature store: 116 columns
✓ Scanned codebase: 112 files

============================================================
✓ Feature Usage Rate: 100.0% (116/116)
✓ DEAD FEATURE CHECK: PASSED (all features used)
```

**Warning (acceptable):**
```
🔍 Detecting Dead Features...

✓ Loaded feature store: 116 columns
✓ Scanned codebase: 112 files

⚠ DEAD FEATURES DETECTED (3):

These columns exist in feature store but are never used in code:

   - experimental_feature_1
   - experimental_feature_2
   - reserved_for_future

Recommended Actions:
1. Remove dead features from feature store (if truly unused)
2. Add code to use these features (if planned)
3. Document as 'reserved for future use' (if experimental)

============================================================
✓ Feature Usage Rate: 97.4% (113/116)
⚠ DEAD FEATURE CHECK: WARNING (3 dead features)
```

---

## 3. Guardrail #3: Config Consistency Test

### 3.1 Purpose

Validate that all config files are consistent with the current feature store schema and module inventory. Prevents config drift.

### 3.2 Implementation

**Script:** `bin/validate_config_consistency.py`

```python
#!/usr/bin/env python3
"""
Config Consistency Validator

Validates:
- All configs reference valid feature store columns
- All configs reference valid modules (archetypes, thresholds, etc.)
- No deprecated parameters in configs
- Required parameters present
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set

FEATURE_STORE_PATH = "data/features_mtf/BTC_1H_2022-2024.parquet"
CONFIG_DIR = "configs/"
SCHEMA_PATH = "docs/FEATURE_STORE_SCHEMA_v2.md"

VALID_ARCHETYPES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'S1', 'S2', 'S3', 'S4', 'S5', 'S8']
DEPRECATED_PARAMS = ['old_fusion_weights', 'legacy_threshold', 'deprecated_gate']

def get_feature_columns(parquet_path: str) -> Set[str]:
    """Get all valid feature columns"""
    df = pd.read_parquet(parquet_path)
    return set(df.columns)

def validate_config(config_path: str, valid_features: Set[str]) -> List[str]:
    """Validate a single config file"""
    errors = []

    with open(config_path) as f:
        config = json.load(f)

    # Check for deprecated parameters
    for deprecated in DEPRECATED_PARAMS:
        if deprecated in json.dumps(config):
            errors.append(f"Deprecated parameter: {deprecated}")

    # Check archetype references
    if 'archetypes' in config:
        for key in config['archetypes']:
            if key.startswith('enable_'):
                archetype = key.replace('enable_', '')
                if archetype not in VALID_ARCHETYPES:
                    errors.append(f"Invalid archetype: {archetype}")

    # Check feature references (in thresholds, weights, etc.)
    config_str = json.dumps(config)
    for potential_feature in re.findall(r'["\'](\w+_\w+)["\']', config_str):
        # Check if it looks like a feature name (contains underscore)
        if '_' in potential_feature and potential_feature not in valid_features:
            # Could be a valid parameter name, skip common ones
            if potential_feature not in ['enable_A', 'enable_S2', 'risk_on', 'risk_off']:
                errors.append(f"Unknown feature reference: {potential_feature}")

    # Check required parameters
    required_params = ['archetypes', 'fusion', 'exits', 'risk']
    for param in required_params:
        if param not in config:
            errors.append(f"Missing required parameter: {param}")

    return errors

def main():
    print("🔧 Validating Config Consistency...\n")

    # Load feature store schema
    if not os.path.exists(FEATURE_STORE_PATH):
        print(f"⚠ Feature store not found: {FEATURE_STORE_PATH}")
        print("   Skipping feature validation")
        valid_features = set()
    else:
        valid_features = get_feature_columns(FEATURE_STORE_PATH)
        print(f"✓ Loaded feature schema: {len(valid_features)} valid features")

    # Find all config files
    config_files = list(Path(CONFIG_DIR).rglob('*.json'))
    print(f"✓ Found {len(config_files)} config files")

    # Validate each config
    all_errors = {}
    for config_file in config_files:
        errors = validate_config(str(config_file), valid_features)
        if errors:
            all_errors[str(config_file)] = errors

    # Report
    if all_errors:
        print(f"\n❌ CONFIG VALIDATION FAILED ({len(all_errors)} configs with errors):\n")
        for config_file, errors in all_errors.items():
            print(f"   {config_file}:")
            for error in errors:
                print(f"      - {error}")
            print()

        print("Recommended Actions:")
        print("1. Remove deprecated parameters")
        print("2. Fix invalid archetype/feature references")
        print("3. Add missing required parameters")

    # Summary
    print("\n" + "="*60)
    valid_configs = len(config_files) - len(all_errors)
    print(f"✓ Valid Configs: {valid_configs}/{len(config_files)}")

    if all_errors:
        print("❌ CONFIG CONSISTENCY: FAILED")
        return 1
    else:
        print("✓ CONFIG CONSISTENCY: PASSED")
        return 0

if __name__ == "__main__":
    import re
    exit(main())
```

### 3.3 CI Integration

```yaml
# .github/workflows/config-consistency.yml
name: Config Consistency

on: [push, pull_request]

jobs:
  validate-configs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install pandas pyarrow
      - name: Validate Config Consistency
        run: python bin/validate_config_consistency.py
```

---

## 4. Guardrail #4: Backtest Expectations Test

### 4.1 Purpose

Ensure gold standard backtest metrics remain within acceptable bounds. Prevents performance regressions.

### 4.2 Implementation

**Script:** `tests/integration/test_gold_standard.py`

```python
#!/usr/bin/env python3
"""
Gold Standard Backtest Expectations Test

Validates:
- Profit Factor within ±5% of baseline (1.10 - 1.22)
- Trade Count within ±10% of baseline (297 - 363)
- Max Drawdown within ±10% of baseline (3.96% - 4.84%)
- Win Rate within ±5% of baseline (62.5% - 69.1%)
"""

import pytest
import subprocess
import json

BASELINE_METRICS = {
    'profit_factor': 1.16,
    'trade_count': 330,
    'max_drawdown': 4.4,
    'win_rate': 65.8,
}

TOLERANCES = {
    'profit_factor': 0.05,  # ±5%
    'trade_count': 0.10,  # ±10%
    'max_drawdown': 0.10,  # ±10%
    'win_rate': 0.05,  # ±5%
}

def run_backtest():
    """Run gold standard backtest"""
    cmd = [
        'python', 'bin/backtest_knowledge_v2.py',
        '--asset', 'BTC',
        '--start', '2024-01-01',
        '--end', '2024-09-30',
        '--config', 'configs/frozen/btc_1h_v2_baseline.json',
        '--output', 'results/gold_standard_latest.json'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"Backtest failed: {result.stderr}")

    # Load results
    with open('results/gold_standard_latest.json') as f:
        return json.load(f)

def test_profit_factor():
    """Test profit factor within bounds"""
    results = run_backtest()
    pf = results['profit_factor']
    baseline = BASELINE_METRICS['profit_factor']
    tolerance = TOLERANCES['profit_factor']

    lower = baseline * (1 - tolerance)
    upper = baseline * (1 + tolerance)

    assert lower <= pf <= upper, \
        f"Profit Factor {pf:.2f} outside range [{lower:.2f}, {upper:.2f}]"

def test_trade_count():
    """Test trade count within bounds"""
    results = run_backtest()
    count = results['trade_count']
    baseline = BASELINE_METRICS['trade_count']
    tolerance = TOLERANCES['trade_count']

    lower = int(baseline * (1 - tolerance))
    upper = int(baseline * (1 + tolerance))

    assert lower <= count <= upper, \
        f"Trade Count {count} outside range [{lower}, {upper}]"

def test_max_drawdown():
    """Test max drawdown within bounds"""
    results = run_backtest()
    dd = results['max_drawdown']
    baseline = BASELINE_METRICS['max_drawdown']
    tolerance = TOLERANCES['max_drawdown']

    lower = baseline * (1 - tolerance)
    upper = baseline * (1 + tolerance)

    assert lower <= dd <= upper, \
        f"Max Drawdown {dd:.2f}% outside range [{lower:.2f}%, {upper:.2f}%]"

def test_win_rate():
    """Test win rate within bounds"""
    results = run_backtest()
    wr = results['win_rate']
    baseline = BASELINE_METRICS['win_rate']
    tolerance = TOLERANCES['win_rate']

    lower = baseline * (1 - tolerance)
    upper = baseline * (1 + tolerance)

    assert lower <= wr <= upper, \
        f"Win Rate {wr:.1f}% outside range [{lower:.1f}%, {upper:.1f}%]"

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
```

### 4.3 CI Integration

```yaml
# .github/workflows/gold-standard.yml
name: Gold Standard Validation

on: [push, pull_request]

jobs:
  gold-standard:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      - name: Download feature store (if needed)
        run: |
          # If feature store not in repo (gitignored), download from artifact storage
          # For now, assume feature store is in repo
          echo "Feature store present"
      - name: Run Gold Standard Backtest
        run: pytest tests/integration/test_gold_standard.py -v
```

### 4.4 Expected Output

**Success:**
```
tests/integration/test_gold_standard.py::test_profit_factor PASSED
tests/integration/test_gold_standard.py::test_trade_count PASSED
tests/integration/test_gold_standard.py::test_max_drawdown PASSED
tests/integration/test_gold_standard.py::test_win_rate PASSED

==================== 4 passed in 45.23s ====================

✓ GOLD STANDARD VALIDATION: PASSED
```

**Failure:**
```
tests/integration/test_gold_standard.py::test_profit_factor FAILED
tests/integration/test_gold_standard.py::test_trade_count PASSED
tests/integration/test_gold_standard.py::test_max_drawdown PASSED
tests/integration/test_gold_standard.py::test_win_rate PASSED

==================== 1 failed, 3 passed in 45.23s ====================

FAILED tests/integration/test_gold_standard.py::test_profit_factor
AssertionError: Profit Factor 1.05 outside range [1.10, 1.22]

❌ GOLD STANDARD VALIDATION: FAILED
```

---

## 5. Pre-Commit Hook Integration

### 5.1 Local Pre-Commit Hook

**File:** `.git/hooks/pre-commit`

```bash
#!/bin/bash
set -e

echo "🔒 Running Pre-Commit Guardrails..."

# Guardrail 1: Blueprint Consistency
echo "\n1️⃣ Checking blueprint consistency..."
python bin/validate_blueprint_consistency.py || exit 1

# Guardrail 2: Dead Features
echo "\n2️⃣ Checking for dead features..."
python bin/detect_dead_features.py || exit 1

# Guardrail 3: Config Consistency
echo "\n3️⃣ Validating config consistency..."
python bin/validate_config_consistency.py || exit 1

# Guardrail 4: Unit Tests (skip gold standard in pre-commit, too slow)
echo "\n4️⃣ Running unit tests..."
pytest tests/unit/ --maxfail=1 -q || exit 1

echo "\n✓ All pre-commit guardrails passed!"
```

**Installation:**
```bash
# Make hook executable
chmod +x .git/hooks/pre-commit
```

---

## 6. Full CI/CD Pipeline

### 6.1 Complete Workflow

**File:** `.github/workflows/ci.yml`

```yaml
name: CI Pipeline

on:
  push:
    branches: [main, feature/*, release/*]
  pull_request:
    branches: [main]

jobs:
  # Job 1: Linting & Formatting
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install flake8 black mypy
      - name: Lint with flake8
        run: flake8 engine/ tests/ --max-line-length=100
      - name: Check formatting with black
        run: black --check engine/ tests/
      - name: Type check with mypy
        run: mypy engine/ --ignore-missing-imports

  # Job 2: Guardrail #1 - Blueprint Consistency
  blueprint-consistency:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Validate Blueprint Consistency
        run: python bin/validate_blueprint_consistency.py

  # Job 3: Guardrail #2 - Dead Features
  dead-features:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install pandas pyarrow
      - name: Detect Dead Features
        run: python bin/detect_dead_features.py

  # Job 4: Guardrail #3 - Config Consistency
  config-consistency:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install pandas pyarrow
      - name: Validate Config Consistency
        run: python bin/validate_config_consistency.py

  # Job 5: Unit Tests
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=engine --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2

  # Job 6: Integration Tests
  integration-tests:
    runs-on: ubuntu-latest
    needs: [unit-tests]
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run integration tests
        run: pytest tests/integration/ -v

  # Job 7: Guardrail #4 - Gold Standard Validation
  gold-standard:
    runs-on: ubuntu-latest
    needs: [integration-tests]
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run Gold Standard Validation
        run: pytest tests/integration/test_gold_standard.py -v

  # Job 8: Summary
  summary:
    runs-on: ubuntu-latest
    needs: [lint, blueprint-consistency, dead-features, config-consistency, unit-tests, integration-tests, gold-standard]
    steps:
      - name: All checks passed
        run: |
          echo "✓ All CI guardrails passed!"
          echo "✓ Linting & Formatting: PASSED"
          echo "✓ Blueprint Consistency: PASSED"
          echo "✓ Dead Features: PASSED"
          echo "✓ Config Consistency: PASSED"
          echo "✓ Unit Tests: PASSED"
          echo "✓ Integration Tests: PASSED"
          echo "✓ Gold Standard: PASSED"
```

---

## 7. Enforcement Policy

### 7.1 Blocking vs Warning

| Guardrail | Severity | Enforcement |
|-----------|----------|-------------|
| Blueprint Consistency | ERROR | BLOCKING (CI fails) |
| Dead Features (>10) | ERROR | BLOCKING (CI fails) |
| Dead Features (≤10) | WARNING | NON-BLOCKING (CI passes with warning) |
| Config Consistency | ERROR | BLOCKING (CI fails) |
| Gold Standard (PF out of range) | ERROR | BLOCKING (CI fails) |
| Unit Tests | ERROR | BLOCKING (CI fails) |

### 7.2 Override Mechanism

**For Emergencies Only:**

```bash
# Skip guardrails (emergency bypass)
git commit --no-verify -m "EMERGENCY: bypass guardrails"

# Document in commit message why guardrails were bypassed
```

**Requires:**
- Explicit approval from System Architect
- Documented in ROLLBACK_LOG.md
- Follow-up PR to fix guardrail failures

---

## 8. Monitoring & Reporting

### 8.1 Guardrail Metrics

**Track Over Time:**
- Blueprint consistency rate (% modules documented)
- Dead feature rate (% unused features)
- Config consistency rate (% valid configs)
- Gold standard pass rate (% commits passing)

**Dashboard:**
```
Guardrail Health (Last 30 days)

Blueprint Consistency:  98.5% ████████████████████ (197/200 commits)
Dead Features:          91.0% ██████████████████░░ (182/200 commits)
Config Consistency:     95.5% ███████████████████░ (191/200 commits)
Gold Standard:          87.0% █████████████████░░░ (174/200 commits)

Overall CI Pass Rate:   83.5% ████████████████░░░░ (167/200 commits)
```

### 8.2 Alerts

**Slack Integration:**
```yaml
# .github/workflows/ci.yml (add to summary job)
- name: Notify on failure
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK }}
    payload: |
      {
        "text": "🚨 CI Guardrails Failed",
        "blocks": [
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "*CI Guardrails Failed* ❌\n\nBranch: `${{ github.ref }}`\nCommit: `${{ github.sha }}`"
            }
          }
        ]
      }
```

---

## 9. References

- **Dev Workflow:** `docs/DEV_WORKFLOW.md`
- **Architecture:** `docs/GHOST_TO_LIVE_ARCHITECTURE.md`
- **Risk Mitigation:** `docs/UPGRADE_RISKS_AND_ROLLBACK.md`
- **Feature Schema:** `docs/FEATURE_STORE_SCHEMA_v2.md`

---

## Appendix A: Guardrail Implementation Timeline

| Phase | Guardrails Enabled | Timeline |
|-------|-------------------|----------|
| Phase 0 | None (manual validation only) | Current |
| Phase 1 | Guardrails #1, #2 (Blueprint, Dead Features) | Week 1 |
| Phase 2 | Guardrails #1, #2, #3 (+ Config Consistency) | Week 2 |
| Phase 3 | All 4 guardrails (+ Gold Standard) | Week 3 |
| Phase 4 | All 4 guardrails (enforced on main) | Week 4 |

---

## Version History

- **v2.0.0** (2025-11-19): Complete CI/CD guardrails specification
- **v1.0.0** (2025-11-14): Initial guardrails design
