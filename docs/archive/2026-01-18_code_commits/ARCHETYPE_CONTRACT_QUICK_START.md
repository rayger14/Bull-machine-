# Archetype Contract System - Quick Start

**TL;DR:** Unified interface + registry + feature validation = No more ghost archetypes

---

## What Is This?

The **Archetype Contract System** provides:

1. **BaseArchetype** - Abstract base class all archetypes must implement
2. **ArchetypeRegistry** - Single YAML file defining all archetypes
3. **FeatureRealityGate** - Pre-backtest feature validation
4. **Clear Maturity States** - stub → development → calibrated → production

**Goal:** Eliminate ghost archetypes by enforcing standard interface and validation.

---

## File Locations

```
Bull-machine-/
├── archetype_registry.yaml                  # Registry of all archetypes
├── engine/
│   ├── archetypes/
│   │   ├── base_archetype.py                # Abstract base class
│   │   ├── registry_manager.py              # Registry loader
│   │   └── ...
│   ├── validation/
│   │   └── feature_reality_gate.py          # Feature validator
│   └── strategies/archetypes/
│       ├── bear/
│       │   ├── liquidity_vacuum_archetype.py  # S1 implementation
│       │   ├── funding_divergence_archetype.py # S4 implementation
│       │   └── ...
│       └── bull/
│           ├── spring_utad_archetype.py       # A implementation (stub)
│           └── ...
├── ARCHETYPE_CONTRACT_DESIGN.md             # Full specification
└── ARCHETYPE_CONTRACT_MIGRATION_GUIDE.md    # Migration instructions
```

---

## Quick Reference

### 1. Create New Archetype

```python
from engine.archetypes.base_archetype import (
    BaseArchetype, ArchetypeScore, ArchetypeVeto, ArchetypeEntry,
    MaturityLevel, SignalType
)

class MyArchetype(BaseArchetype):
    # 1. Set class attributes
    ARCHETYPE_ID = "X1"
    ARCHETYPE_NAME = "My Pattern"
    MATURITY = MaturityLevel.DEVELOPMENT
    DIRECTION = SignalType.LONG
    REGIME_TAGS = ["risk_on"]
    REQUIRES_ENGINES = ["smc", "liquidity"]

    # 2. Declare required features
    def required_features(self):
        return ['bos_detected', 'liquidity_score', 'volume_zscore']

    # 3. Implement scoring logic
    def score(self, context):
        row = context.row
        score = 0.0  # Calculate from features
        return ArchetypeScore(
            total_score=score,
            component_scores={'bos': 0.5, 'liq': 0.3},
            reasons=['BOS detected', 'High liquidity'],
            metadata={}
        )

    # 4. Implement veto logic
    def veto(self, context):
        if context.regime_label not in ['risk_on']:
            return ArchetypeVeto(
                is_vetoed=True,
                reason='Wrong regime',
                veto_type='regime_mismatch'
            )
        return ArchetypeVeto.no_veto()

    # 5. Implement entry logic
    def entry(self, context):
        return ArchetypeEntry(
            signal=SignalType.LONG,
            confidence=0.5,
            entry_price=None,
            metadata={'stop_loss_pct': -0.02, 'take_profit_pct': 0.05}
        )
```

### 2. Register Archetype

Add to `archetype_registry.yaml`:

```yaml
- id: X1
  name: "My Pattern"
  slug: "my_pattern"
  class: "engine.strategies.archetypes.bull.MyArchetype"
  maturity: development
  direction: long
  regime_tags: [risk_on]
  requires_engines: [smc, liquidity]
  requires_features:
    critical: [bos_detected, liquidity_score]
    recommended: [volume_zscore]
    optional: []
  enable_flag: enable_X1
```

### 3. Use in Backtest

```python
from engine.archetypes.registry_manager import get_registry
from engine.validation.feature_reality_gate import FeatureRealityGate

# Load registry
registry = get_registry()

# Get enabled archetypes
archetypes = registry.get_archetypes(
    maturity=['production', 'calibrated'],
    enabled_only=True,
    config=config
)

# Validate features
gate = FeatureRealityGate()
report = gate.validate_all(archetypes, df)

# Instantiate archetypes
instances = registry.instantiate_all(
    maturity=['production'],
    enabled_only=True,
    config=config
)

# Run backtest
for idx, row in df.iterrows():
    context = RuntimeContext(...)

    for arch_id, arch in instances.items():
        veto = arch.veto(context)
        if veto.is_vetoed:
            continue

        score = arch.score(context)
        if score.total_score < threshold:
            continue

        entry = arch.entry(context)
        # Execute trade...
```

---

## CLI Tools

### Inspect Registry

```bash
python -m engine.archetypes.registry_manager
```

Output:
```
================================================================================
ARCHETYPE REGISTRY STATUS
================================================================================
Total archetypes: 8
Deprecated: 5
Loaded classes: 6

  PRODUCTION     :  3  [S1, S4, S5]
  CALIBRATED     :  3  [H, B, K]
  DEVELOPMENT    :  0  []
  STUB           :  2  [A, C]
```

### Validate Features

```bash
python -m engine.validation.feature_reality_gate \
    --features data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet
```

Output:
```
================================================================================
FEATURE REALITY GATE REPORT
================================================================================
Total archetypes validated: 6
  ✓ Can run: 5
  ⚠ Degraded mode: 1
  ✗ Blocked: 0

Per-archetype coverage:
  ✓ S1    Liquidity Vacuum Reversal      100.0% (10/10 features)
  ✓ S4    Funding Divergence              92.3% (12/13 features)
      RECOMMENDED MISSING: shorts_liquidations
  ✓ S5    Long Squeeze Cascade           100.0% (11/11 features)
  ...
```

---

## Maturity Levels

| Level | Description | Validation | Usage |
|-------|-------------|------------|-------|
| **stub** | Placeholder, no code | Skip validation | Registry only |
| **development** | Partial implementation | Validate interface | Testing only |
| **calibrated** | Validated on history | Full validation | Backtest ready |
| **production** | Live-ready | Full validation + performance | Live trading |

---

## Common Patterns

### Pattern 1: Simple Archetype (No Runtime Features)

```python
class SimpleArchetype(BaseArchetype):
    def required_features(self):
        return ['feature1', 'feature2']

    def score(self, context):
        # Use existing features only
        return ArchetypeScore(...)
```

### Pattern 2: Archetype With Runtime Features

```python
class RuntimeArchetype(BaseArchetype):
    def __init__(self):
        super().__init__()
        from .my_runtime_features import MyRuntimeFeatures
        self.runtime_features = MyRuntimeFeatures()

    def enrich_features(self, df):
        """Called BEFORE backtest to add runtime features"""
        return self.runtime_features.enrich_dataframe(df)

    def required_features(self):
        # Return BASE features (runtime features computed in enrich)
        return ['base_feature1', 'base_feature2']

    def score(self, context):
        # Use both base + runtime features
        runtime_feat = context.row.get('my_runtime_feature')
        return ArchetypeScore(...)
```

### Pattern 3: Stub Archetype (Not Yet Implemented)

```python
from engine.archetypes.base_archetype import StubArchetype

class MyStubArchetype(StubArchetype):
    ARCHETYPE_ID = "Z"
    ARCHETYPE_NAME = "My Stub"
    MATURITY = MaturityLevel.STUB
    DIRECTION = SignalType.LONG
    REGIME_TAGS = ["risk_on"]
    REQUIRES_ENGINES = ["wyckoff"]

    # No methods needed - StubArchetype raises NotImplementedError
```

---

## Troubleshooting

### Problem: Archetype Not Loading

**Error:** `✗ X1    Failed to import: No module named '...'`

**Solution:**
- Check `class` path in `archetype_registry.yaml`
- Verify file exists at correct location
- Check for syntax errors in archetype file

### Problem: Missing Features

**Error:** `CRITICAL MISSING: liquidity_drain_pct`

**Solution:**
- Add feature to feature store, OR
- Mark as `recommended` instead of `critical`, OR
- Implement runtime feature enrichment

### Problem: Veto Always Blocking

**Error:** Archetype never takes trades

**Solution:**
- Check `veto()` logic - may be too restrictive
- Log veto reasons: `logger.info(f"Veto: {veto.reason}")`
- Verify regime labels match `REGIME_TAGS`

---

## Best Practices

1. **Start with stub:** Create StubArchetype first, promote to development later
2. **Test features first:** Validate required_features() with FeatureRealityGate before implementing
3. **Separate concerns:** score() = pattern recognition, veto() = safety, entry() = execution
4. **Document reasons:** Always populate `reasons` in ArchetypeScore for interpretability
5. **Use diagnostics():** Override for better post-trade analysis
6. **Version control:** Registry changes = code changes = git commits

---

## Next Steps

1. **Read full spec:** See `ARCHETYPE_CONTRACT_DESIGN.md`
2. **Migrate existing:** See `ARCHETYPE_CONTRACT_MIGRATION_GUIDE.md`
3. **Test:** Run CLI tools to validate setup
4. **Implement:** Start with stub, promote to development, calibrate, then production

---

**Questions?** See full documentation in `ARCHETYPE_CONTRACT_DESIGN.md`
