# Archetype System Documentation

## Overview
This directory contains documentation for the Bull Machine archetype detection and trading system. Archetypes are market pattern detection modules that identify specific trading setups using domain knowledge engines.

## Quick Start
**Start here**: [ARCHETYPE_ENGINE_QUICK_START.md](./ARCHETYPE_ENGINE_QUICK_START.md)

## Core Documentation

### System Architecture
- **[ARCHETYPE_ENGINE_DELIVERY_COMPLETE.md](./ARCHETYPE_ENGINE_DELIVERY_COMPLETE.md)** - Complete domain engine implementation report
- **[ARCHETYPE_ENGINE_DOCUMENTATION_INDEX.md](./ARCHETYPE_ENGINE_DOCUMENTATION_INDEX.md)** - Master documentation index
- **[ARCHETYPE_INVENTORY_AND_HEALTH_AUDIT.md](./ARCHETYPE_INVENTORY_AND_HEALTH_AUDIT.md)** - System health and inventory

### Implementation Status
- **[ARCHETYPE_ENGINE_FIX_COMPLETE.md](./ARCHETYPE_ENGINE_FIX_COMPLETE.md)** - Recent fixes and improvements
- **[ARCHETYPE_ENGINE_FIX_SUMMARY.md](./ARCHETYPE_ENGINE_FIX_SUMMARY.md)** - Fix summary and impact analysis

## Validation & Testing

### Validation Reports
Located in [validation/](./validation/)
- **[ARCHETYPE_VALIDATION_COMPLETE.md](./validation/ARCHETYPE_VALIDATION_COMPLETE.md)** - Comprehensive validation results
- **[ARCHETYPE_FIX_VALIDATION_REPORT.md](./validation/ARCHETYPE_FIX_VALIDATION_REPORT.md)** - Post-fix validation
- **[ARCHETYPE_KNOWLEDGE_VALIDATION_REPORT.md](./validation/ARCHETYPE_KNOWLEDGE_VALIDATION_REPORT.md)** - Domain knowledge validation
- **[ARCHETYPE_NATIVE_VALIDATION_REPORT.md](./validation/ARCHETYPE_NATIVE_VALIDATION_REPORT.md)** - Native implementation validation

## Archetypes Catalog

### Bull Market Archetypes (S1-S4)
1. **S1 - Liquidity Vacuum Reversal**
   - Pattern: Sharp drop followed by strong reversal at demand
   - Domain engines: Wyckoff (Spring A/B), SMC (BOS, demand zones), Temporal
   - Features: 37 domain features

2. **S4 - Funding Divergence Long**
   - Pattern: Negative funding + accumulation = reversal
   - Domain engines: Wyckoff (accumulation), SMC (BOS), Temporal
   - Features: 33 domain features

3. **S2 - Failed Rally Short** (Bear Archetype)
   - Pattern: Weak rally in downtrend fails
   - Domain engines: Wyckoff (UTAD), SMC (supply), Temporal

4. **S3 - BOS/CHOCH Long**
   - Pattern: Market structure break confirming trend change
   - Domain engines: SMC (BOS/CHOCH), Wyckoff

### Bear Market Archetypes (S5+)
1. **S5 - Long Squeeze Short**
   - Pattern: Overextended longs get liquidated
   - Domain engines: Wyckoff (distribution), SMC (supply), HOB
   - Features: 35 domain features

## Domain Engines

### Integrated Engines (Dec 2024)
1. **Wyckoff Events** (13 events)
   - SC, BC, Spring A/B, UTAD, LPS, LPSY, etc.
   - Boost: 1.5x - 2.5x per event

2. **SMC (Smart Money Concepts)** (4 features)
   - BOS (Break of Structure), CHOCH
   - Supply/Demand Zones, Liquidity Sweeps
   - Boost: 1.4x - 2.0x

3. **HOB (Hidden Order Book)** (3 features)
   - Demand/Supply zones from order flow
   - Bid/Ask imbalance detection
   - Boost: 1.3x - 1.5x

4. **Temporal Fusion** (4 features)
   - Fibonacci time clusters
   - Multi-timeframe confluence
   - Pattern resistance/support clusters
   - Boost: 1.5x - 1.8x

### Feature Flag Control
All domain engines are controlled by feature flags in config:
```json
{
  "enable_wyckoff": true,
  "enable_smc": true,
  "enable_hob": true,
  "enable_temporal": true,
  "enable_macro": false
}
```

## Code Structure

### Core Implementation
- **engine/archetypes/logic_v2_adapter.py** - Domain engine implementation
- **engine/archetypes/threshold_policy.py** - Threshold and boost logic
- **engine/features/registry.py** - Feature definitions
- **engine/wyckoff/wyckoff_engine.py** - Wyckoff event detection

### Testing
- **bin/test_domain_wiring.py** - Core vs Full variant comparison
- **bin/test_archetype_model.py** - Archetype model testing
- **bin/test_archetype_wrapper_fix.py** - Wrapper layer validation

## Performance Metrics

### Domain Engine Impact
- **Max Theoretical Boost**: 95x (full confluence)
- **Realistic Boost**: 8x - 12x
- **Veto Protection**: 15+ hard/soft vetoes prevent bad entries
- **Feature Coverage**: 44 unique domain features

### Validation Results
See [validation/](./validation/) for detailed reports.

## Historical Archive

### Archived Documentation
Located in [archive/](./archive/)
- Historical audit reports
- Optimization roadmaps
- Legacy diagnostic reports

Archived docs are kept for historical reference but may be outdated.

## Related Documentation

### Domain Engine Documentation
See [../domain_engine/](../domain_engine/) for:
- Domain engine wiring details
- Feature generation pipeline
- Domain feature availability reports
- Critical bug fix documentation

### Technical Documentation
See [../technical/](../technical/) for:
- Implementation specifications
- Architecture diagrams
- Integration guides

## Quick Links

- [Main README](../../README.md)
- [Domain Engine Guide](../domain_engine/DOMAIN_ENGINE_GUIDE.md)
- [Feature Store Schema](../FEATURE_STORE_SCHEMA_v2.md)
- [Deployment Guide](../DEPLOYMENT_GUIDE_INDEX.md)

## Contributing

When adding new archetypes:
1. Implement detection logic in `engine/archetypes/logic_v2_adapter.py`
2. Add feature definitions to `engine/features/registry.py`
3. Create validation tests in `bin/test_*.py`
4. Document in this directory
5. Update this README

## Status

**Last Updated**: 2025-12-11
**System Version**: v2.5 (Domain Engines Complete)
**Production Ready**: Yes
**Feature Flag Status**: All engines operational

---

For questions or issues, see the main project README or create a GitHub issue.
