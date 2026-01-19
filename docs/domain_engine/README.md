# Domain Engine Documentation

This directory contains comprehensive documentation for the Domain Engine implementation, which integrates Wyckoff, SMC, HOB, Temporal, and Macro analysis engines into the Bull Machine trading system.

## Quick Start

**New to Domain Engines?** Start here:
1. `DOMAIN_ENGINE_GUIDE.md` - Operator manual and overview
2. `DOMAIN_ENGINE_WIRING_QUICK_REFERENCE.md` - Quick reference card
3. `COMPLETE_ENGINE_WIRING_REPORT.md` - Complete feature map

## Document Index

### Core Documentation

**DOMAIN_ENGINE_GUIDE.md**
- Purpose: Comprehensive operator manual
- Audience: System operators, developers
- Contents: Architecture, features, usage, troubleshooting
- When to read: First time working with domain engines

**COMPLETE_ENGINE_WIRING_REPORT.md**
- Purpose: Complete feature map with boost values
- Audience: Technical developers, optimization engineers
- Contents: All 44 features wired across S1, S4, S5 archetypes
- When to read: Understanding feature integration details

**DOMAIN_ENGINE_WIRING_COMPLETE.md**
- Purpose: Implementation completion report
- Audience: Project managers, technical leads
- Contents: What was built, how it works, status
- When to read: Project status and completion verification

### Quick References

**DOMAIN_ENGINE_WIRING_QUICK_REFERENCE.md**
- Purpose: One-page quick reference
- Audience: All users
- Contents: Key features, boost values, feature flags
- When to read: Quick lookup during development

**DOMAIN_FEATURE_QUICK_REF.md**
- Purpose: Feature-specific quick reference
- Audience: Feature engineers
- Contents: Feature names, specifications, usage
- When to read: Working with specific features

### Implementation Details

**COMPLETE_FEATURE_GENERATION_REPORT.md**
- Purpose: Feature generation pipeline documentation
- Audience: Data engineers, feature developers
- Contents: Tools, scripts, generation process
- When to read: Generating or backfilling features

**DOMAIN_FEATURE_AVAILABILITY_REPORT_2022.md**
- Purpose: Feature availability assessment for 2022
- Audience: Backtest engineers, analysts
- Contents: What features exist, coverage gaps
- When to read: Planning backtests or feature generation

### Diagnostic & Troubleshooting

**DOMAIN_ENGINES_DIAGNOSTIC_REPORT.md**
- Purpose: Diagnostic findings and issues
- Audience: Developers, debuggers
- Contents: Issues discovered, root causes, solutions
- When to read: Debugging domain engine behavior

**DOMAIN_ENGINE_STATUS_REPORT.md**
- Purpose: Current implementation status
- Audience: Project stakeholders
- Contents: What's done, what's pending, next steps
- When to read: Checking current state

### Bug Fixes

**CRITICAL_BUG_FIX_FEATURE_FLAGS.md**
- Purpose: Documentation of critical bug fix
- Audience: Developers, QA engineers
- Contents: Bug description, root cause, fix, impact
- When to read: Understanding feature flag propagation issue

## Feature Coverage

### Wyckoff Engine (13 events)
- Spring A/B, SC (Selling Climax), ST (Secondary Test)
- AR (Automatic Rally), LPS (Last Point of Support)
- SOS (Sign of Strength), SOW (Sign of Weakness)
- BC (Buying Climax), UTAD (Upthrust After Distribution)
- LPSY (Last Point of Supply)

### SMC Engine (4 features)
- 4H BOS (Break of Structure)
- Demand/Supply zones
- Liquidity sweep
- CHOCH (Change of Character)

### HOB Engine (3 features)
- Demand/Supply zones (order book based)
- Bid/Ask imbalance

### Temporal Engine (4 features)
- Fibonacci time clusters
- Temporal confluence
- Resistance/Support temporal clusters
- Wyckoff-PTI fusion

### Macro Engine (1 feature)
- Macro regime alignment

**Total**: 44 unique features across 5 engines

## Archetype Integration

### S1 - Liquidity Vacuum (37 features)
- **Wyckoff**: 13 events with boost values 1.40x-2.50x
- **SMC**: 9 features with boost values 1.40x-2.00x
- **Temporal**: 8 features with boost values 1.50x-1.80x
- **HOB**: 3 features with boost values 1.30x-1.50x
- **Macro**: 1 feature with boost value 1.20x
- **Max Boost**: Up to 95x (full confluence, realistic: 8-12x)

### S4 - Funding Divergence (33 features)
- **Wyckoff**: 12 events with boost values 1.50x-2.50x
- **SMC**: 8 features with boost values 1.60x-2.00x
- **Temporal**: 3 features with boost values 1.40x-1.70x
- **HOB**: 2 features with boost values 1.30x-1.40x
- **Max Boost**: Up to 60x (full confluence, realistic: 6-10x)

### S5 - Long Squeeze (35 features)
- **Wyckoff**: 12 events with boost values 1.60x-2.50x
- **SMC**: 8 features with boost values 1.60x-2.00x
- **Temporal**: 5 features with boost values 1.50x-1.80x
- **HOB**: 5 features with boost values 1.30x-1.50x
- **Max Boost**: Up to 70x (full confluence, realistic: 7-11x)

## Feature Flags

All domain engines are controlled by feature flags:
- `enable_wyckoff`: Enable Wyckoff event detection
- `enable_smc`: Enable Smart Money Concepts
- `enable_temporal`: Enable Temporal Fusion
- `enable_hob`: Enable Hidden Order Book analysis
- `enable_macro`: Enable Macro regime alignment

Set these in your config to control which engines are active.

## Veto System

**Hard Vetoes** (abort signal completely):
- S1: Distribution phase, accumulation veto
- S4: Distribution phase, SOW veto
- S5: Accumulation phase, spring veto

**Soft Vetoes** (reduce boost):
- S1: Supply zones (0.70x), resistance clusters (0.75x)
- S4: Supply zones (0.70x), SOW soft veto (0.70x)
- S5: Support clusters (0.75x), demand zones penalty

## Performance Impact

**Expected Improvements**:
- Profit Factor: +10-30% on high-confidence signals
- Win Rate: +5-15% on confluence signals
- Sharpe Ratio: +0.1-0.3 with proper filtering
- Drawdown: -2-8% with better entry timing

**Realistic Boost Range**:
- Typical: 2x-4x on single engine alignment
- Good: 5x-8x on multi-engine confluence
- Excellent: 10x-15x on full temporal + structural confluence
- Maximum: Up to 95x (theoretical, rare)

## Usage Examples

### Enable All Engines
```json
{
  "enable_wyckoff": true,
  "enable_smc": true,
  "enable_temporal": true,
  "enable_hob": true,
  "enable_macro": true
}
```

### Conservative (Wyckoff Only)
```json
{
  "enable_wyckoff": true,
  "enable_smc": false,
  "enable_temporal": false,
  "enable_hob": false,
  "enable_macro": false
}
```

### Structural Focus (Wyckoff + SMC)
```json
{
  "enable_wyckoff": true,
  "enable_smc": true,
  "enable_temporal": false,
  "enable_hob": false,
  "enable_macro": false
}
```

## Related Documentation

**In Parent Directory**:
- `/docs/ALPHA_COMPLETENESS_VERIFICATION_REPORT.md` - Feature completeness check
- `/docs/ALPHA_GAP_ACTION_PLAN.md` - Remaining gaps and action plan

**In Root Directory**:
- `GIT_COMMIT_PLAN.md` - How this code was committed
- `CHANGELOG.md` - Version history and changes

**In Engine Directory**:
- `/engine/archetypes/logic_v2_adapter.py` - Implementation code
- `/engine/features/registry.py` - Feature specifications

## Contributing

When adding new features:
1. Update feature specifications in `registry.py`
2. Add boost logic in `logic_v2_adapter.py`
3. Document in this directory
4. Update `COMPLETE_ENGINE_WIRING_REPORT.md`
5. Add to `CHANGELOG.md`

## Support

For questions or issues:
1. Check `DOMAIN_ENGINE_GUIDE.md` troubleshooting section
2. Review `DOMAIN_ENGINES_DIAGNOSTIC_REPORT.md` for known issues
3. Check `CRITICAL_BUG_FIX_FEATURE_FLAGS.md` for bug history
4. Consult technical lead if unresolved

## Version History

**v1.0.0** - December 11, 2024
- Initial domain engine implementation
- 44 features across 5 engines
- Complete S1, S4, S5 integration
- Feature flag system
- Veto protection system

---

**Last Updated**: December 11, 2024
**Status**: Complete and Ready for Production Testing
**Maintainer**: Bull Machine Development Team
