# Open Interest Data Availability Issue

**Date**: 2025-11-19
**Status**: BLOCKED - Historical OI data not available

## Summary

Phase 1 backfill for Open Interest (OI) features cannot proceed due to lack of historical data for 2022-2024 period.

## Investigation Results

### 1. OKX API (`/api/v5/rubik/stat/contracts/open-interest-history`)
- **Status**: NO historical data available
- **Test**: Queried May 2022 (Terra collapse period)
- **Result**: API returned code='0' (success) but ZERO records
- **Coverage**: Only recent data (~2025 onwards)

### 2. Existing CSV Files
- **File**: `data/OI_1H.csv` → symlink to Binance data
- **Coverage**: Sep 20 - Oct 2, 2025 (300 hours)
- **Status**: Future data only, not useful for historical backtest

### 3. Alternative Sources
- Binance, Bybit, Deribit: Require premium API access or data purchase
- Free tier APIs: Do not provide historical OI data beyond ~90 days

## Impact Analysis

### Affected Archetype
- **S5 (Long Squeeze)**: Requires `oi_change_pct_24h` for liquidation cascade detection

### Current Behavior
- S5 already has **graceful degradation**: returns 0.0 when OI data is missing
- S5 currently produces ZERO matches (expected, since OI feature is missing)

### Phase 1 Validation Impact
- **Minimal**: S5 is 1 of 6 bear market archetypes
- Other archetypes (S1-S4, S6) do NOT require OI data
- System can still validate with S5 gracefully degraded

## Decision: Skip OI, Proceed with Phase 1

### Rationale
1. **Graceful Degradation Works**: S5 returns 0.0 safely when OI is missing
2. **One Archetype Among Many**: 5 other bear archetypes still functional
3. **Other Features Available**: liquidity_score, macro features can proceed
4. **Production Path**: OI can be added later via live data feed (OKX real-time API works)

### Path Forward
1. ✅ Complete `liquidity_score` backfill (next task)
2. ✅ Complete macro features backfill (DXY, VIX, MOVE, rates)
3. ✅ Run Phase 1 validation with S5 gracefully degraded
4. 📝 Document that S5 requires live OI feed for production use

## Alternative: Synthetic OI Proxy (Future Work)

If OI signal proves critical post-Phase 1, implement **Option B** from `OI_PIPELINE_SPEC.md`:

### Synthetic OI Score Formula
```python
oi_proxy = (
    0.40 * volume_z_change_24h +      # Volume spike
    0.30 * abs(funding_rate_change) + # Funding acceleration
    0.20 * price_vol_divergence +     # Divergence
    0.10 * wick_ratio                 # Liquidation proxy
)
```

**Effort**: ~4 hours
**Accuracy**: Approximate correlation with real OI (~70-80%)

## Files Modified
- `bin/fix_oi_change_pipeline.py`: Updated endpoint, fixed data parsing (for future use)
- `bin/test_okx_api.py`: Fixed test for OKX API format validation

## References
- OI Pipeline Spec: `docs/OI_PIPELINE_SPEC.md`
- S5 Archetype: `engine/archetypes/bear_patterns_phase1.py` lines 450-550
- Feature Registry: `engine/features/registry.py` (OI features NOT added)
