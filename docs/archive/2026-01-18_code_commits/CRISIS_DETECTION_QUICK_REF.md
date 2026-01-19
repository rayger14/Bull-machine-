# Crisis Detection Quick Reference

## Features Required

```python
# From engine/temporal/gann_cycles.py
'thermo_floor'      # Mining cost floor price (BTC only)
'thermo_distance'   # (price - floor) / floor
'lppls_veto'        # Boolean: blowoff detected
'lppls_confidence'  # Score 0.0-1.0
'symbol'            # Asset symbol (e.g., 'BTCUSDT')
```

---

## LONG Archetypes (S1, S4, H, B, K, A, C)

### LPPLS Veto (in `_check_vetoes()`)
```python
lppls_veto = row.get('lppls_veto', False)
lppls_confidence = row.get('lppls_confidence', 0.0)
if lppls_veto and lppls_confidence > 0.75:
    return f'lppls_blowoff_detected_conf_{lppls_confidence:.2f}'
```

### Thermo-floor Boost (in `detect()` after fusion)
```python
symbol = row.get('symbol', 'BTCUSDT')
if 'BTC' in symbol:
    thermo_distance = row.get('thermo_distance', 0.0)
    if thermo_distance < -0.10:
        fusion_score *= 2.00
        logger.debug(f"[{ID}] Thermo boost: {thermo_distance:.2f}")
```

---

## SHORT Archetype (S5)

### LPPLS Boost (in `detect()` after fusion)
```python
lppls_veto = row.get('lppls_veto', False)
lppls_confidence = row.get('lppls_confidence', 0.0)
if lppls_veto and lppls_confidence > 0.75:
    fusion_score *= 2.00
    logger.debug(f"[S5] LPPLS boost: {lppls_confidence:.2f}")
```

### Thermo-floor Veto (in `_check_vetoes()`)
```python
symbol = row.get('symbol', 'BTCUSDT')
if 'BTC' in symbol:
    thermo_distance = row.get('thermo_distance', 0.0)
    if thermo_distance < 0.10:
        return f'thermo_floor_capitulation_veto_{thermo_distance:.2f}'
```

---

## Summary Table

| Feature | LONG Archetypes | SHORT Archetype (S5) |
|---------|----------------|---------------------|
| **LPPLS** | ❌ VETO tops (conf > 0.75) | ✅ BOOST tops 2× (conf > 0.75) |
| **Thermo** | ✅ BOOST bottoms 2× (dist < -0.10) | ❌ VETO bottoms (dist < 0.10) |

---

## Testing Checklist

- [ ] Verify temporal features in feature DataFrame
- [ ] Test LPPLS veto on 2021-11 top (should block longs)
- [ ] Test thermo boost on 2022-06 bottom (should boost longs 2×)
- [ ] Test S5 LPPLS boost on 2021-04 top (should boost shorts 2×)
- [ ] Test S5 thermo veto on 2022-11 bottom (should block shorts)
- [ ] Run full backtest 2022-2024
- [ ] Validate +25 bps improvement target

---

## Files Modified

1. `bear/liquidity_vacuum.py` - LPPLS veto + thermo boost
2. `bear/funding_divergence.py` - LPPLS veto + thermo boost
3. `bear/long_squeeze.py` - LPPLS boost + thermo veto
4. `bull/trap_within_trend.py` - LPPLS veto + thermo boost
5. `bull/order_block_retest.py` - LPPLS veto + thermo boost
6. `bull/wick_trap_moneytaur.py` - LPPLS veto + thermo boost
7. `bull/spring_utad.py` - LPPLS veto + thermo boost
8. `bull/bos_choch_reversal.py` - LPPLS veto + thermo boost
