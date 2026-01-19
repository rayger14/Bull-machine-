# Domain Boost Multipliers - Reference Table

Quick reference for all boost/veto multipliers used in Archetypes A, B, H.

---

## Wyckoff Engine Multipliers

| Signal | Multiplier | Type | Description |
|--------|------------|------|-------------|
| Distribution Phase | 0.70x | VETO | Soft veto during distribution |
| UTAD Event | 0.70x | VETO | Upthrust after distribution |
| BC Event | 0.70x | VETO | Buying climax |
| **Spring A** | **2.50x** | **BOOST** | Deep fake breakdown (strongest) |
| **Spring B** | **2.50x/2.00x** | **BOOST** | Shallow spring |
| **Accumulation Phase** | **2.00x** | **BOOST** | Phase A confirmation |
| LPS (Last Point Support) | 1.50x | BOOST | Final test before markup |
| PS (Preliminary Support) | 1.30x | BOOST | Early accumulation |
| ST (Secondary Test) | 1.40x-1.50x | BOOST | Retest of lows |
| AR (Automatic Rally) | 1.30x | BOOST | Relief bounce |
| SOS (Sign of Strength) | 1.35x | BOOST | Decisive up move |

---

## SMC Engine Multipliers

| Signal | Multiplier | Type | Description |
|--------|------------|------|-------------|
| Supply Zone | 0.70x | VETO | Supply overhead (resistance) |
| 4H Bearish BOS | 0.70x | VETO | Bearish structure shift |
| **4H Bullish BOS** | **2.00x** | **BOOST** | Institutional shift (strongest) |
| 1H Bullish BOS | 1.40x | BOOST | Structural shift |
| Demand Zone | 1.50x-1.60x | BOOST | Institutional support |
| **Liquidity Sweep** | **1.80x** | **BOOST** | Stop hunt reversal |
| **Order Block Retest** | **1.80x** | **BOOST** | High probability retest |
| CHOCH | 1.50x-1.60x | BOOST | Change of Character |

---

## Temporal Engine Multipliers

| Signal | Multiplier | Type | Description |
|--------|------------|------|-------------|
| Resistance Cluster | 0.75x | VETO | Temporal resistance overhead |
| **Fibonacci Time Cluster** | **1.70x** | **BOOST** | Geometric reversal point |
| Multi-TF Confluence | 1.40x | BOOST | Multi-timeframe alignment |
| 4H High Fusion (>0.70) | 1.60x | BOOST | Strong trend alignment |
| Wyckoff-PTI Confluence | 1.20x-1.50x | BOOST | Combined pattern |

---

## HOB (Order Book) Multipliers

| Signal | Multiplier | Type | Description |
|--------|------------|------|-------------|
| Supply Zone | 0.70x | VETO | Supply wall overhead |
| **Demand Zone** | **1.50x** | **BOOST** | Institutional bid wall |
| Strong Bid Imbalance (>60%) | 1.30x | BOOST | More bids than asks |
| Moderate Bid Imbalance (>40%) | 1.15x | BOOST | Buyer imbalance |

---

## Macro Engine Multipliers

| Signal | Multiplier | Type | Description |
|--------|------------|------|-------------|
| High Crisis (>60%) | 0.85x | VETO | Extreme macro stress |
| **Risk-On (<30%)** | **1.20x** | **BOOST** | Favorable environment |

---

## Fusion Engine

| Signal | Multiplier | Type | Description |
|--------|------------|------|-------------|
| Fusion Meta-Layer | 1.0x | NEUTRAL | Handled globally |

---

## Multiplicative Stacking Examples

Boosts **multiply** together (not add):

```
Example 1: Maximum Boost (Spring A + 4H BOS + Fib Time)
  2.50 × 2.00 × 1.70 = 8.5x total boost

Example 2: Typical Boost (Accumulation + Demand Zone + Risk-On)
  2.00 × 1.60 × 1.20 = 3.84x total boost

Example 3: Veto Stack (Distribution + Supply + Crisis)
  0.70 × 0.70 × 0.85 = 0.42x total (58% penalty)

Example 4: Mixed (Spring B + Demand - Supply)
  2.00 × 1.50 × 0.70 = 2.1x total boost
```

---

## Application Order

1. Start: `domain_boost = 1.0`
2. Apply VETOES (multiply by <1.0)
3. Apply BOOSTS (multiply by >1.0)
4. Final: `score = score × domain_boost`
5. Gate check: `if score < threshold: reject`

**Critical**: Domain boost is applied **BEFORE** the fusion gate check, allowing marginal signals to qualify.

---

## Pattern Type Differences

### LONG Archetypes (A, B, H)
- **Veto on**: Distribution, UTAD, BC, Supply Zones, Resistance
- **Boost on**: Accumulation, Springs, Demand Zones, Support

### SHORT Archetypes (S1, S2, S4, S5)
- **Veto on**: Accumulation, Support, Demand Zones
- **Boost on**: Distribution, UTAD, BC, Supply Zones

---

## Usage in Code

```python
# Wyckoff boost example
if wyckoff_spring_a:
    domain_boost *= 2.50
    domain_signals.append("wyckoff_spring_a_trap_reversal")

# SMC boost example
if tf4h_bos_bullish:
    domain_boost *= 2.00
    domain_signals.append("smc_4h_bos_bullish_institutional")

# Apply boost
score_before_domain = score
score = score * domain_boost

# Return with metadata
meta = {
    "domain_boost": domain_boost,
    "domain_signals": domain_signals,
    "score_before_domain": score_before_domain
}
```

---

**Last Updated**: 2025-12-12  
**Archetypes**: A, B, H (LONG patterns)  
**Reference**: S1 implementation (lines 1740-1988)
