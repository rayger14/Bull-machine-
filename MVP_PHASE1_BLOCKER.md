# MVP Phase 1: SMC Structure Detection Blocker

**Status**: BLOCKED - SMC structures not detecting on SPY  
**Date**: 2025-10-20  
**Phase**: Phase 1 - Structure Invalidation Exits

## TL;DR

Phase 1 SMC structure extraction (Order Blocks, FVGs, BOS) produces **zero detections** on SPY 2024 data, causing structure invalidation exit logic to never fire.

## Problem

### SPY Feature Store Analysis
All OB/BB/FVG level columns are 0% populated (completely NULL).

### Direct SMC Test
SMC engine on 200-bar SPY window:
- Order blocks: 0
- FVGs: 0  
- BOS events: 0

### Backtest Impact
- Structure Invalidation Exits: 0 (0.0%)
- Phase 1 logic never triggers

## Root Cause

SMC concepts (Order Blocks, FVGs) designed for crypto 24/7 markets. SPY characteristics:
- RTH only (gaps, lower volatility)
- Less prone to liquidation wicks
- Smaller price swings vs crypto

## Next Steps

**WAIT for BTC/ETH builds** (~30 min):
- If BTC/ETH show good detection (>50%) → Tune thresholds for SPY
- If BTC/ETH also zero → Fix SMC engine itself

## Resolution Options

1. **Tune SMC thresholds for SPY** (lower wick ratios, volume surges)
2. **Skip Phase 1 for SPY** (test on crypto only)
3. **Use 4H SMC structures** (larger timeframe for equities)
4. **Alternative structures** (HOB/BOMS/FRVP instead of SMC)

**Recommendation**: Wait for BTC/ETH results, then decide.
