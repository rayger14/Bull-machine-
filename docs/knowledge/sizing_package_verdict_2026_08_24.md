# Sizing Package — flat-notional + post-impulse dial — VALIDATED (2026-08-24)

Three pre-registered interventional A/Bs, all divergence-verified, entries identical (3,512) in every run:
| variant | PnL | PF | MaxDD | Sharpe | years up |
|---|---|---|---|---|---|
| baseline (ATR-inverse, no dial) | $268.9K | 1.41 | −16.5% | 1.53 | — |
| flat-notional only | $300.3K (+11.7%) | 1.41 | −16.0% | 1.44 | 5/5 |
| impulse dial only (1.25x ≤3d post +10%/3d impulse; 0.75x dead; 1.0x base — causal) | $283.9K (+5.6%) | **1.47** | **−13.1%** | **1.62** | 5/5 (bear best) |
| **COMBINED** | **$307.2K (+14.2%)** | 1.46 | **−13.9%** | 1.54 | **5/5** |
Per-archetype: wick_trap +19% ($61.0→72.4K), LC up, OBR loss shrinks. Dial de-sizes dead tape → bear-year bleed shrinks (2022 +$5.5K); flat-notional feeds the wide-stop winner class (Q4 = 51% WR, 55% of profits at smallest size under baseline). Effects near-additive; dial fixes flat's Sharpe dip.

**Implementation if approved**: backtest line 1449 → `notional = risk$/0.025 * dial_mult`; live runner needs the same divisor change + daily causal dial from own candle history (r3(daily)≥10% within 3d → 1.25x; 12d range ≤8% → 1.0x; else 0.75x). Sizing-family change (historically 2/2 accepted); honest limits: single instrument, 2020-2024, no offline holdout exists — real OOS = live forward with a 4-week review. NOT applied anywhere yet.
