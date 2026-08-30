# V23 Parity Store — Re-baseline & First Verdicts (2026-08-29)

## THE STORE
V23_PARITY: 59,901 bars (2018-03→2024-12), 359 cols. Built by the REAL LiveFeatureComputer (TA-Lib verified vs server) in 8 parity-faithful shards (fresh 1000-bar warmups = live-restart semantics; seam QA: only long-memory/externally-fetched cols diverge at boundaries). External witnesses (Vision derivatives, macro) overlaid from validated sources with provenance. Spot-parity: chop pass 52.6% (V12 broken 5.9%), wick 0-1 live def, wyckoff pin broken (11.8% vs 97%), adx_14 exists, pre-2022 blackout gone.

## THE HONEST BASELINE (current live config, 2020-2024)
**PF 1.25, $118K, MaxDD −31.8%, Sharpe 0.61, 1,884 trades.** Old store said $304K/−15.8%. **Honest backtest DD (−31.8%) ≈ live actual DD (−32.7%) — the backtest predicts live for the first time.** Real book = flush family: wick_trap +$56.4K, liquidity_sweep +$51.6K, LC +$13.8K (fire rate 0.4→1.6/mo; residual gap to live's ~6 remains). Worst: spring −$9.7K (proxy confirmed rotten on honest data). 2022 bear −$27K.

## SIZING PACKAGE DECOMPOSITION ON V23 — the deployed package inverts
| | OFF | DIAL-only | FLAT-only | BOTH (deployed) |
|---|---|---|---|---|
| PnL | $117.8K | $101.7K | **$137.7K** | $118.5K |
| MaxDD | −26.1 | −31.1 | **−25.5** | −31.8 |
| years better | — | 1/5 | **4/5** | — |
**FLAT-NOTIONAL: KEEP (+17%, better DD). TAPE DIAL: RETIRE live** (−$16K/worse DD on honest data; the June live-cohort effect was real but does not survive 5 honest years; live loss cluster 08-27/28 incl. dial-boosted top wick_trap was the live illustration). Flag split shipped (tape_dial_enabled=false); config(dial-off) == flat-only verified to the penny ($137,711.04).

## PROVEN-TRIO SELECTIVITY (310 positions, honest store)
By 20-day range location: bottom third 30% WR −$10.7K (knife-catches, whale-zone verdict re-confirmed); middle +$40.5K; **top third +$91.8K @50%** — the trio's money is trend-pullback buying near highs, NOT bottom-fishing. W/L otherwise near-inseparable at entry (only swing_low_touches 13v10 and post-impulse 23%v16% separate). → Queued V23-native studies: (1) de-size bottom-third dip-buys (sizing), (2) restore entry spacing live as cooldown (bypass currently skips it; Mancini re-signal law + 08-27/28 cluster), (3) swing_low_touches maturity boost bucket test.

## EXPOSED CONCLUSIONS STATUS
Re-verdicted: sizing package (above). Still to re-verdict on V23: wyckoff_phase & distribution_exhaustion boosts (free-pass leg), OBR hard gate (PR #72), whale-penalty taker threshold, sweep-location (August REJECT ran on contaminated store). Industry context: pro systematic desks −10..−20% MaxDD @ Sharpe 1-2; we are −32% @ 0.6 — the gap is throttle/concentration, not entries.

## OBR hard-gate silo re-test (2026-08-30): PR #72 CLOSED UNMERGED
Fair trial (silo, V23): soft PF 1.00/−$84 vs hard PF 0.82/−$10,953/DD −27.6% — the hard gate makes OBR WORSE on honest data. The original broken-store portfolio validation (+$1.1K) was artifact. Identity-gates campaign final tally: OBR-hard mirage, CB enforcement reject, boms floor dead config — ZERO shipped changes, honestly. OBR stays a Reject in both modes; 13th entry-rule funeral. Also same-day: wick_trap vol-floor 1.5 REJECT (−$23K) and LC high-side-only REJECT (−$18.6K) in silo — winners' edges confirmed participation-broad on honest instruments.
