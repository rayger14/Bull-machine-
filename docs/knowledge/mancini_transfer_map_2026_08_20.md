# Mancini→BM Transfer Map + 75/15/10 Exit A/B — 2026-08-20

**Source**: full read-only inventory of the Mancini ES bot (~/Mancini bot/Mancini; corpus baseline +$43.6K/−$5.2K DD, 2024-07→2026-02). Its core trade IS wick_trap (sweep below significant low → reclaim → confirm) — evidence unusually transferable.

## 75/15/10 exit A/B (BM full backtest 2020-2024, all archetypes, forced override)
Variant: 75% @1R, 15% @2R, 10% runner, trail 3.5×ATR after T1, NO breakeven (BM retracted 2×).
| | base 10/30/50 | 75/15/10 |
|---|---|---|
| PnL | $268.9K | $240.9K (−10%) |
| PF/Sharpe | 1.41/1.53 | 1.39/1.50 |
| MaxDD | −16.5% | −15.2% (better) |
| trades | 3512 | 2694 (−23%) |
| avg/trade | $77 | $89 (+17%); avg win +37%, avg loss smaller |
**VERDICT: REJECT as straight swap.** Per-trade quality improves everywhere; total PnL falls because longer holds block position slots (−23% turnover). BM PnL = turnover × small edge. A fair re-test requires runner-slot demotion (Mancini's fix, +$10.7K there) — not justified while entry quality is the known leak (their 5y sweep verbatim: "every configuration has every year red regardless of exit tuning. The issue isn't exits — it's entry quality" = BM bucket-A finding).
**$2M-wallet control (same day)**: identical trade counts (3512/2694) and ratios at 20x capital — margin was NEVER the constraint. The −818 trades are STRUCTURAL (archetype/direction position slots occupied by longer holds), so the turnover loss holds at any wallet size and equals lost data points under the edge-discovery mandate. Runner-slot demotion is the hard prerequisite for any long-runner exit scheme in BM.
Method note: first A/B ran byte-identical (patched dead default_rules; real ladder lives in create_default_exit_config per-archetype). Golden-master lesson self-demonstrated.

## Transfer candidates (evidence-backed, in priority order)
1. **Sizing-inversion check** — their meta-label audit: tight stops = 33%-WR anti-tell; deep flushes = 80% WR + biggest winners; "size down on wide stops" RETIRED as inverted. BM sizes inversely to ATR-stop distance → may be shrinking deep-flush winners exactly the same way (cf. BM unified forward-test: tight stop stabbed 7/10). Pure backtest on existing data. NEXT STUDY.
2. **Velocity-at-reclaim boost for wick_trap** — their dominant, regime-invariant quality signal (holds every year, 498k events; "gate on speed not the clock"). BM wick_trap has no velocity term. Boost-shaped.
3. **Macro-event blackout** — measured: P(outsized bar) FOMC 95%/CPI 93%/NFP 70% vs 3% baseline; hard-block shipped there. BM trades through FOMC blind. Stand-down calendar, not a fusion filter.
4. **Re-entry throttling** — un-freezing their accidental no-re-signal discipline cost −$15.1K on 2× entries. BM analog: trap_within_trend pileups (2+ open → PF 0.42). Per-archetype cooldown study.
5. **Runner mechanics IF ever revisited**: T2 snapped to real levels; trail ratchets EOD-only (per-bar chase-trail was their "888 killer", exited before +404pts — BM's per-bar ATR chase is the same pattern); runners demote to background slot.

## Anti-transfer (their closed doors that map to ours)
- Regime gate on full stack: −$9.9K there; 0/9 filters here. Convergent.
- **Order-flow absorption from aggregate trades: KILLED on adversarial verification there** — read their post-mortem before building BM's B-tier absorption_flag.
- Levels-as-targets: premature exits (+67→+50 avg win). Caution for level-snapped targets.
- Non-price data (TICK/VPIN/CVD): all killed. Their edge source = human newsletter (engine-only is −$7K 2021-24); BM's = structural detection (fusion is anti-signal). Neither engine survives on generic indicators.

## Process rules adopted from their ledger
Golden-master before trusting any A/B (demonstrated today); count entries before believing results; one concern per change; pre-registered gates; None-sentinel replay flags.
