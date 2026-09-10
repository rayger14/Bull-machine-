# External execution libraries: accuracy assessment

## Decision

Do not add a new production dependency for the fusion scorecard. This unit compares recorded evidence and requires no new simulator. Investigate a separately pinned NautilusTrader execution benchmark after the decision stream and fill contract are frozen. Reserve hftbacktest for a depth/queue study with suitable data. No package was installed or production dependency modified in this phase.

An execution library can model specified order timing and fills. It cannot recover missing historical feature receipts, identify an unknown venue, authenticate a source/config version, or reconstruct unlogged candidates. Earlier causal-parent and replay work has improved those interfaces, but historical live-consumption completeness remains uncertified. A different library is not, by itself, a more accurate strategy backtest.

## Existing code is not the external Nautilus engine

`engine/integrations/nautilus_strategy.py:39` imports the repository's own `EventEngine`, not `nautilus_trader`. `engine/integrations/event_engine.py:198` exposes market submission; `:214` processes pending market orders against a bar open with fixed-basis-point slippage. Naming it Nautilus does not provide the external engine's event, quote or depth models. This is a read-only architectural observation, not an active-path replacement.

## Candidates and limits

**NautilusTrader:** the better later candidate for general event-driven order/execution comparison. Its current documentation supports Python3.12–3.14 and macOS15+ARM64, unlike this research interpreter's Python3.9.6; use an isolated environment, not an in-place upgrade. Stable and release-candidate documentation must be matched to the exact pinned package; do not assume latest-doc features exist in an older stable build. See [official installation and supported platforms](https://nautilustrader.io/docs/latest/getting_started/installation/).

It accepts different market-data granularities and book types, but bars cannot supply historical bid/ask queues or unobserved event ordering. Select the data model to match the actual source; do not manufacture depth from candles. See [official data and venue contracts](https://nautilustrader.io/docs/latest/concepts/backtesting/data-and-venues/).

**hftbacktest:** useful for exchange-depth, latency and queue modeling when event-level data exists. Replayed orders do not alter the historical market; market impact is absent. Its no-partial-fill model can fill liquidity-taking orders at the best price regardless of displayed quantity, while partial-fill models still cannot change the recorded future book. Queue position under market-by-price data remains modeled. See [official fill/queue assumptions](https://hftbacktest.readthedocs.io/en/latest/order_fill.html). These limitations matter before claiming improved fills.

The independent assessment found the local environment has older NumPy/Numba versions than the root project declarations. No resolver or dependency upgrade was run: preserve the currently verified research environment and isolate any future package experiment.

## Proposed small execution benchmark

No market data or profitability search is needed for this first comparison. Freeze synthetic quote/order events and expected behavior:

1. A signal is available at 01:00:00 UTC; order-insert latency is 90 seconds.
2. A quote at 01:01:29.900 is bid 99.9 / ask 100.1; it must not fill the not-yet-ready order.
3. The first quote after readiness, 01:01:30.100, is bid 100.4 / ask 100.6. Under the explicitly chosen first-post-readiness-quote model, a market buy fills at 100.6, not at the earlier ask.
4. A subsequent sell stop at 95 faces a first executable bid 92 after a gap. Model the fill at the available bid with declared latency; do not award 95 merely because the order's stop level was 95.

This first-post-readiness-quote model is a proposed causal fixture contract, not a universal description of how resting venue liquidity behaves or a universally conservative price. An engine that fills against an already-active quote at order arrival requires a different explicitly stated contract, not a silent expectation change. Compare Bull Machine's declared research execution contract against a NautilusTrader adapter under the same fixed assumptions and preserve each order-state transition. A later hftbacktest comparison requires its own explicitly encoded depth/trade events, not a generic quote fixture.

Use an exact release/version and lockfile in a separate research environment, verify official provenance and license before installation, and record source hashes and runtime versions. No live integration or promotion follows from passing synthetic fixtures. Exchange-native quote/depth/receipt evidence and comparisons against observed fills remain necessary for empirical calibration.
