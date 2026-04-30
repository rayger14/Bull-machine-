# Dashboard Version History

## v7.5 (2026-02-19) — Honesty Audit + Misleading Number Cleanup
- **REMOVED**: Price Outlook (frontend fiction), Base Development maturity %, Sequence Position 9/10, `st_bc` ghost event, typical_durations
- **RELABELED**: "1H Score"→"1H Peak", "4H Phase"→"4H Peak", "1D Score"→"1D Peak", weights→"varies by archetype"
- **KEPT**: Active event chips, detection evidence, event history, structural narratives, methodology disclosure
- File: 1,359→995 lines

## v7.4 (2026-02-19) — Wyckoff Enrichment + Real Detection Evidence
- 1D Score FIX: 3-bar lookback, all 13 events. Score now 0.843 (was 0.0)
- UT/UTAD look-ahead bias FIX: replaced shift(-3) with same-bar detection
- Cycle Timeline, Event Confidence Breakdown, Event History, Structural Narratives
- Instance var pattern: enrichment data on `self.last_wyckoff_event_history`

## v7.3 (2026-02-19) — Market Briefing + Interactive Capital Flows + 1D Wyckoff Fix
- **MarketBriefing.tsx** (375 lines): Synthesizes regime, Wyckoff, macro, F&G, positions
  - buildHeadline(), buildImmediateOutlook(), buildWatchList()
- **CapitalFlows.tsx** (465 lines): 8 clickable asset nodes, 10 flow edges, NODE/EDGE_EDUCATION
- 1D Wyckoff: tf1d_daily_bars through full pipeline
- Layout: MarketBriefing at top, OpenPositions after ThresholdHero

## v7.2 (2026-02-19) — Education & Observation Mode
- Wyckoff Price Outlook, archetype observation mode, "What is Wyckoff?" education
- CMI "How This Works", gauge subtitles, correlation/cointegration messaging

## v7.1 (2026-02-18) — Types Rewrite & Enrichment
- Types rewrite, BTC price dedup, Wyckoff detection evidence
- Stress scenarios, capital flows, macro outlook, USDC.D, 1000-candle warmup

## v7.0 (2026-02-18) — React SPA Redesign
- **Architecture**: React 19 + TypeScript + Vite 6 + Tailwind CSS 4
- **Stack**: TanStack Query 5, Zustand 5, Recharts, Lightweight Charts, Framer Motion, ky, Lucide React
- **Design**: Glassmorphism dark theme, frosted glass cards, cyan/emerald accents, collapsible sidebar
- **Routing**: React Router 7 HashRouter, 5 lazy-loaded pages
- **Backend**: Flask dashboard.py 3,579→341 lines, serves dashboard/dist/ static files
- **Build**: `cd dashboard && npm run build` → dashboard/dist/ (14 chunks, ~1MB gzip)
- Replaced 3,579-line Flask+Alpine.js inline HTML

## v6 (2026-02-16) — 6 Enhancement Tasks
- Trades Tab, Regime-Colored Equity Curve, CMI Component History Chart
- Signal Funnel/Rejection Statistics, Wyckoff Cycle Visualization, Enhanced Position Cards
- 5 tabs: Dashboard, Strategy, Signals, Backtest, Trades

## v5 (2026-02-13) — Expandable CMI + Macro
- CMI gauges with component breakdowns + raw features
- Macro environment cards (F&G, BTC.D, USDT.D, VIX_Z, DXY_Z, Gold_Z, Oil_Z, Yield Curve)
- Macro Outlook: 4 timeframe predictions with factor breakdowns
- Archetype cards with gate conditions + collapsible code
