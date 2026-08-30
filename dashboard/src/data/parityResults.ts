// V23 PARITY-STORE research verdicts — 2026-08-29
// Source: docs/knowledge/v23_rebaseline_verdicts_2026_08_29.md + silo leaderboard.
// SILO = archetype backtested ALONE (no slot competition) on the honest V23
// store (2020-2024, live-identical feature computation). This is the
// graduation instrument; the book comparison below is the portfolio view.

export interface ParityResult {
  positions: number;
  wr: number;        // %
  pf: number;
  pnl: number;       // USD, 5yr silo, $2M wallet, flat-notional sizing
  maxDD: number;     // %
  verdict: 'GRADUATE' | 'REJECT' | 'THIN';
  note?: string;
}

export const PARITY_DATE = '2026-08-29';
export const PARITY_METHOD =
  'Siloed backtests on the V23 parity store (2018-2024 rebuilt with the live feature computer; ' +
  'honest DD matches live within 1pp). Each archetype tested alone — no dedup/slot interference. ' +
  'Graduation rule (pre-registered): PF ≥ 1.2 and positive PnL.';

export const PARITY_RESULTS: Record<string, ParityResult> = {
  liquidity_sweep:      { positions: 226, wr: 77.4, pf: 1.54, pnl: 73494,  maxDD: -13.0, verdict: 'GRADUATE' },
  wick_trap:            { positions: 150, wr: 78.5, pf: 1.56, pnl: 69323,  maxDD: -10.5, verdict: 'GRADUATE' },
  liquidity_compression:{ positions: 132, wr: 78.7, pf: 1.43, pnl: 29836,  maxDD: -6.2,  verdict: 'GRADUATE' },
  failed_continuation:  { positions: 100, wr: 76.7, pf: 1.25, pnl: 12615,  maxDD: -14.9, verdict: 'GRADUATE',
    note: 'Prior "BROKEN/never fires" verdict overturned by honest data.' },
  confluence_breakout:  { positions: 41,  wr: 77.2, pf: 1.70, pnl: 11001,  maxDD: -4.3,  verdict: 'GRADUATE',
    note: 'Highest PF in the book when siloed — prior "worst bleeder" reputation was slot-noise + contaminated data. n=41.' },
  oi_divergence:        { positions: 46,  wr: 70.0, pf: 1.48, pnl: 8981,   maxDD: -5.5,  verdict: 'GRADUATE',
    note: 'Positive even on the perp OI witness; CME witness promotion pending shadow.' },
  hob_reaction:         { positions: 72,  wr: 74.7, pf: 1.24, pnl: 7598,   maxDD: -9.0,  verdict: 'GRADUATE' },
  funding_divergence:   { positions: 4,   wr: 81.8, pf: 1.49, pnl: 1333,   maxDD: -2.5,  verdict: 'THIN',
    note: 'n=4 — witness gates starved (22x); passes on numbers but sample is uncallable.' },
  order_block_retest:   { positions: 102, wr: 72.7, pf: 1.00, pnl: -84,    maxDD: -18.2, verdict: 'REJECT',
    note: 'Soft-gate variant. Hard-gate identity restoration (PR #72) pending silo re-test.' },
  exhaustion_reversal:  { positions: 72,  wr: 75.5, pf: 0.98, pnl: -699,   maxDD: -15.9, verdict: 'REJECT' },
  long_squeeze:         { positions: 1,   wr: 0.0,  pf: 0.0,  pnl: -772,   maxDD: -1.0,  verdict: 'THIN',
    note: 'n=1 — effectively never fires on honest data.' },
  liquidity_vacuum:     { positions: 30,  wr: 73.0, pf: 0.76, pnl: -3125,  maxDD: -4.7,  verdict: 'REJECT' },
  fvg_continuation:     { positions: 214, wr: 67.1, pf: 0.90, pnl: -7587,  maxDD: -22.0, verdict: 'REJECT' },
  spring:               { positions: 53,  wr: 64.8, pf: 0.80, pnl: -7639,  maxDD: -22.4, verdict: 'REJECT',
    note: 'Rolling-low proxy fires mid-trend, not at structural lows (audit add.31) — confirmed on honest data.' },
  retest_cluster:       { positions: 63,  wr: 64.8, pf: 0.67, pnl: -14608, maxDD: -19.5, verdict: 'REJECT',
    note: 'Worst archetype on honest data.' },
  trap_within_trend:    { positions: 155, wr: 77.5, pf: 1.08, pnl: 12880,  maxDD: -18.8, verdict: 'REJECT',
    note: 'Positive PnL but PF 1.08 < 1.2 graduation bar; DD heavy.' },
};

export const BOOK_COMPARISON = [
  { label: 'Current 16-archetype book', pnl: 118452, pf: 1.25, maxDD: -31.8, sharpe: 0.61,
    note: 'All archetypes, equal size (live data-collection mandate). Honest-store DD matches live (-32.7%).' },
  { label: 'Graduated book (8 winners only)', pnl: 153741, pf: 1.42, maxDD: -17.4, sharpe: 1.00,
    note: '+30% PnL at nearly half the drawdown. The rejects cost ~$35K + 14pp DD per 5yr.' },
  { label: 'Silo ceiling (no interaction)', pnl: 214181, pf: NaN, maxDD: NaN, sharpe: NaN,
    note: 'Sum of graduate silos — upper bound, not a portfolio.' },
];
