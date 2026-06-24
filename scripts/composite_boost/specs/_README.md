# Composite Boost Specs

Each JSON file describes one composite-feature sizing-boost rule discovered from
the 94-trade live audit (2026-04-20 → 2026-06-02). Cutoffs are live-data
terciles (HIGH = top tercile, LOW = bottom tercile) — see
`/tmp/composite_output.txt` for the derivation.

Top-of-funnel question: which combinations hold up on 2020-2024 historical data?

Run via `scripts/composite_boost/run_variant.py --spec <FILE>`.
