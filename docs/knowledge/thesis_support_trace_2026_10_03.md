# Support setup confirmation trace

This saved-source trace explains the final gate of the October 3 census. It does
not score trades or test alternate thresholds. All times are UTC and prices are
BTC quote prices from the pinned archive.

Each episode requires a subsequent complete minute close **strictly above the
support candle's high**, and **strictly before support availability plus 15
minutes**. Therefore 14 complete minute closes are eligible, not the close at
the expiry itself. All 14 were present for every row. The maximum below summarizes
that entire decision window; it is not evidence available before the window ends.

| Spring available | Support available | Required close above | Highest eligible minute close | First confirmation |
| --- | --- | ---: | ---: | --- |
| 2024-03-17 04:00 | 2024-03-17 13:00 | 67474.8 | 67234.4 | Expired |
| 2024-03-17 08:00 | 2024-03-17 13:00 | 67474.8 | 67234.4 | Expired |
| 2024-04-27 04:00 | 2024-04-28 04:00 | 64350.0 | 64018.5 | Expired |
| 2024-04-27 12:00 | 2024-04-28 23:00 | 63691.5 | 63090.5 | Expired |
| 2024-05-06 20:00 | 2024-05-07 11:00 | 64150.0 | 64090.7 | Expired |
| 2024-05-11 00:00 | 2024-05-11 18:00 | 61215.6 | 61042.5 | Expired |
| 2024-05-29 20:00 | 2024-05-30 07:00 | 68227.3 | 67964.0 | Expired |
| 2024-07-12 08:00 | 2024-07-12 21:00 | 57622.9 | 57621.4 | Expired |
| 2025-01-10 00:00 | 2025-01-10 16:00 | 94498.4 | 93628.0 | Expired |
| 2025-03-31 12:00 | 2025-03-31 23:00 | 82538.1 | 82559.0 | 23:07 |
| 2025-04-27 16:00 | 2025-04-27 22:00 | 94517.9 | 94234.0 | Expired |
| 2025-07-30 00:00 | 2025-07-30 04:00 | 118210.0 | 118224.9 | 04:08 |
| 2025-09-16 04:00 | 2025-09-17 15:00 | 116207.3 | 115739.8 | Expired |
| 2025-10-22 12:00 | 2025-10-22 15:00 | 108990.0 | 108985.8 | Expired |
| 2025-11-16 20:00 | 2025-11-17 09:00 | 95980.1 | 95600.6 | Expired |
| 2026-01-12 00:00 | 2026-01-12 18:00 | 92264.2 | 91710.8 | Expired |
| 2026-02-12 20:00 | 2026-02-13 15:00 | 67999.9 | 68481.8 | 15:10 |

There are 17 episode rows but 16 distinct support candles/windows: the March 17,
2024 rows originate from two different parent lineages and share the same later
support. They are not independent pieces of performance evidence. Confirmation
times use the date in the support column. Expired means the required close did
not occur in this window, not that the hypothetical trade would have lost money.

The two near misses are not grounds for moving the threshold: choosing a buffer
after seeing these prices would create a new, data-informed hypothesis. Review
the trader-source justification for this exact gate before proposing a revision.

Source: `results/thesis_census_2026_10_03/source_v1/source.json`, SHA256
`b27440a9247ea93b77458f08fc13009ab56679a43bbfc66aba3a86d43039730b`.
Read the [census results and limits](thesis_census_results_2026_10_03.md) for context.
