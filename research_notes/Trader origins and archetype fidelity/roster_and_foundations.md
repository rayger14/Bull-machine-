# Bull Machine roster, provenance and historical foundations

Scope: bounded audit completed 2026-10-01 on `quant/archetype-evidence-audit` at `85923a4`. I searched the current code/docs/configs and the specifically linked Mancini folder; I did not read every branch, git-history object, market-data artifact or social account. “Complete” below means every teaching/person provenance class discovered in that scope, with aliases and unknowns kept explicit.

## Who is actually in the teaching roster?

### Takeaway

The founding roster is six names, not five: Bojan, Moneytaur, Wyckoff Insider, IamZeroIka, Crypto Chase and Phoenix. Mancini was added later. Richard Wyckoff and W.D. Gann are historical foundations; `@TheAstronomer` is a separate code-only attribution. Eponymous indicators/research methods are incidental dependencies, not trader lineage.

### Cited Findings

| Provenance class | Discovered identity / alias | Evidence status |
|---|---|---|
| Founding teaching | Bojan → canonical `@Bojan_618` | Founding name in the six-person inventory; fresh original pair read. — [inventory](../../docs/knowledge/founding_knowledge_archaeology_2026_07_17.md#L3); [original](https://x.com/Bojan_618/status/2000564037780946977) |
| Founding teaching | `@Moneytaur` shorthand → canonical `@Moneytaur_` | Fresh originals establish the underscore handle. — [inventory](../../docs/knowledge/founding_knowledge_archaeology_2026_07_17.md#L3); [original](https://x.com/Moneytaur_/status/1819660778359644193) |
| Founding teaching | `@Wyckoff_Insider` | Contemporary X educator, not Richard D. Wyckoff. Three fresh originals read. — [inventory](../../docs/knowledge/founding_knowledge_archaeology_2026_07_17.md#L3); [original](https://x.com/Wyckoff_Insider/status/1949423128217633091) |
| Founding teaching | `@IamZeroIka`; legacy code spelling `@ZeroIKA` | Canonical account freshly read; old spelling is an alias, not another trader. — [original](https://x.com/IamZeroIka/status/1779921797434982462); [legacy label](../../docs/releases/V1.8.5.md#L25) |
| Founding teaching | Crypto Chase → canonical `@Crypto_Chase`; `@cryptochae` is uncorroborated typo | Author-controlled Linktree connects the matching X and YouTube channel; two author videos were transcribed. — [Linktree](https://linktr.ee/Crypto_Chase); [source ledger](original_trader_sources.md#which-founding-traders-have-freshly-read-originals) |
| Founding teaching, unresolved | Phoenix | Name exists in the founding inventory, but the scoped corpus supplies no handle, URL, distinct teaching or defensible identity match. — [inventory](../../docs/knowledge/founding_knowledge_archaeology_2026_07_17.md#L3) |
| Later transfer | Adam Mancini / Trade Companion | Explicitly introduced through a later ES-bot transfer map; the known local archive is original-like extracted article text, while `analysis_report.md` is generated synthesis. — [transfer map](../../docs/knowledge/mancini_transfer_map_2026_08_20.md#L1); [archive provenance](../Bull%20Machine%20teaching%20coverage/source_inventory.md#L14) |
| Separate code attribution | `@TheAstronomer` | Appears only as the named framework for `macro_echo`; no evidence equates it with Phoenix. — [macro_echo.py](../../engine/exits/macro_echo.py#L1) |
| Historical foundation | Richard D. Wyckoff; W.D. Gann | Their inspected historical works are method foundations, not the contemporary handles above. — [Wyckoff scan](https://archive.org/details/studiesintaperea00wyckrich); [Gann scan](https://archive.org/details/in.ernet.dli.2015.166248) |
| Incidental method/indicator authors | Welles Wilder; Marcos López de Prado; eponymous Bollinger and Kelly labels | Used for smoothing, HRP/research validation, bands and sizing; no evidence places them in the founding teaching roster. — [Wilder code](../../engine/models/baselines/rsi_mean_reversion.py#L19); [HRP code](../../engine/portfolio/hrp_allocator.py#L1); [Kelly integration](../../engine/integrations/isolated_archetype_engine.py#L1) |
| Unresolved family, not a person | SMC / OB / FVG / BOS; Fibonacci | These are project vocabulary/mathematical lineage. IamZeroIka discusses OB/FVG, but neither the scoped repository nor inspected source attributes Bull Machine's rules to ICT/Michael Huddleston. — [IamZeroIka original](https://x.com/IamZeroIka/status/1779921797434982462); [code-fidelity audit](active_code_fidelity.md#L101) |

#### Mancini later-transfer spot check

The two records below were preselected because the earlier audit already cited their dates/titles, before their bodies were read; this avoids choosing only favorable outcomes.

- **2025-11-19**, “Nvidia Earnings After 4pm. Volatility Coming For SPX, But What Way? November 20th Plan,” slug `nvidia-earnings-after-4pm-volatility`: identifies prior-day, multi-hour/20+ point, or clustered lows; requires loss → recovery → structural acceptance; describes nested reclaims; locates the stop below the complete structure's low; manages level-to-level with a small runner. — [saved original-text extraction](</Users/rayghandchi/Mancini bot/Mancini/data/substack/all_posts.json>)
- **2025-12-04**, “Another New High of Week For SPX, But Are Bulls Running Out Of Steam? Dec 5th Plan,” slug `another-new-high-of-week-for-spx`: qualifies acceptance by flush depth/volatility, warns against chasing the first recovery, invalidates below the structure low, sizes down for wide risk, then scales at successive levels and trails about 10%. — [saved original-text extraction](</Users/rayghandchi/Mancini bot/Mancini/data/substack/all_posts.json>)

These are ES newsletter teachings, not evidence that the BTC engine implements them or that their stated outcomes generalize. The JSON retains title/date/slug/text but lacks canonical URLs, per-record acquisition receipts and a chart archive. — [archive limitations](../Bull%20Machine%20teaching%20coverage/source_inventory.md#L26)

### Inferences

- A defensible roster must keep identity, curriculum and code derivation as separate fields. A code comment bearing a name is not proof that the named author prescribed the executable thresholds.
- `@Wyckoff_Insider` should always be labeled “contemporary source”; Richard Wyckoff should be labeled “historical primary.” Collapsing them creates false provenance.

### Gaps

- Phoenix remains unresolved. No account was guessed from a similar display name.
- The census is current-checkout complete under the stated exclusions, not a claim that every historical branch or deleted file was read.
- The fresh web ledger contains 11 grouped teaching items across 12 top-level URLs (10 X statuses and two videos); the Bojan main/reply is one grouped item. The two Mancini records above are local archive-text reads, not fresh web sources. — [bounded ledger](original_trader_sources.md#which-founding-traders-have-freshly-read-originals)

## What do the historical Wyckoff and Gann sources support?

### Takeaway

Wyckoff's inspected original supports accumulation/distribution, price-volume reading and waiting for range evidence; the familiar A–E/spring/test schematic is supported here only by a modern school summary. Gann's inspected original supports treating elapsed time and multiple chart horizons as analytical dimensions, but not Bull Machine's precise Square-of-9 approximations or Fib day counts.

### Cited Findings

- Richard Wyckoff's 1910 *Studies in Tape Reading* describes accumulation, marking up and distribution, warns that preparation can take weeks or months, and recommends waiting for a narrow-range break when accumulation versus distribution is uncertain. — [University of California scan](https://archive.org/details/studiesintaperea00wyckrich); [public-domain OCR](https://archive.org/stream/studiesintaperea00wyckrich/studiesintaperea00wyckrich_djvu.txt)
- The inspected 1910 text does not contain the modern named “spring” event or an A–E schematic. A contemporary Wyckoff-school tutorial supplies phases A–E, spring/test/SOS/LPS, explicitly notes that a spring is not required, and treats UTAD as optional. — [Modern Wyckoff Analytics summary](https://www.wyckoffanalytics.com/wyckoff-method/)
- W.D. Gann's combined *Truth of the Stock Tape* (1923) / *Wall Street Stock Selector* (1930) scan emphasizes time charts, daily/weekly/monthly/yearly horizons, and the idea that an otherwise similar price formation differs when elapsed time differs. The same volume advertises a proprietary “Master Time Factor” without publishing a reproducible rule. — [Historical scan](https://archive.org/details/in.ernet.dli.2015.166248); [OCR](https://archive.org/stream/in.ernet.dli.2015.166248/2015.166248.Truth-Of-The-Stock-Tape-Study-Of-The-Stock-And-Commodity-Markets-With-Charts-And-Rules-For-Successful-Trading-And-Investing_djvu.txt)
- The contemporary WI Gann post names Square of 9, 360 degrees and calendar cycles, but provides no pivot, price scaling, start date, timezone, tolerance or reset rule. It is secondary interpretation, not a W.D. Gann primary source. — [WI02](https://x.com/Wyckoff_Insider/status/1949399442873786607)
- Bull Machine's richer `GannAnalyzer` calls trailing-range quartiles a simplified Square of Nine and uses Fibonacci `[21,34,55,89]` days; these are explicit project approximations. — [gann.py](../../engine/temporal/gann.py#L39)

### Inferences

- A defensible **Wyckoff backbone** is contextual supply/demand and price-volume sequence inside a persistent range, with patience for evidence. “Parent/child range,” exact event clocks and modern A–E labels are later operationalizations unless separately sourced.
- A defensible **Gann hypothesis** is that elapsed time and horizon may matter. Exact price-angle normalization, Square-of-9 conversion, anniversary/calendar selection and forecast accuracy remain unverified.

### Gaps

- No authenticated original course scan was recovered that proves Richard Wyckoff used the exact modern spring/test/A–E taxonomy inspected here.
- No inspected Gann primary supplied a complete, independently reproducible Square-of-9 or 1x1 trading algorithm. Historical author claims do not establish edge.

## How should foundations, Fib/Gann and the active 17 be reconciled?

### Takeaway

The current 17 are project-defined identities sharing optional Wyckoff context, not 17 teacher-faithful Wyckoff setups. Coarse Fib price/time features are active in narrow places; the rich hidden-extension and Gann-calendar story is mostly alternate/dormant or absent from champion decisions.

### Cited Findings

- The verified active roster is 17 enabled YAML definitions: 14 long, one short and two neutral. Every definition gives Wyckoff a positive fusion weight, but no identity hard-requires a named Wyckoff range/event lifecycle. — [active audit](active_code_fidelity.md#L5)
- The identities commonly detect current state—wick, RSI, FVG, funding or volume—without binding a persistent level/range ID, ordered trigger, expiry and invalidation. The full per-archetype translation table documents each gap. — [17-row mapping](active_code_fidelity.md#L57)
- Active Fib price proximity uses retracements and no hidden extension lifecycle; only `order_block_retest` and `retest_cluster` directly consume temporal fields. No champion YAML consumes Gann time-cluster output. — [price/time audit](active_code_fidelity.md#L28)
- WI names several ratios but supplies none of the required anchor/lifecycle fields; Moneytaur's hidden-level statement supplies no Fib construction. — [WI01](https://x.com/Wyckoff_Insider/status/1949423128217633091); [M03](https://x.com/Moneytaur_/status/1717106022265897019)
- The founding archaeology itself classifies hidden Fibs and Gann/temporal machinery as dormant, while describing surviving Bojan/Wyckoff elements as full, proxy or dormant project translations. — [archaeology](../../docs/knowledge/founding_knowledge_archaeology_2026_07_17.md#L8)

### Inferences

A defensible combined hypothesis would assert:

1. Preserve a causal, immutable HTF range/level and originating timeframe.
2. Represent an ordered event sequence and require direction-appropriate confirmation on the intended execution timeframe.
3. Keep destination/attraction separate from entry permission; attach explicit expiry and invalidation.
4. Treat Fib price zones, Fib time windows and source-versioned Gann calendar windows as separate optional hypotheses with explicit anchors, units and tolerances.
5. Test each addition in isolation after the representation contract is correct.

It would **not** assert that every archetype must be Wyckoff-shaped, that WI/Moneytaur supplied the project's hidden-Fib math, that Gann timing is active or authenticated, that the 17 are a canonical teacher taxonomy, or that any teaching establishes economic edge.

### Gaps

- Source fidelity cannot choose profitable thresholds. Any future claim about expectancy requires preregistered, causal evaluation; this audit did not backtest, certify traders or change the engine.
