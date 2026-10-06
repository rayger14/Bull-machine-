# Teaching corpus inventory and provenance

## Where are the richest original materials, and what provenance survives?

### Takeaway

The checkout is rich in **compiled teaching interpretations**, but poor in locally saved originals. The strongest in-repo provenance is a source index for eight X posts; the strongest actual local text archive is outside the Bull Machine tree in the explicitly linked Mancini project.

### Cited Findings

- The current repo's primary-source ledger is explicitly a paraphrased index, “not a saved full-text archive” or complete curriculum. It records eight dated status URLs: three Wyckoff_Insider, two Bojan, and three Moneytaur. Contemporary Wyckoff_Insider is correctly separated from Richard Wyckoff. — [ledger scope and records](/Users/rayghandchi/Bull%20Machine/Bull-machine-/Bull-machine-/docs/knowledge/trader_primary_source_ledger_2026_09_12.md:6)
- Original-reading depth is uneven: WI01 was partial; WI02 had screenshot-boundary gaps; WI03's main post was read but its illustration only partly visible. Bojan's two short texts were read, but the chart thumbnail was not reconstructed. Moneytaur M01 was expanded with a damaged passage; M02/M03 were search excerpts, not full posts. — [WI records](/Users/rayghandchi/Bull%20Machine/Bull-machine-/Bull-machine-/docs/knowledge/trader_primary_source_ledger_2026_09_12.md:23), [Bojan record](/Users/rayghandchi/Bull%20Machine/Bull-machine-/Bull-machine-/docs/knowledge/trader_primary_source_ledger_2026_09_12.md:69), [Moneytaur records](/Users/rayghandchi/Bull%20Machine/Bull-machine-/Bull-machine-/docs/knowledge/trader_primary_source_ledger_2026_09_12.md:89)
- `wyckoff_audit.md` is the richest local synthesis of WI/Bojan concepts and chart readings, including return-to-zone entry, LPS-anchored stops, opposing-zone partials, and fixed parent ranges. It is not a raw post export: it provides no status URLs for its “six batches,” mixes source observations with project experiments, and itself says sourced objects must be separated from invented magnitudes. — [execution synthesis](/Users/rayghandchi/Bull%20Machine/Bull-machine-/Bull-machine-/docs/knowledge/wyckoff_audit.md:308), [provenance correction](/Users/rayghandchi/Bull%20Machine/Bull-machine-/Bull-machine-/docs/knowledge/wyckoff_audit.md:905)
- A materially richer, explicitly linked corpus exists at `/Users/rayghandchi/Mancini bot/Mancini`: `data/substack/all_posts.json` is a 24 MB extraction with 585 records, 505 nonempty text fields and 80 blanks, dated 2024-06-27 through 2026-05-14 (SHA-256 `a5225fe6f4fa1759fe6af697da7cf9a09a74364d019803988a2b3832decc1d01`). Records retain title/date/slug/text/wordcount but no canonical URL. Acquisition code identifies Adam Mancini's Trade Companion domain and a cookie-authenticated archive/post API. — [parser contract](/Users/rayghandchi/Mancini%20bot/Mancini/backtest/parse_all_mancini_posts.py:1), [source/API](/Users/rayghandchi/Mancini%20bot/Mancini/backtest/parse_all_mancini_posts.py:31)
- Mancini derivatives are not independent evidence: `analysis_report.md` says three agents summarized 500 paywalled posts; `short_excerpts*.json` are selections from the archive; `data/mancini_levels/` has 400 parsed daily JSONs; training has 410 regex-price files, 410 LLM plans, 315 engine-level files and model/trade artifacts. The report is a useful index, not an original. — [summary provenance](/Users/rayghandchi/Mancini%20bot/Mancini/data/substack/analysis_report.md:1), [ignored local artifacts](/Users/rayghandchi/Mancini%20bot/Mancini/.gitignore:9)
- Bull Machine already points to that external project as its Mancini source, but reduces it to a transfer map and one A/B. — [transfer map](/Users/rayghandchi/Bull%20Machine/Bull-machine-/Bull-machine-/docs/knowledge/mancini_transfer_map_2026_08_20.md:1)

### Inferences

- Provenance strength is: Mancini extracted article text (strongest, but noisy/local-only) > eight URL/date-linked X observations > `wyckoff_audit` and archaeology summaries > code/config/training labels. A named file or trader label is not an original source.
- Mancini slugs permit reconstructing likely article URLs against the stored base domain, but this audit did not browse or validate them.

### Gaps

- No raw X export, screenshot, transcript, or chart image is stored for Moneytaur, WI, or Bojan. Intraday timezone and exact chart-price associations are mostly unrecoverable from current files.
- Mancini extraction receipts, download timestamp, content hashes per article, and canonical URLs are absent from `all_posts.json`; some records contain page boilerplate/embedded preload text.

## Do existing teachings cover context, levels, sequence, entries, invalidation, management, and structure within structure?

### Takeaway

Yes at the **concept/synthesis** level, especially WI/Bojan plus Mancini; no at complete original-source fidelity. All requested dimensions appear somewhere, but no single authenticated corpus covers them end to end for every named trader or all 17 archetypes.

### Cited Findings

- Moneytaur M01 supports identified levels, horizon-specific confirmation, a planned destination and avoidance when targets are unclear; it does not supply a universal stop formula or threshold. — [Moneytaur access note](/Users/rayghandchi/Bull%20Machine/Bull-machine-/Bull-machine-/docs/knowledge/moneytaur_primary_sources_2026_09_12.md:17)
- WI03 explicitly separates higher-timeframe analysis, middle-timeframe context and lower-timeframe execution. The ledger's safe inference is to separate thesis, trigger, invalidation and destination. — [WI03](/Users/rayghandchi/Bull%20Machine/Bull-machine-/Bull-machine-/docs/knowledge/trader_primary_source_ledger_2026_09_12.md:56), [inference boundary](/Users/rayghandchi/Bull%20Machine/Bull-machine-/Bull-machine-/docs/knowledge/trader_primary_source_ledger_2026_09_12.md:105)
- The compiled WI/Bojan synthesis covers ordered events (model/LPS then MSS then return-to-zone), structural stops, partials/runners, zone conversion after close-beyond invalidation, fixed HTF ranges and parent/child timeframe roles. — [entry/management](/Users/rayghandchi/Bull%20Machine/Bull-machine-/Bull-machine-/docs/knowledge/wyckoff_audit.md:317), [Bojan lifecycle](/Users/rayghandchi/Bull%20Machine/Bull-machine-/Bull-machine-/docs/knowledge/wyckoff_audit.md:338), [HTF states](/Users/rayghandchi/Bull%20Machine/Bull-machine-/Bull-machine-/docs/knowledge/wyckoff_audit.md:400)
- Mancini's raw archive is especially strong for specific predeclared levels and day-by-day context. Its derived report maps significant lows, sweep/recovery/acceptance, stop below the complete sweep, level-to-level scale-outs and a runner; because the map is agent-produced, exact claims should be checked against dated article records before transfer. — [setup/entry/invalidation map](/Users/rayghandchi/Mancini%20bot/Mancini/data/substack/analysis_report.md:12), [management map](/Users/rayghandchi/Mancini%20bot/Mancini/data/substack/analysis_report.md:57)
- The founding inventory names Bojan, Moneytaur, Wyckoff_Insider, IamZeroIka, Crypto Chase and Phoenix, but is code archaeology/synthesis rather than an original teaching archive. — [founding inventory](/Users/rayghandchi/Bull%20Machine/Bull-machine-/Bull-machine-/docs/knowledge/founding_knowledge_archaeology_2026_07_17.md:1)

### Inferences

- There is enough locally saved teaching to define the *schema* requested by the user: source-versioned parent context, named level/zone, ordered available-at events, entry, invalidation, destination and management. There is not enough original evidence to claim every numerical choice or trader attribution is faithful.
- Mancini should be consulted directly for level/reclaim/management examples before relying on Bull Machine's one-page transfer summary.

### Gaps

- Exact original-source mappings for IamZeroIka, CryptoChase and Phoenix are absent. The ledger explicitly leaves them unaudited. — [open coverage](/Users/rayghandchi/Bull%20Machine/Bull-machine-/Bull-machine-/docs/knowledge/trader_primary_source_ledger_2026_09_12.md:123)
- Moneytaur remains thin; additional WI range/phi posts were located but not fully read. No primary Richard Wyckoff or W.D. Gann historical text was examined. Bibliographic mentions are not locally read books. — [open coverage](/Users/rayghandchi/Bull%20Machine/Bull-machine-/Bull-machine-/docs/knowledge/trader_primary_source_ledger_2026_09_12.md:123)

## What duplicates, absent media, local-only resources, and audit limits matter?

### Takeaway

Sibling checkouts add little independent teaching evidence: core knowledge files are mostly byte-identical copies or older variants. No standalone book/media files were found in the scanned formats; ignored Mancini text is the major local-only exception.

### Cited Findings

- Scanned roots: current nested checkout (3,955 files); its parent checkout census (11,234) includes that nested tree and is therefore not additive; separate roots `Bull-machine--1` (1,592), `one-strategy` (1,663), `risk-neutral-deploy` (1,588), and `wyckoff-campaign-v2` (1,595). Commands used `rg --files -uu`, targeted `rg`, `find`, `jq`, `shasum`, and `git check-ignore`; excluded `.git`, venvs, `node_modules`, caches, worktrees and bulk parquet/log/market data.
- Hashing shows `founding_knowledge_archaeology` and `trader_knowledge_audit` are identical across six visible copies; `trader_knowledge_standdown_sweep` is identical across three; current `wyckoff_audit` is identical in current/parent/`Bull-machine--1`, while three sibling variants differ. Copies are not corroborating sources.
- `one-strategy/idea_lab/` contains source-inspired `structural_range.py`, `htf_pivots.py`, `bojan_detector.py` and experiments, but no raw teaching archive. Current `ARCHETYPE_KNOWLEDGE.md` is explicitly synthesized backtest knowledge, not trader source material. — [synthesis label](/Users/rayghandchi/Bull%20Machine/Bull-machine-/Bull-machine-/ARCHETYPE_KNOWLEDGE.md:1)
- Across `/Users/rayghandchi/Bull Machine` there were zero PDF/EPUB/MOBI, local image, audio or video files after exclusions. The targeted Mancini root likewise had none. Fifteen image-URL strings occur inside its extracted page text, but no structured/local chart assets. `chart_logs` is a symlink to `data`; symlinks were inventoried but not followed to avoid duplicate traversal.

### Inferences

- “Entire repo” should not be reported as every file semantically read. This was a filename/metadata census, name/URL/content search, hash deduplication and targeted reading of the richest candidates.
- Before a fidelity fix, the teaching corpus is **not complete**: preserve the compiled synthesis, but build a per-claim source ledger and promote only original-linked claims. Mancini's ignored archive should be treated as a local dependency and backed by receipts/hashes if used.

### Gaps

- No web browsing, downloading, OCR, transcription or dependency installation was performed. Symlink targets were not recursively followed; unrelated personal folders were not searched. The only outside-root expansion was the exact Mancini project already cited by Bull Machine.
- Git history/other branches were not exhaustively inventoried; current checkout and visible sibling worktrees were the bounded scope. Older differing `wyckoff_audit` variants may preserve historical synthesis, not additional originals.
