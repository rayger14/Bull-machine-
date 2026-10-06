# Primary-source reconciliation: IamZeroIka and Crypto Chase

Bounded pass completed October 1, 2026. Direct X profile opens returned 403, and direct status opens were cache misses. No signed-in browser control was available. Third-party mirrors below are used only to locate handles, dates, status IDs, and unverified copied text—not as rule authority. Charts, replies, video content, and attachment details remain uninspected.

## Identity resolution

### IamZeroIka

- **Repo evidence:** the founding inventory names `@IamZeroIka` ([founding archaeology, line 4](../../docs/knowledge/founding_knowledge_archaeology_2026_07_17.md#L4)); the later source ledger says its identity and source mapping still required verification ([ledger, lines 123–130](../../docs/knowledge/trader_primary_source_ledger_2026_09_12.md#L123)).
- **Canonical profile:** [x.com/IamZeroIka](https://x.com/IamZeroIka), **high confidence**. The repo spelling agrees with the profile link exposed by the indexed author page and with Zero Ika's author/business site identifying its founder as `@IamZeroIka` ([Dojo Trading](https://dojo-trading.com/features/telegram)). The profile itself was not readable now.

### Crypto Chase

- **Repo evidence:** the same inventory uses the display name “Crypto Chase” ([line 5](../../docs/knowledge/founding_knowledge_archaeology_2026_07_17.md#L5)); older implementation notes attribute liquidity-sweep/fake-break logic to that name ([SPY report, lines 39–42](../../docs/archive/SPY_ORDERFLOW_FINAL_REPORT.md#L39)). The ledger still classified the identity as unresolved.
- **Canonical profile:** [x.com/Crypto_Chase](https://x.com/Crypto_Chase), **high confidence**. Crypto Chase's author-controlled [Linktree](https://linktr.ee/Crypto_Chase) labels him “Trader and educator” and its Twitter button resolves to `twitter.com/Crypto_Chase`; it also links the matching YouTube curriculum. The user-spelled `@cryptochae` produced no corroborating local or author-controlled mapping and is best treated as a typo, not a second identity.

## Bounded teaching records

### IamZeroIka (two located originals; original pages unreadable now)

1. **Trading-plan thread — February 10, 2023.** Original: [x.com/IamZeroIka/status/1623715383218929666](https://x.com/IamZeroIka/status/1623715383218929666). A [third-party unroll](https://en.rattibha.com/thread/1623715383218929666) locates the 25-part thread and copies text describing a personal, mainly long/passive process: assess HH/HL versus LH/LL, daily 9/21 EMA posture, leading sectors, relative strength, accumulation areas, and closes above prior highs/resistance before a risk-managed trade. **Access class:** PARTIAL locator/copied-text evidence; original, charts, replies, and edit state uninspected. **Supported only if authenticated:** a top-down selection workflow and close-based structural confirmation, expressly personalized rather than universal. **Missing:** causal swing algorithm, venue, tolerance, setup timeframe beyond examples, entry clock, numeric invalidation/stop, sizing, and management schedule.

2. **Price-action-over-indicators post — April 15, 2024.** Original: [x.com/IamZeroIka/status/1779921797434982462](https://x.com/IamZeroIka/status/1779921797434982462). [Thread Reader](https://threadreaderapp.com/thread/1779921797434982462.html) identifies the date/status and copies a single long post arguing that price action plus fundamental context should form the thesis before lagging indicators, which may support rather than originate it. **Access class:** PARTIAL locator/copied-text evidence; original and any media uninspected. **Supported only if authenticated:** indicator outputs are subordinate context, not standalone triggers. **Missing:** operational price-action definition, level anchors, exceptions, entry/invalidation/management, and any evidence for the claimed proportions or efficacy.

These located texts do **not** authenticate the repo's specific “1/3 body close” or ten-bar CVD-slope claims. Those remain project attributions in code ([LCA, lines 83–120](../../bull_machine/modules/orderflow/lca.py#L83), [lines 145–190](../../bull_machine/modules/orderflow/lca.py#L145)) and planning notes referencing an unavailable “post:39” ([V161 plan, lines 303–320](../../docs/archive/V161_BUILD_PLAN.md#L303)). Exact original status/media for that claim is still needed.

### Crypto Chase (two author-owned videos; content inaccessible now)

1. **“Candle Closes, ‘Acceptance’, and Invalidation” — April 27, 2023.** Original author-channel URL: [YouTube](https://www.youtube.com/watch?v=MRMikMIlOwM). Crypto Chase's [Linktree](https://linktr.ee/Crypto_Chase) links this exact video. **Access class:** METADATA ONLY; title/date and authorship path visible, audiovisual content/transcript/charts uninspected. It establishes that these topics are in his curriculum, but supports no exact acceptance threshold, anchor, exception, or invalidation formula.

2. **“Liquidity and the Daily Timeframe” — July 22, 2022.** Original author-channel URL: [YouTube](https://www.youtube.com/watch?v=ouxW_yUISgs). The official Linktree links it and indexed YouTube metadata supplies the date. **Access class:** METADATA ONLY; video, transcript, charts, and comments uninspected. It establishes a liquidity/daily-timeframe teaching topic, not the repo's two-bar sweep/recovery rule, 40% close-position threshold, or `1.4×` volume condition.

Accordingly, code comments calling a prior-bar break/current-bar recovery “Crypto Chase logic” ([LCA, lines 88–100](../../bull_machine/modules/orderflow/lca.py#L88)) and rapid reversal “Crypto Chase style” ([PO3, lines 63–86](../../bull_machine/strategy/po3_detection.py#L63)) remain unauthenticated project translations. The exact need is a readable author transcript or original post/video segment with timestamps, chart anchors, confirmation clock, invalidation, and management exceptions.

## Effect on the local audit

The identity gap is materially narrowed: both canonical handles are now high-confidence, and `@cryptochae` should map to `@Crypto_Chase`. The teaching-fidelity gap is **not closed**. Two IamZeroIka originals were located but not read at source; they provide only partial discovery text and do not substantiate the repo's numerical rules. Crypto Chase's author-owned curriculum is verified, but its relevant content was inaccessible beyond metadata. The prior audit's warning therefore stands: preserve compiled hypotheses as project interpretations, not author-written doctrine, and make no edge claim from these records.
