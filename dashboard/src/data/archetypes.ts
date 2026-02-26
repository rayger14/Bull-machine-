export interface ArchetypeInfo {
  name: string;
  proven: boolean;
  calibrated: boolean;
  pf: string | null;
  dir: string;
  trades?: string;
  desc: string;
  explanation: string;
  whyItWorks: string;
  weights: { wyckoff: number; liquidity: number; momentum: number; smc: number };
  gates: string[];
  code?: string;
}

export const ARCHETYPES: Record<string, ArchetypeInfo> = {
  trap_within_trend: {
    name: 'Trap Within Trend (H)',
    proven: true,
    calibrated: true,
    pf: '1.52',
    dir: 'long',
    trades: '71 trades (2020-2024)',
    desc: 'False breakdown in strong uptrend. Price briefly breaks below support, trapping shorts, then reverses sharply. PnL: $18K.',
    explanation:
      'Detects false breakdowns within established uptrends where price briefly pierces support, trapping shorts, then reverses violently. Requires strong ADX (trending), low liquidity (thin books amplify reversals), and a wick against the trend confirming rejection. BOS flag must confirm structural break.',
    whyItWorks:
      'Short sellers enter on the support break, but institutional buyers absorb selling pressure at lower prices. When the trap is sprung, short covering + new longs create explosive moves.',
    weights: { wyckoff: 0.35, liquidity: 0.35, momentum: 0.15, smc: 0.15 },
    gates: [
      'ADX >= adx_threshold (tuned: 15.0)',
      'liquidity_score < liquidity_threshold (tuned: 0.50)',
      'wick > wick_multiplier * body (default 2.0x)',
      'BOS flag != 0 (bullish or bearish BOS present)',
      'fusion_score >= dynamic threshold',
    ],
  },
  liquidity_sweep: {
    name: 'Liquidity Sweep (G)',
    proven: true,
    calibrated: true,
    pf: '1.89',
    dir: 'long',
    trades: '138 trades (2020-2024)',
    desc: 'Wyckoff Phase B re-accumulation. Professional money accumulates during sideways action while trapping retail traders.',
    explanation:
      'Identifies Wyckoff Phase B re-accumulation patterns where professional money is building positions during apparent sideways price action. Requires confirmed Phase B with BOMS strength showing institutional order flow and elevated volume confirming accumulation activity.',
    whyItWorks:
      'During Phase B, smart money creates sideways chop to shake out weak hands. The BOMS strength signal confirms genuine institutional accumulation rather than distribution, providing early entry before the markup phase.',
    weights: { wyckoff: 0.30, liquidity: 0.40, momentum: 0.15, smc: 0.15 },
    gates: [
      'wyckoff_phase_abc == "B" (Phase B re-accumulation)',
      'tf1d_boms_strength >= 0.30 (institutional order flow)',
      'volume_z >= 1.0 (elevated accumulation volume)',
      'fusion_score >= fusion_threshold (default 0.40)',
    ],
  },
  wick_trap: {
    name: 'Wick Trap (K)',
    proven: true,
    calibrated: true,
    pf: '1.86',
    dir: 'long',
    trades: '313 trades (2020-2024)',
    desc: 'Bullish wick rejection reversal. Large lower wick = buyers aggressively rejecting lower prices. Top PnL generator ($118K). Dominant archetype (~34%).',
    explanation:
      'Fires when a candle produces a large lower wick (wick_lower_ratio >= 0.60) while RSI is oversold (RSI <= 40). The wick signals aggressive buyer absorption at lower prices. Domain engines then apply Wyckoff, SMC, and temporal boosts/vetoes before the fusion threshold gate.',
    whyItWorks:
      'Large lower wicks represent failed selling attempts where institutional buyers step in aggressively. Combined with oversold RSI, this creates a high-probability mean reversion entry as trapped shorts are forced to cover.',
    weights: { wyckoff: 0.20, liquidity: 0.40, momentum: 0.15, smc: 0.25 },
    gates: [
      'wick_lower_ratio >= wick_lower_threshold (default 0.60)',
      'RSI <= rsi_threshold (default 40)',
      'fusion_score >= fusion_threshold (default 0.40)',
      'regime_confidence >= adaptive threshold',
    ],
  },
  liquidity_compression: {
    name: 'Liquidity Compression (E)',
    proven: false,
    calibrated: false,
    pf: '0.95',
    dir: 'long',
    desc: 'Compression before expansion. Low ATR + narrow range + building liquidity signals a coiled spring. DISABLED -- net loser at any leverage.',
    explanation:
      'Detects low-volatility compression zones where ATR percentile drops below 0.25, range narrows to less than 0.5x ATR, and liquidity is within a moderate band. These coiled springs tend to produce explosive breakouts, but the direction is uncertain, making this archetype unreliable.',
    whyItWorks:
      'Bollinger Band squeezes and ATR compression precede volatility expansions. The challenge is that breakout direction is difficult to predict, leading to frequent stop-outs before the real move.',
    weights: { wyckoff: 0.20, liquidity: 0.50, momentum: 0.15, smc: 0.15 },
    gates: [
      'ATR percentile < 0.25 (low volatility)',
      'range/ATR < 0.50 (narrow range)',
      'liquidity_score in [liquidity_min, liquidity_max] (default [0.25, 0.70])',
      'fusion_score >= fusion_threshold (default 0.35)',
    ],
  },
  retest_cluster: {
    name: 'Retest Cluster (L)',
    proven: true,
    calibrated: true,
    pf: '1.36',
    dir: 'long',
    trades: '217 trades (2020-2024)',
    desc: 'Fakeout then genuine structural reversal. Previous bar had bearish BOS (trap), current bar has bullish BOS (real move). Lower threshold (0.12).',
    explanation:
      'Detects fakeout-then-real-move sequences: the previous bar produced a bearish BOS (trapping longs), and the current bar produces a bullish BOS (the genuine reversal). This two-bar pattern captures the moment when failed breakdowns snap back with conviction.',
    whyItWorks:
      'Failed bearish breakdowns trap shorts and exhaust sellers. When the bullish BOS fires on the next bar, it confirms the trap was complete and initiates a short-covering cascade that propels price higher.',
    weights: { wyckoff: 0.25, liquidity: 0.30, momentum: 0.20, smc: 0.25 },
    gates: [
      'tf1h_bos_bullish == true (current bar)',
      'tf1h_bos_bearish == true (previous bar -- the fakeout)',
      'fusion_score >= fusion_threshold (default 0.40)',
      'regime_confidence >= adaptive threshold',
    ],
  },
  liquidity_vacuum: {
    name: 'Liquidity Vacuum (S1)',
    proven: true,
    calibrated: true,
    pf: 'Inf',
    dir: 'long',
    trades: '12 trades (2020-2024)',
    desc: 'Crisis capitulation reversal at panic lows. Multi-bar exhaustion detection catches violent bounces when sellers exhaust themselves.',
    explanation:
      'Detects capitulation events where orderbook liquidity evaporates during sell-offs, creating air pockets for explosive bounces. V2 uses multi-bar detection: capitulation depth >= -20% from 30d high, crisis composite >= 0.40, and at least one of volume climax (3-bar) > 0.25 OR wick exhaustion (3-bar) > 0.30.',
    whyItWorks:
      'During capitulation, sellers exhaust themselves and bids evaporate. When the vacuum fills, there is zero resistance to the upside, causing violent short-covering bounces. Real capitulation events are messy (signals span 2-3 bars), which is why multi-bar detection is critical.',
    weights: { wyckoff: 0.35, liquidity: 0.45, momentum: 0.10, smc: 0.10 },
    gates: [
      'capitulation_depth >= capitulation_depth_max (default -0.20)',
      'crisis_composite >= crisis_composite_min (default 0.40)',
      'volume_climax_3b >= 0.25 OR wick_exhaustion_3b >= 0.30',
      'regime_confidence >= adaptive threshold',
      'fusion_score >= fusion_threshold (default 0.30)',
    ],
  },
  failed_continuation: {
    name: 'Failed Continuation (D)',
    proven: false,
    calibrated: false,
    pf: null,
    dir: 'long',
    desc: 'Failed bearish continuation reversal. FVG present + weak RSI + falling ADX signals bearish exhaustion. Gate fix unlocked this archetype.',
    explanation:
      'Fires when a fair value gap (FVG) is present on the 1H timeframe, RSI is below 50 (weak but not extreme), and ADX is falling compared to the previous bar (trend losing strength). The falling ADX signals the bearish continuation is failing, creating a reversal opportunity.',
    whyItWorks:
      'When a bearish FVG forms but ADX starts declining, it indicates the selling momentum is exhausting. Bears expected continuation but the trend is weakening, creating a gap that gets filled as sellers lose conviction.',
    weights: { wyckoff: 0.25, liquidity: 0.25, momentum: 0.35, smc: 0.15 },
    gates: [
      'tf1h_fvg_present == 1 (fair value gap exists)',
      'RSI < rsi_max (default 50)',
      'ADX falling (current ADX < previous bar ADX)',
      'liquidity_score >= 0.35',
      'fusion_score >= fusion_threshold (default 0.42)',
    ],
  },
  spring: {
    name: 'Spring / UTAD (A)',
    proven: true,
    calibrated: true,
    pf: '0.87',
    dir: 'long',
    trades: '35 trades (2020-2024)',
    desc: 'Wyckoff spring detection with multi-path logic. Primary: Wyckoff spring events. Secondary: PTI trap. Tertiary: synthetic wick + volume + displacement.',
    explanation:
      'Multi-path spring detection: Path 1 fires on Wyckoff spring_a/spring_b events (highest confidence). Path 2 uses PTI trap type detection (spring/utad with PTI score >= 0.30). Path 3 synthesizes from wick rejection + volume climax + displacement >= ATR multiplier. All paths receive domain engine boosts from Wyckoff accumulation signals.',
    whyItWorks:
      'Wyckoff springs are the classic institutional accumulation pattern -- smart money shakes out weak hands below support, then marks up price. The multi-path approach catches springs whether they are clean events or messy real-world capitulations.',
    weights: { wyckoff: 0.60, liquidity: 0.20, momentum: 0.10, smc: 0.10 },
    gates: [
      'wyckoff_spring_a OR wyckoff_spring_b OR (wyckoff_lps + wick >= 0.60 in Phase C)',
      'OR pti_trap_type in [spring, utad] AND pti_score >= 0.30',
      'OR wick_lower >= 0.60 AND volume_climax AND displacement >= disp_multiplier * ATR',
      'fusion_score >= fusion_threshold (default 0.33)',
    ],
  },
  order_block_retest: {
    name: 'Order Block Retest (B)',
    proven: false,
    calibrated: false,
    pf: null,
    dir: 'long',
    desc: 'SMC order block retest with BOMS strength + Wyckoff + near BOS zone. Disabled in RISK_ON regime (reversal pattern fails in trends).',
    explanation:
      'Detects retests of smart money order blocks using bullish BOS as primary trigger, BOMS strength >= 0.30 for institutional confirmation, and Wyckoff score >= 0.35. In crisis markets, BOMS is made optional (BOS + Wyckoff sufficient). Vetoed in RISK_ON regime where reversal patterns fail.',
    whyItWorks:
      'Order blocks are zones where institutional traders placed large orders. When price retests these zones, the same institutions defend their positions, creating reliable support bounces. The BOS confirms structure was broken in their favor.',
    weights: { wyckoff: 0.30, liquidity: 0.15, momentum: 0.15, smc: 0.40 },
    gates: [
      'bos_bullish == true (structural break confirmed)',
      'boms_strength >= boms_strength_min (default 0.30, optional in crisis)',
      'wyckoff_score >= wyckoff_min (default 0.35)',
      'regime != bull (reversal fails in strong trends)',
      'fusion_score >= fusion_threshold (default 0.374)',
    ],
  },
  fvg_continuation: {
    name: 'FVG Continuation (C)',
    proven: true,
    calibrated: true,
    pf: '1.34',
    dir: 'long',
    trades: '51 trades (2020-2024)',
    desc: 'BOS/CHOCH reversal pattern. Bullish BOS triggers entry, CHOCH adds confidence bonus, wick rejection adds further bonus.',
    explanation:
      'Fires when bullish BOS is detected on the 1H timeframe. Base score is 0.35 for BOS alone, with a +0.10 bonus if CHOCH (change of character) confirms, and +0.20 bonus for wick rejection (wick_lower_ratio >= 0.55). Regime penalties reduce confidence in crisis (-50%) and bear (-25%).',
    whyItWorks:
      'A bullish Break of Structure signals the end of bearish control. When CHOCH confirms the character change and wicks show buyer absorption, it creates a high-confluence reversal setup where the structural shift is validated from multiple angles.',
    weights: { wyckoff: 0.15, liquidity: 0.20, momentum: 0.30, smc: 0.35 },
    gates: [
      'tf1h_bos_bullish == true (required)',
      'CHOCH flag adds +0.10 bonus (optional)',
      'wick_lower_ratio >= 0.55 adds +0.20 bonus (optional)',
      'fusion_score >= fusion_threshold (default 0.40)',
      'regime penalty: crisis -50%, bear -25%',
    ],
  },
  exhaustion_reversal: {
    name: 'Exhaustion Reversal (F)',
    proven: false,
    calibrated: false,
    pf: null,
    dir: 'long',
    desc: 'Momentum exhaustion at extremes. Extreme RSI + high ATR percentile + volume spike + dropping liquidity.',
    explanation:
      'Detects momentum exhaustion when RSI hits extremes (> 78 or < 22), ATR percentile exceeds 0.90 (high volatility), volume z-score exceeds 1.0, and liquidity is either dropping from previous bar or below 0.40. These conditions signal that the current move has overextended.',
    whyItWorks:
      'Extreme RSI with volume spikes at high volatility indicates the final capitulation wave. When liquidity drops simultaneously, there are no more sellers/buyers left to sustain the move, creating a natural reversal point.',
    weights: { wyckoff: 0.25, liquidity: 0.20, momentum: 0.45, smc: 0.10 },
    gates: [
      'RSI > 78 OR RSI < 22 (extreme reading)',
      'ATR percentile > 0.90 (high volatility)',
      'volume_zscore > 1.0 (volume spike)',
      'liquidity dropping OR liquidity < 0.40',
      'fusion_score >= fusion_threshold (default 0.38)',
    ],
  },
  confluence_breakout: {
    name: 'Confluence Breakout (M)',
    proven: false,
    calibrated: false,
    pf: null,
    dir: 'long',
    desc: 'Multi-timeframe coil breakout. 4H coil score detects compression, 4H BOS confirms breakout direction.',
    explanation:
      'Uses 4H coil score (ATR + BB squeeze metric) to detect volatility compression, then waits for a 4H bullish BOS to confirm breakout direction. Bonus scoring for multi-timeframe coil alignment (1H + 1D also coiling). Falls back to legacy ATR-based detection if coil features are unavailable.',
    whyItWorks:
      'Volatility compression (Bollinger Band squeeze + low ATR) creates potential energy like a coiled spring. When the breakout occurs with structural confirmation (BOS), the compressed energy releases in a sustained directional move.',
    weights: { wyckoff: 0.30, liquidity: 0.25, momentum: 0.20, smc: 0.25 },
    gates: [
      'tf4h_coil_score >= min_coil_score (default 0.55)',
      'tf4h_coil_breakout == true (if strict mode)',
      'tf4h_bos_bullish == true (directional confirmation)',
      'fusion_score >= fusion_threshold (default 0.35)',
    ],
  },
  whipsaw: {
    name: 'Whipsaw (S3)',
    proven: false,
    calibrated: false,
    pf: null,
    dir: 'neutral',
    desc: 'Distribution climax short. Volume climax + overbought RSI signals exhaustion at distribution highs.',
    explanation:
      'Fires when volume_climax_last_3b >= 1.0 (multi-bar volume exhaustion) and RSI >= 70 (overbought). This combination detects distribution climaxes where buying volume spikes at highs but fails to push price further, signaling smart money distribution.',
    whyItWorks:
      'Volume climaxes at overbought levels represent the final wave of retail buying while institutions distribute. The simultaneous exhaustion of buying power and institutional selling creates a reliable short-term reversal point.',
    weights: { wyckoff: 0.25, liquidity: 0.30, momentum: 0.30, smc: 0.15 },
    gates: [
      'volume_climax_last_3b >= 1.0 (multi-bar climax)',
      'RSI >= 70 (overbought exhaustion)',
      'fusion_score >= fusion_threshold (default 0.40)',
      'regime_confidence >= adaptive threshold',
    ],
  },
  funding_divergence: {
    name: 'Funding Divergence (S4)',
    proven: true,
    calibrated: true,
    pf: '2.83',
    dir: 'long',
    trades: '19 trades (2020-2024)',
    desc: 'Short squeeze from extreme negative funding. Overcrowded shorts + price resilience = violent squeeze up.',
    explanation:
      'Detects overcrowded short positions via extreme negative funding (funding_Z < -1.2), combined with price resilience (price not falling despite bearish funding) and low liquidity (< 0.30). Includes SMC veto: aborts if 4H BOS is bearish (institutional sellers active). Optional volume quiet bonus for coiled spring effect.',
    whyItWorks:
      'Shorts paying high negative funding is unsustainable. When price shows resilience despite extreme bearish funding, it signals divergence -- shorts are trapped. Thin liquidity amplifies the squeeze when shorts panic-cover.',
    weights: { wyckoff: 0.20, liquidity: 0.50, momentum: 0.20, smc: 0.10 },
    gates: [
      'funding_Z < funding_z_max (default -1.2, extreme negative)',
      'liquidity_score < liquidity_max (default 0.30)',
      'tf4h_bos_bearish == false (SMC veto: no institutional selling)',
      'price_resilience >= resilience_min (default 0.50, if available)',
      'fusion_score >= fusion_threshold (default 0.40)',
    ],
  },
  long_squeeze: {
    name: 'Long Squeeze (S5)',
    proven: true,
    calibrated: true,
    pf: '0.13',
    dir: 'short',
    trades: '25 trades (2020-2024)',
    desc: 'Short archetype. Overcrowded longs + exhaustion = cascade down. Positive funding extreme + overbought RSI + thin liquidity.',
    explanation:
      'Detects overcrowded long positions via high positive funding (funding_Z > 1.2), RSI overbought (> 70), and low liquidity (< 0.25). OI spike is optional bonus. Graceful degradation: fires with 3 core components when OI data is unavailable (2022-2023).',
    whyItWorks:
      'Longs paying high positive funding is unsustainable. When RSI is overbought with thin liquidity, there are no buyers left and the orderbook has no bids to catch the fall. Any selling triggers a cascade as overleveraged longs get liquidated.',
    weights: { wyckoff: 0.20, liquidity: 0.50, momentum: 0.20, smc: 0.10 },
    gates: [
      'funding_Z >= funding_z_min (default 1.2, positive extreme)',
      'RSI >= rsi_min (default 70, overbought)',
      'liquidity_score < liquidity_max (default 0.25)',
      'OI change > threshold (optional bonus if available)',
      'fusion_score >= fusion_threshold (default 0.35)',
    ],
  },
  volume_fade_chop: {
    name: 'Volume Fade Chop (S8)',
    proven: false,
    calibrated: false,
    pf: null,
    dir: 'neutral',
    desc: 'Low-volume fade scalper for neutral/choppy conditions. Volume fade + low volatility = chop regime.',
    explanation:
      'Fires when volume z-score drops below -0.5 (volume fade) AND ATR is less than 0.6% of price (low volatility chop). Falls back to Bollinger Band width < 3% as volatility proxy if ATR is unavailable. Captures range-bound mean reversion opportunities in dead markets.',
    whyItWorks:
      'When volume dries up and volatility compresses, markets enter chop mode where price oscillates within a range. Fading moves in this environment exploits the mean-reverting nature of low-conviction price action.',
    weights: { wyckoff: 0.15, liquidity: 0.35, momentum: 0.40, smc: 0.10 },
    gates: [
      'volume_zscore <= -0.5 (volume fade)',
      'ATR < 0.6% of close price (low volatility)',
      'OR bb_width < 0.03 (Bollinger squeeze fallback)',
      'fusion_score >= fusion_threshold (default 0.40)',
    ],
  },
  oi_divergence: {
    name: 'OI Divergence (S9)',
    proven: false,
    calibrated: false,
    pf: '0.04',
    dir: 'long',
    trades: '2 trades (shadow mode)',
    desc: 'Open interest / price divergence detector. Price rises but OI falls = hollow move, institutional exit. Shadow mode — only 2 trades so far.',
    explanation:
      'Detects bearish divergence where price is rising but Open Interest is declining, signaling that institutions are closing positions into strength. Requires Binance Futures OI data (available since 2022). Currently in shadow mode for data collection.',
    whyItWorks:
      'When OI declines while price rises, it means new money is NOT entering — the rally is driven by short covering, not conviction. These hollow rallies tend to reverse once the covering exhausts.',
    weights: { wyckoff: 0.15, liquidity: 0.25, momentum: 0.25, smc: 0.35 },
    gates: [
      'oi_price_divergence == -1 (bearish divergence detected)',
      'Binance Futures OI data available (post-2022)',
      'fusion_score >= dynamic threshold',
    ],
  },
};
