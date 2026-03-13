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
  gateMode: 'hard' | 'soft';
  code?: string;
}

export const ARCHETYPES: Record<string, ArchetypeInfo> = {
  wick_trap: {
    name: 'Wick Trap (K)',
    proven: true,
    calibrated: true,
    pf: '1.59',
    dir: 'long',
    trades: '109 trades (2020-2024)',
    desc: 'Bullish wick rejection reversal. Large lower wick = buyers aggressively rejecting lower prices. PnL: $22K.',
    explanation:
      'Fires when a candle produces a wick anomaly (lower wick > 35% of range) with at-or-above median volume. Hard gate mode means ALL gates must pass. The wick signals aggressive buyer absorption at lower prices.',
    whyItWorks:
      'Large lower wicks represent failed selling attempts where institutional buyers step in aggressively. Combined with volume confirmation, this creates a high-probability mean reversion entry as trapped shorts are forced to cover.',
    weights: { wyckoff: 0.20, liquidity: 0.40, momentum: 0.15, smc: 0.25 },
    gateMode: 'hard',
    gates: [
      'derived:wick_anomaly == true (lower wick > 35% of range)',
      'volume_zscore >= 0.0 (at least median volume)',
    ],
  },
  liquidity_sweep: {
    name: 'Liquidity Sweep (G)',
    proven: true,
    calibrated: true,
    pf: '2.21',
    dir: 'long',
    trades: '98 trades (2020-2024)',
    desc: 'Liquidity grab reversal. Sweep below support with wick rejection and rising liquidity from oversold. PnL: $27K.',
    explanation:
      'Identifies liquidity sweeps where price wicks below key levels to grab stops, then reclaims. Requires minimum liquidity score and directional wick validation (lower wick dominant for long). Soft gates penalize fusion score on failure.',
    whyItWorks:
      'Market makers sweep stop-loss clusters below support to fill large buy orders at better prices. The wick rejection and liquidity reclaim signal the sweep is complete and price will reverse.',
    weights: { wyckoff: 0.30, liquidity: 0.40, momentum: 0.15, smc: 0.15 },
    gateMode: 'soft',
    gates: [
      'liquidity_score >= 0.2 (minimum liquidity for sweep)',
      'wick_lower_ratio >= 1.3 (sweep direction — lower wick > body)',
    ],
  },
  retest_cluster: {
    name: 'Retest Cluster (L)',
    proven: true,
    calibrated: true,
    pf: '1.92',
    dir: 'long',
    trades: '117 trades (2020-2024)',
    desc: 'Fakeout then genuine structural reversal with temporal confluence. PnL: $23K.',
    explanation:
      'Detects fakeout-then-real-move sequences with volume confirmation and temporal confluence. Temporal confluence score (Optuna-tuned to 0.45) filters low-quality clusters. RSI extreme gate adds mean-reversion timing.',
    whyItWorks:
      'Failed breakdowns trap shorts and exhaust sellers. Volume spike + temporal confluence validates the setup is real, not noise. At 0.45 threshold, only high-quality clusters pass.',
    weights: { wyckoff: 0.25, liquidity: 0.30, momentum: 0.20, smc: 0.25 },
    gateMode: 'soft',
    gates: [
      'volume_zscore >= 0.5 (volume spike for real move)',
      'derived:rsi_extreme_65 == true (RSI at extreme)',
      'temporal_confluence_score >= 0.45 (Optuna-tuned cluster quality)',
    ],
  },
  trap_within_trend: {
    name: 'Trap Within Trend (H)',
    proven: true,
    calibrated: true,
    pf: '3.54',
    dir: 'long',
    trades: '18 trades (2020-2024)',
    desc: 'False breakdown in strong uptrend with wick rejection. PnL: $8K. High quality, selective.',
    explanation:
      'Requires wick anomaly + median volume (inherited) plus wick direction validation and setup recency. ADX >= 10 structural check ensures a trend exists. bars_since_pivot <= 110 filters stale setups. Hard gate mode.',
    whyItWorks:
      'Short sellers enter on the support break, but institutional buyers absorb selling at lower prices within an established trend. The wick rejection against the trend direction is the spring-loaded reversal signal.',
    weights: { wyckoff: 0.35, liquidity: 0.35, momentum: 0.15, smc: 0.15 },
    gateMode: 'hard',
    gates: [
      'derived:wick_anomaly == true (wick against trend)',
      'volume_zscore >= 0.0 (at least median volume)',
      'wick_lower_ratio >= 0.25 (directional wick — Optuna-tuned)',
      'bars_since_pivot <= 110 (setup recency — Optuna-tuned)',
    ],
  },
  failed_continuation: {
    name: 'Failed Continuation (D)',
    proven: true,
    calibrated: true,
    pf: '13.47',
    dir: 'long',
    trades: '33 trades (2020-2024)',
    desc: 'Failed bearish continuation reversal. FVG + weak RSI + fading volume. PnL: $13K. Highest PF archetype.',
    explanation:
      'Fires when a fair value gap (FVG) exists, RSI is weak (< 55), volume is not extreme (< 3.5x), and effort-to-result ratio is low (< 1.4). These conditions signal bearish momentum is exhausting. Soft gates.',
    whyItWorks:
      'When a bearish FVG forms but volume and effort are fading, sellers are losing conviction. The gap gets filled as bears exit and longs enter the vacuum.',
    weights: { wyckoff: 0.25, liquidity: 0.25, momentum: 0.35, smc: 0.15 },
    gateMode: 'soft',
    gates: [
      'derived:any_fvg == true (FVG must be present)',
      'rsi_14 <= 55 (RSI must be weak)',
      'volume_zscore <= 3.5 (volume fading — Optuna-tuned)',
      'effort_result_ratio <= 1.4 (low effort confirms fail — Optuna-tuned)',
    ],
  },
  liquidity_compression: {
    name: 'Liquidity Compression (E)',
    proven: false,
    calibrated: true,
    pf: null,
    dir: 'long',
    desc: 'Volume exhaustion at compression. Climax volume + RSI extreme + tight BB. Rewritten from ATR compression.',
    explanation:
      'Rewritten identity: detects volume exhaustion events at price compressions. Requires low ATR (compression), climax volume flag, high volume z-score (>= 2.0), RSI at extreme (>65 or <35), and tight Bollinger Bands (< 2% width). Hard gate mode — all must pass.',
    whyItWorks:
      'Volume climaxes at compression zones signal the final burst of activity before a directional move. When RSI is at extremes within tight BBs, the exhaustion is confirmed and a reversal is imminent.',
    weights: { wyckoff: 0.20, liquidity: 0.50, momentum: 0.15, smc: 0.15 },
    gateMode: 'hard',
    gates: [
      'atr_percentile <= 0.35 (low ATR compression)',
      'climax_volume_flag == 1 (volume climax event)',
      'volume_zscore >= 2.0 (high volume validates exhaustion)',
      'derived:rsi_extreme_65 == true (RSI at extreme)',
      'derived:bb_tight_compression == true (BB width < 2%)',
    ],
  },
  spring: {
    name: 'Spring / UTAD (A)',
    proven: true,
    calibrated: true,
    pf: '8.40',
    dir: 'long',
    trades: '8 trades (2020-2024)',
    desc: 'Wyckoff spring detection. Highly selective, very profitable. PnL: $5.2K.',
    explanation:
      'Multi-path spring detection with Wyckoff bullish score and PTI score gates. Fires on Wyckoff spring events, PTI trap detection, or synthetic wick + volume + displacement signals.',
    whyItWorks:
      'Wyckoff springs are the classic institutional accumulation pattern — smart money shakes out weak hands below support, then marks up price aggressively.',
    weights: { wyckoff: 0.60, liquidity: 0.20, momentum: 0.10, smc: 0.10 },
    gateMode: 'soft',
    gates: [
      'wyckoff_bullish_score >= 0.15 (Wyckoff confirmation)',
      'tf1h_pti_score >= 0.10 (PTI trap detection)',
    ],
  },
  liquidity_vacuum: {
    name: 'Liquidity Vacuum (S1)',
    proven: true,
    calibrated: true,
    pf: 'Inf',
    dir: 'long',
    trades: '~10 trades (2020-2024)',
    desc: 'Crisis capitulation reversal at panic lows. Requires low liquidity + volume + wick exhaustion. PnL: $7K.',
    explanation:
      'Detects capitulation events where orderbook liquidity evaporates. Requires liquidity score <= 0.45 (thin books), at least median volume, and wick exhaustion >= 1.4 (Optuna-tuned) confirming selling exhaustion. Hard gate mode.',
    whyItWorks:
      'During capitulation, sellers exhaust themselves and bids evaporate. When the vacuum fills, zero resistance to the upside causes violent short-covering bounces.',
    weights: { wyckoff: 0.35, liquidity: 0.45, momentum: 0.10, smc: 0.10 },
    gateMode: 'hard',
    gates: [
      'liquidity_score <= 0.45 (low liquidity / capitulation)',
      'volume_zscore >= 0.0 (above-median volume)',
      'wick_exhaustion_last_3b >= 1.4 (exhaustion — Optuna-tuned)',
    ],
  },
  funding_divergence: {
    name: 'Funding Divergence (S4)',
    proven: true,
    calibrated: true,
    pf: '2.11',
    dir: 'long',
    trades: '26 trades (2020-2024)',
    desc: 'Short squeeze from extreme negative funding. Overcrowded shorts + OI divergence. PnL: $5.4K.',
    explanation:
      'Detects overcrowded short positions via extreme negative funding (funding_Z < -0.5), with OI divergence and LS ratio confirmation. New gates add OI price divergence and OI change 4h filters.',
    whyItWorks:
      'Shorts paying high negative funding is unsustainable. When OI diverges from price, shorts are trapped and the squeeze ignites.',
    weights: { wyckoff: 0.20, liquidity: 0.50, momentum: 0.20, smc: 0.10 },
    gateMode: 'soft',
    gates: [
      'funding_Z <= -0.5 (extreme negative funding)',
      'funding_oi_divergence == 1 (OI/funding divergence)',
      'ls_ratio_extreme <= -0.5 (shorts overcrowded)',
      'oi_price_divergence >= 0.01 (OI declining while price holds)',
      'oi_change_4h <= 0.05 (OI should be dropping)',
    ],
  },
  long_squeeze: {
    name: 'Long Squeeze (S5)',
    proven: true,
    calibrated: true,
    pf: '3.83',
    dir: 'short',
    trades: '23 trades (2020-2024)',
    desc: 'Short archetype. Overcrowded longs + exhaustion = cascade down. PnL: $5.1K.',
    explanation:
      'Detects overcrowded long positions via high positive funding, RSI overbought, and thin liquidity. Only short archetype in the system.',
    whyItWorks:
      'Longs paying high positive funding is unsustainable. Overbought RSI + thin liquidity means no buyers left. Any selling triggers cascading liquidations.',
    weights: { wyckoff: 0.20, liquidity: 0.50, momentum: 0.20, smc: 0.10 },
    gateMode: 'soft',
    gates: [
      'funding_Z >= 1.0 (positive extreme)',
      'RSI >= 60 (overbought)',
      'ls_ratio_extreme >= 0.5 (longs overcrowded)',
    ],
  },
  order_block_retest: {
    name: 'Order Block Retest (B)',
    proven: false,
    calibrated: false,
    pf: '0.00',
    dir: 'long',
    trades: '1 trade (2020-2024)',
    desc: 'SMC order block retest. Low trade count, needs more data.',
    explanation:
      'Detects retests of smart money order blocks using Wyckoff + fib time confluence. Soft gate penalty for failed gates.',
    whyItWorks:
      'Order blocks are zones where institutional traders placed large orders. Retests create reliable support bounces.',
    weights: { wyckoff: 0.30, liquidity: 0.15, momentum: 0.15, smc: 0.40 },
    gateMode: 'soft',
    gates: [
      'wyckoff_bullish_score >= 0.05 (structural confirmation)',
      'fib_time_confluence >= 0.1 (fib time ratio)',
    ],
  },
  exhaustion_reversal: {
    name: 'Exhaustion Reversal (F)',
    proven: true,
    calibrated: true,
    pf: '2.83',
    dir: 'long',
    trades: '29 trades (2020-2024)',
    desc: 'Momentum exhaustion at RSI extremes with volume spike. PnL: $6.4K.',
    explanation:
      'Detects momentum exhaustion when RSI hits extremes (> 78 or < 22), ATR percentile exceeds 0.90, and volume z-score exceeds 1.0.',
    whyItWorks:
      'Extreme RSI with volume spikes at high volatility indicates the final capitulation wave. No more sellers left, creating a natural reversal point.',
    weights: { wyckoff: 0.25, liquidity: 0.20, momentum: 0.45, smc: 0.10 },
    gateMode: 'soft',
    gates: [
      'RSI > 78 OR RSI < 22 (extreme reading)',
      'ATR percentile > 0.90 (high volatility)',
      'volume_zscore > 1.0 (volume spike)',
    ],
  },
  fvg_continuation: {
    name: 'FVG Continuation (C)',
    proven: false,
    calibrated: false,
    pf: '0.40',
    dir: 'long',
    trades: '10 trades (2020-2024)',
    desc: 'BOS/CHOCH reversal. Net loser currently. PnL: -$1.7K.',
    explanation:
      'Fires when bullish BOS is detected on 1H. CHOCH and wick rejection add confidence bonuses.',
    whyItWorks:
      'Break of Structure signals end of bearish control. CHOCH + wick rejection validate the reversal.',
    weights: { wyckoff: 0.15, liquidity: 0.20, momentum: 0.30, smc: 0.35 },
    gateMode: 'soft',
    gates: [
      'tf1h_bos_bullish == true (required)',
      'CHOCH flag adds bonus (optional)',
    ],
  },
  confluence_breakout: {
    name: 'Confluence Breakout (M)',
    proven: false,
    calibrated: false,
    pf: null,
    dir: 'long',
    desc: 'Multi-timeframe coil breakout with volume and 4H fusion confirmation.',
    explanation:
      'Uses ATR compression to detect coiled springs, then requires 4H fusion score and volume confirmation for the breakout.',
    whyItWorks:
      'Volatility compression creates potential energy. Volume + multi-TF confirmation validates the breakout direction.',
    weights: { wyckoff: 0.30, liquidity: 0.25, momentum: 0.20, smc: 0.25 },
    gateMode: 'soft',
    gates: [
      'atr_percentile <= 0.4 (compression)',
      'tf4h_fusion_score >= 0.2 (MTF confirmation)',
      'volume_zscore >= 0.5 (volume confirms breakout)',
    ],
  },
  whipsaw: {
    name: 'Whipsaw (S3)',
    proven: false,
    calibrated: false,
    pf: null,
    dir: 'neutral',
    desc: 'Distribution climax short with Wyckoff SOW confirmation.',
    explanation:
      'Volume climax + overbought RSI + Wyckoff Sign of Weakness signals exhaustion at distribution highs.',
    whyItWorks:
      'Volume climaxes at overbought levels + SOW confirmation represent the final wave of retail buying while institutions distribute.',
    weights: { wyckoff: 0.25, liquidity: 0.30, momentum: 0.30, smc: 0.15 },
    gateMode: 'soft',
    gates: [
      'volume_zscore <= 0.5 (volume fading)',
      'wyckoff_sow == true (Sign of Weakness — distribution context)',
    ],
  },
  volume_fade_chop: {
    name: 'Volume Fade Chop (S8)',
    proven: false,
    calibrated: false,
    pf: null,
    dir: 'neutral',
    desc: 'Low-volume fade scalper for choppy conditions.',
    explanation:
      'Fires when volume fades, ADX is low (no trend), and effort-to-result ratio confirms low conviction.',
    whyItWorks:
      'Dead markets with fading volume mean-revert. Low effort-to-result confirms the chop is genuine, not accumulation.',
    weights: { wyckoff: 0.15, liquidity: 0.35, momentum: 0.40, smc: 0.10 },
    gateMode: 'soft',
    gates: [
      'volume_zscore <= 0.5 (volume fade)',
      'adx_14 <= 25.0 (no trend)',
      'effort_result_ratio <= 1.5 (low effort confirms chop)',
    ],
  },
  oi_divergence: {
    name: 'OI Divergence (S9)',
    proven: false,
    calibrated: false,
    pf: '0.69',
    dir: 'long',
    trades: '4 trades (shadow mode)',
    desc: 'OI/price divergence detector. Shadow mode. PnL: -$460.',
    explanation:
      'Detects divergence where price rises but OI falls, signaling hollow moves. Requires Binance Futures data (post-2022).',
    whyItWorks:
      'When OI declines while price rises, the rally is driven by short covering, not conviction. These hollow rallies reverse.',
    weights: { wyckoff: 0.15, liquidity: 0.25, momentum: 0.25, smc: 0.35 },
    gateMode: 'soft',
    gates: [
      'oi_price_divergence == -1 (bearish divergence)',
      'Binance OI data required (post-2022)',
    ],
  },
};
