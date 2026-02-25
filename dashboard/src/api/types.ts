// --- /api/status response ---
export interface StatusResponse {
  heartbeat: Heartbeat;
  performance: Performance;
  funding: FundingCosts;
  server_time: string;
}

export interface Heartbeat {
  timestamp?: string;
  updated_at?: string;
  btc_price?: number;
  regime?: string;
  equity?: number;
  leverage?: number;
  threshold?: number;
  risk_temp?: number;
  instability?: number;
  crisis_prob?: number;
  positions?: number;
  completed_trades?: number;
  net_pnl?: number;
  return_pct?: number;
  max_drawdown_pct?: number;
  win_rate?: number;
  profit_factor?: number;
  data_source?: string;
  session_start?: string;
  bars_processed?: number;
  total_signals?: number;
  signals_allocated?: number;
  signals_rejected?: number;
  signals_this_bar?: number;
  last_signal_time?: string;

  // CMI breakdown
  cmi_breakdown?: CMIBreakdown;

  // Macro
  macro?: MacroData;
  macro_outlook?: Record<string, MacroOutlookTimeframe>;

  // Open positions
  open_position_details?: OpenPosition[];

  // Stress scenarios
  active_stress_scenarios?: StressScenario[];

  // Cointegration
  cointegration?: CointegrationData;

  // Correlations
  macro_correlations?: MacroCorrelations;

  // Capital flows
  capital_flows?: CapitalFlowsData;

  // Wyckoff
  wyckoff?: WyckoffData;

  // Narrative
  last_signal_narrative?: SignalNarrative | null;

  // Last bar signals
  last_bar_signals?: Signal[];

  // Funding
  funding?: FundingCardData;

  // Factor attribution
  factor_attribution_summary?: Record<string, FactorAttrBucket>;

  // Oracle synthesis
  oracle?: OracleData;

  // Phantom trade tracker
  phantom_tracker?: PhantomTracker;

  // CMI weight comparison (hand-tuned vs optimized)
  cmi_comparison?: CMIComparison | null;

  // Whale / Institutional Intelligence
  whale_intelligence?: WhaleIntelligenceData;
}

export interface CMIComparisonSide {
  risk_temp: number;
  instability: number;
  crisis_prob: number;
  threshold: number;
}

export interface CMIComparison {
  hand_tuned: CMIComparisonSide;
  optimized: CMIComparisonSide;
  agreement: boolean;
  delta: number;
}

export interface CMIBreakdown {
  risk_temp_components?: Record<string, number>;
  risk_temp_weights?: Record<string, number>;
  instability_components?: Record<string, number>;
  instability_weights?: Record<string, number>;
  crisis_components?: Record<string, number>;
  crisis_weights?: Record<string, number>;
  raw_features?: Record<string, number>;
  threshold_config?: {
    base_threshold?: number;
    dynamic_threshold?: number;
    temp_range?: number;
    instab_range?: number;
    crisis_coeff?: number;
  };
}

export interface MacroData {
  fear_greed?: number;
  fear_greed_label?: string;
  btc_dominance?: number;
  usdt_dominance?: number;
  usdc_dominance?: number;
  vix_z?: number;
  dxy_z?: number;
  gold_z?: number;
  oil_z?: number;
  yield_curve?: number;
}

export interface MacroOutlookFactor {
  name?: string;
  signal?: number;
  weight?: number;
  contribution?: number;
}

export interface MacroOutlookCase {
  factors?: string[];
  score?: number;
}

export interface MacroOutlookTraderSignals {
  wyckoff?: string[];
  moneytaur?: string[];
  zeroika?: string[];
}

export interface MacroOutlookTimeframe {
  label?: string;
  score?: number;
  narrative?: string;
  regime?: string;
  regime_label?: string;
  factors?: MacroOutlookFactor[];
  bull_case?: MacroOutlookCase;
  bear_case?: MacroOutlookCase;
  key_movers?: string[];
  trader_signals?: MacroOutlookTraderSignals;
  states?: Record<string, string>;
}

export interface OpenPosition {
  id?: string;
  archetype?: string;
  direction?: string;
  entry_time?: string;
  entry_price?: number;
  current_price?: number;
  unrealized_pnl?: number;
  unrealized_pnl_pct?: number;
  stop_loss?: number;
  take_profit?: number;
  position_size_usd?: number;
  leverage?: number;
  risk_reward?: number;
  sl_distance_pct?: number;
  tp_distance_pct?: number;
  // Full trade context
  bars_held?: number;
  fusion_score?: number;
  regime_at_entry?: string;
  original_quantity?: number;
  current_quantity?: number;
  atr_at_entry?: number;
  trailing_stop?: number | null;
  executed_scale_outs?: number[];
  total_exits_pct?: number;
  // Threshold context at entry
  threshold_at_entry?: number;
  threshold_margin?: number;
  would_have_passed?: boolean;
  risk_temp_at_entry?: number;
  instability_at_entry?: number;
  crisis_prob_at_entry?: number;
  // Entry reasoning
  narrative?: SignalNarrative;
  factor_attribution?: Record<string, number>;
}

export interface StressScenario {
  name?: string;
  active?: boolean;
  severity?: string;
  current_values?: Record<string, number>;
  description?: string;
  recommendation?: string;
  historical?: {
    avg_return_24h?: number;
    avg_return_72h?: number;
    avg_return_168h?: number;
    worst_return_24h?: number;
    max_drawdown_pct?: number;
    num_episodes?: number;
    occurrences?: number;
    pct_of_history?: number;
    avg_duration_hours?: number;
    median_return_24h?: number;
    pct_positive_24h?: number;
    pct_positive_168h?: number;
    estimated_win_rate?: number;
    avg_threshold_during?: number;
  };
}

export interface CointegrationData {
  pairs?: CointegrationPair[];
  n_bars_available?: number;
  min_bars_required?: number;
  has_opportunity?: boolean;
  status?: string;
  method?: string;
  last_computed?: string;
}

export interface CointegrationPair {
  pair?: string;
  cointegrated?: boolean;
  half_life?: number;
  half_life_hours?: number;
  z_score?: number;
  current_zscore?: number;
  p_value?: number;
  signal?: string;
  has_opportunity?: boolean;
  stability?: string;
  n_valid_bars?: number;
  adf_statistic?: number;
  beta?: number;
}

export interface MacroCorrelations {
  regime?: string;
  window_20?: Record<string, number>;
  window_60?: Record<string, number>;
  avg_abs_corr_20?: number;
  n_bars?: number;
  n_macro_bars?: number;
  min_bars_20?: number;
  min_bars_60?: number;
}

export interface CapitalFlowNode {
  label?: string;
  state?: string;
  status?: string;
  value?: string;
}

export interface CapitalFlowEdge {
  active?: boolean;
  direction?: string;
  from?: string;
  to?: string;
  label?: string;
  source_fn?: string;
  strength?: number;
}

export interface CapitalFlowsData {
  nodes?: Record<string, CapitalFlowNode>;
  edges?: Record<string, CapitalFlowEdge>;
}

export interface WyckoffEventData {
  active?: boolean;
  confidence?: number;
}

export interface WyckoffMarketContext {
  volume_z?: number;
  close_position?: number;
  lower_wick_pct?: number;
  upper_wick_pct?: number;
  rsi_14?: number;
  adx?: number;
  range_z?: number;
  close?: number;
}

export interface WyckoffData {
  score?: number;
  event_confidence?: number;
  phase?: string;               // A/B/C/D/E/neutral
  sequence_position?: number;   // 1-10 within cycle
  events?: Record<string, WyckoffEventData>;
  market_context?: WyckoffMarketContext;
  tf4h_phase_score?: number;
  tf1d_score?: number;
  tf1d_m1_signal?: number;
  tf1d_m2_signal?: number;
  tf1d_bars?: number;

  // MTF directional scores (graded 0-1, diversity-weighted)
  bullish_1h?: number;
  bearish_1h?: number;
  bullish_4h?: number;
  bearish_4h?: number;
  bullish_1d?: number;
  bearish_1d?: number;

  // NEW: Cycle tracking
  cycle_start?: string;
  cycle_duration_hours?: number;
  phase_transitions?: WyckoffPhaseTransition[];

  // NEW: Event enrichment
  event_history?: WyckoffEventHistoryItem[];
  conviction?: WyckoffConviction;
  event_narratives?: Record<string, string>;

  // NEW: Reference data
  typical_durations?: Record<string, WyckoffTypicalDuration>;
  methodology?: WyckoffMethodology;
}

export interface WyckoffPhaseTransition {
  from_phase?: string;
  to_phase?: string;
  timestamp?: string;
  price?: number;
}

export interface WyckoffEventHistoryItem {
  event?: string;
  timestamp?: string;
  price?: number;
  high?: number;
  low?: number;
  confidence?: number;
  volume_z?: number;
}

export interface WyckoffConvictionComponent {
  event?: string;
  confidence?: number;
  weight?: number;
  contribution?: number;
}

export interface WyckoffConviction {
  total_score?: number;
  components?: WyckoffConvictionComponent[];
  reason?: string;
}

export interface WyckoffTypicalDuration {
  hours?: string;
  description?: string;
}

export interface WyckoffMethodology {
  type?: string;
  description?: string;
  limitations?: string[];
}

export interface SignalNarrative {
  headline?: string;
  summary?: string;
  text?: string;
  fusion_score?: number;
  threshold?: number;
  regime?: string;
  cmi_components?: Record<string, number>;
  confluence_factors?: string[];
  risk_factors?: string[];
  position_sizing?: Record<string, number | string>;
  domain_scores?: {
    wyckoff?: number;
    liquidity?: number;
    momentum?: number;
    smc?: number;
  };
  gate_values?: Record<string, number>;
  regime_context?: string;
}

export interface FundingCardData {
  last_rate_bps?: number;
  annualized_pct?: number;
  total_cost_usd?: number;
  funding_z?: number;
  total_events?: number;
}

export interface FactorAttrBucket {
  avg_pct?: number;
  win_contribution?: number;
  loss_contribution?: number;
}

export interface Performance {
  completed_trades?: number;
  win_rate_pct?: number;
  profit_factor?: number;
  total_return_pct?: number;
  max_drawdown_pct?: number;
  total_pnl?: number;
  net_pnl_after_funding?: number;
  current_equity?: number;
  initial_cash?: number;
  open_positions?: number;
  signals_allocated?: number;
  signals_rejected?: number;
  total_signals_generated?: number;
  total_funding_cost_usd?: number;
  adapter_source?: string;
  session_start?: string;
  last_updated?: string;
  bars_processed?: number;
  [key: string]: number | string | undefined;
}

export interface FundingCosts {
  cost_by_position?: Record<string, number>;
  last_funding_rate?: number;
  last_funding_timestamp?: string;
  total_funding_cost_usd?: number;
  total_funding_events?: number;
  [key: string]: number | string | Record<string, number> | undefined;
}

// --- /api/equity-history response ---
export interface EquityRow {
  timestamp: string;
  equity: string;
  btc_price: string;
  threshold?: string;
  regime?: string;
  risk_temp?: string;
  instability?: string;
  crisis_prob?: string;
}

// --- /api/signal-log response ---
export interface Signal {
  timestamp?: string;
  archetype?: string;
  status?: string;
  fusion_score?: number;
  threshold?: number;
  regime?: string;
  rejection_stage?: string;
  rejection_reason?: string;
  narrative?: SignalNarrative;
  gate_values?: Record<string, number | string | boolean>;
  [key: string]: unknown;
}

// --- Counterfactual analysis types ---
export interface CounterfactualResult {
  scenario: string;
  param_changed: string;
  param_original: number;
  param_alternative: number;
  exit_price: number;
  exit_reason: string;
  pnl: number;
  pnl_pct: number;
  duration_hours: number;
  pnl_delta: number;
  pnl_pct_delta: number;
}

export interface TradeCounterfactual {
  trade_entry_ts: string;
  trade_archetype: string;
  actual_pnl: number;
  actual_pnl_pct: number;
  actual_exit_reason: string;
  best_scenario: string;
  best_pnl_delta: number;
  worst_scenario: string;
  worst_pnl_delta: number;
  was_optimal: boolean;
  scenarios: Record<string, CounterfactualResult>;
}

// --- /api/trades response ---
export interface Trade {
  timestamp_entry?: string;
  timestamp_exit?: string;
  archetype?: string;
  direction?: string;
  entry_price?: number;
  exit_price?: number;
  pnl?: number;
  pnl_pct?: number;
  fusion_score?: number;
  threshold?: number;
  regime?: string;
  duration_hours?: number;
  exit_reason?: string;
  factor_attribution?: Record<string, number>;
  narrative?: SignalNarrative;
  stop_loss?: number;
  take_profit?: number;
  atr_at_entry?: number;
  threshold_at_entry?: number;
  threshold_margin?: number;
  risk_temp_at_entry?: number;
  instability_at_entry?: number;
  crisis_prob_at_entry?: number;
  leverage_applied?: number;
  position_size_usd?: number;
  counterfactual?: TradeCounterfactual;
  [key: string]: unknown;
}

// --- /api/candle-history response ---
export interface CandleRow {
  timestamp: string;
  close: string;
  volume: string;
}

// --- Oracle synthesis types ---
export interface OracleData {
  posture: 'RISK_ON' | 'CAUTIOUS' | 'DEFENSIVE' | 'CRISIS';
  confidence: number;
  bias: 'bullish' | 'bearish' | 'neutral';
  bias_strength: number;
  thesis: string;
  one_liner: string;
  outlook: {
    short_term: OracleTimeframe;
    medium_term: OracleTimeframe;
    long_term: OracleTimeframe;
  };
  aligned_factors: string[];
  conflicting_factors: string[];
  risks: OracleRisk[];
  catalysts: string[];
  engine_status: OracleEngineStatus;
  market_structure: OracleMarketStructure;
  macro_summary: Record<string, OracleMacroItem>;
}

export interface OracleTimeframe {
  label: string;
  confidence: number;
  summary: string;
}

export interface OracleRisk {
  name: string;
  probability: number;
  impact: string;
  status: 'active' | 'watching' | 'low_risk';
}

export interface OracleEngineStatus {
  posture_description: string;
  active_positions: number;
  recent_performance: string;
  threshold_context: string;
}

export interface OracleMarketStructure {
  summary: string;
  phase: string;
  key_levels: { support: number; resistance: number; invalidation: number };
  next_expected: string;
}

export interface OracleMacroItem {
  state: string;
  impact: string;
  detail: string;
}

// --- Whale / Institutional Intelligence types ---
export interface WhaleRawData {
  oi_value?: number | null;
  oi_change_4h?: number | null;
  oi_change_24h?: number | null;
  funding_rate?: number | null;
  funding_z?: number | null;
  ls_ratio_extreme?: number | null;
  taker_imbalance?: number | null;
}

export interface WhaleDerivedData {
  oi_price_divergence?: number | null;
  funding_oi_divergence?: number | null;
  derivatives_heat?: number | null;
  oi_momentum?: number | null;
  funding_health?: number | null;
  taker_conviction?: number | null;
}

export interface WhaleConflictSignals {
  funding_overcrowded?: boolean;
  oi_declining?: boolean;
  aggressive_selling?: boolean;
  ls_ratio_extreme?: boolean;
}

export interface WhaleConflict {
  count?: number;
  penalty_multiplier?: number;
  signals?: WhaleConflictSignals;
}

export interface WhaleCMIStatus {
  derivatives_heat_weight?: number;
  note?: string;
}

export type WhaleSentiment = 'strongly_bullish' | 'bullish' | 'neutral' | 'bearish' | 'strongly_bearish' | 'no_data';

export interface WhaleIntelligenceData {
  raw?: WhaleRawData;
  derived?: WhaleDerivedData;
  conflict?: WhaleConflict;
  sentiment?: WhaleSentiment;
  has_data?: boolean;
  cmi_status?: WhaleCMIStatus;
}

// --- Backtest types ---
export interface BacktestParams {
  capital?: number;
  leverage?: number;
  commission?: string;
  slippage?: number;
  start_date?: string;
  end_date?: string;
}

export interface BacktestJob {
  status: 'queued' | 'running' | 'complete' | 'error';
  result?: BacktestResult;
  error?: string;
  started_at?: string;
  stdout?: string;
}

export interface FeatureStoreMeta {
  path: string;
  columns: number;
  rows: number;
  column_list: string[];
  date_range?: string;
}

export interface BacktestResult {
  stats: Record<string, number | string>;
  breakdown: Array<Record<string, string>>;
  equity: Array<{ timestamp: string; equity: string }>;
  stdout?: string;
  params?: BacktestParams;
  feature_store?: FeatureStoreMeta;
}

// --- Phantom trade tracker types ---
export interface PhantomTrade {
  archetype?: string;
  direction?: string;
  entry_price?: number;
  exit_price?: number;
  pnl?: number;
  pnl_pct?: number;
  fusion_score?: number;
  exit_reason?: string;
  duration_hours?: number;
  rejection_reason?: string;
  rejection_stage?: string;
}

export interface PhantomActive {
  archetype?: string;
  direction?: string;
  entry_price?: number;
  fusion_score?: number;
  stop_loss?: number;
  take_profit?: number;
  rejection_reason?: string;
}

export interface FusionBucket {
  wins: number;
  losses: number;
  total_pnl: number;
  count: number;
}

export interface PhantomArchBreakdown {
  wins: number;
  losses: number;
  pnl: number;
}

export interface PhantomTracker {
  total_phantom_signals?: number;
  completed_phantom_trades?: number;
  active_phantom_positions?: number;
  phantom_wins?: number;
  phantom_losses?: number;
  phantom_win_rate?: number;
  phantom_pnl?: number;
  phantom_avg_pnl?: number;
  real_pnl?: number;
  insight?: string;
  by_archetype?: Record<string, PhantomArchBreakdown>;
  trades?: PhantomTrade[];
  active?: PhantomActive[];
  fusion_buckets?: Record<string, FusionBucket>;
}
