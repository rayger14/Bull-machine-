import { useState } from 'react';
import { Info } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import { fearGreedLabel } from '../../utils/format';
import type { MacroData, OracleData } from '../../api/types';

interface MacroEnvironmentProps {
  macro: MacroData | null | undefined;
  oracle?: OracleData | null;
}

const MACRO_INFO: Record<string, string> = {
  'Fear & Greed': 'Crypto market sentiment index (0=Extreme Fear, 100=Extreme Greed). Extreme fear (<15) historically precedes recoveries -- but drawdowns can extend. Our engine raises the threshold during extreme fear, requiring stronger signals.',
  'BTC.D': "BTC's share of total crypto market cap. Rising BTC.D = flight to quality (altcoins bleeding). Falling BTC.D = risk-on alt rotation. We track this to gauge whether capital is rotating within crypto or exiting entirely.",
  'USDT.D': "Tether's share of total crypto market. Rising USDT.D = capital fleeing risk into stables (bearish). Falling USDT.D = stablecoins converting back to crypto (bullish). Breakout above 7.5% signals heavy de-risking.",
  'USDC.D': "Circle's USDC share of total crypto market. Tracks institutional stablecoin flows (USDC is preferred by US institutions). Combined USDT+USDC dominance > 10% = significant capital parked in stables (bearish crypto).",
  'VIX Z': 'CBOE Volatility Index standardized score. VIX measures expected S&P500 volatility -- proxy for institutional fear. Z > 2 = panic across traditional markets, which spills into crypto as institutions de-risk all portfolios.',
  'DXY Z': 'Dollar Index standardized score. Strong dollar (Z > 1.5) drains liquidity from risk assets including crypto. Weak dollar (Z < -1.5) adds liquidity -- historically one of the strongest tailwinds for BTC.',
  'Gold Z': 'Gold standardized score. Rising gold (Z > 1) = flight-to-safety active. Historically, when gold surges while crypto falls, it signals macro fear is driving capital away from ALL risk assets.',
  'Oil Z': 'Crude oil standardized score. Oil spikes (Z > 2) create inflationary pressure and stagflation risk. Combined with dollar strength, this is one of the most toxic environments for crypto.',
  'Yield Curve': '10Y-2Y Treasury spread. Negative = inverted (recession signal). Positive = normal. Inversions historically precede recessions by 12-18 months. Crypto typically suffers during late-cycle inversions.',
};

// Oracle macro_summary key mapping for each indicator label
const ORACLE_MACRO_KEYS: Record<string, string> = {
  'Fear & Greed': 'sentiment',
  'VIX Z': 'volatility',
  'DXY Z': 'dollar',
  'Gold Z': 'gold',
  'Oil Z': 'oil',
  'Yield Curve': 'bonds',
  'BTC.D': 'btc_dominance',
  'USDT.D': 'stablecoins',
  'USDC.D': 'stablecoins',
};

type TrafficLight = 'bullish' | 'bearish' | 'neutral';

/** Determine traffic light status for each indicator */
function getTrafficLight(label: string, value: number | undefined): TrafficLight {
  if (value == null) return 'neutral';

  switch (label) {
    case 'VIX Z': {
      // VIX: low = green, medium = yellow, high = red
      // Using Z-score thresholds that map to approximately VIX 18/25
      if (value < -0.5) return 'bullish';   // low VIX
      if (value > 0.5) return 'bearish';    // high VIX
      return 'neutral';
    }
    case 'DXY Z': {
      // Weak dollar good for BTC, strong dollar bad
      if (value < -0.5) return 'bullish';
      if (value > 0.5) return 'bearish';
      return 'neutral';
    }
    case 'Gold Z': {
      // Gold rising = risk-on / inflation hedge, generally positive for BTC
      if (value > 0.5) return 'bullish';
      if (value < -0.5) return 'bearish';
      return 'neutral';
    }
    case 'Oil Z': {
      // Low oil costs good, high oil = stagflation risk
      if (value < 0.5) return 'bullish';
      if (value > 1.0) return 'bearish';
      return 'neutral';
    }
    case 'Fear & Greed': {
      // Extreme fear = contrarian buy, extreme greed = warning
      if (value < 25) return 'bullish';   // extreme fear = contrarian buy
      if (value > 75) return 'bearish';   // extreme greed = overheated
      return 'neutral';
    }
    case 'BTC.D': {
      // BTC dominance is generally neutral; both extremes are worth watching
      return 'neutral';
    }
    case 'USDT.D': {
      // High stablecoin dominance = flight to safety (bearish), low = risk-on
      if (value > 7.5) return 'bearish';
      if (value < 5.0) return 'bullish';
      return 'neutral';
    }
    case 'USDC.D': {
      // Similar logic to USDT.D
      if (value > 3.0) return 'bearish';
      if (value < 1.5) return 'bullish';
      return 'neutral';
    }
    case 'Yield Curve': {
      // Positive = normal/green, negative/inverted = red, near zero = yellow
      if (value > 0.1) return 'bullish';
      if (value < -0.1) return 'bearish';
      return 'neutral';
    }
    default:
      return 'neutral';
  }
}

/** Get traffic light CSS class for the dot */
function trafficDotClass(status: TrafficLight): string {
  if (status === 'bullish') return 'bg-green-400';
  if (status === 'bearish') return 'bg-red-400';
  return 'bg-yellow-400';
}

/** Get human-readable state label for indicators */
function getStateLabel(label: string, value: number | undefined): string {
  if (value == null) return '--';

  switch (label) {
    case 'VIX Z': {
      const absZ = Math.abs(value);
      if (absZ < 0.5) return 'LOW -- calm markets, good for crypto';
      if (absZ < 1.5) return 'MODERATE -- some caution warranted';
      return 'HIGH -- risk-off, institutional fear';
    }
    case 'DXY Z': {
      if (value < -1.0) return 'WEAK -- liquidity tailwind for BTC';
      if (value < -0.5) return 'SOFTENING -- mildly supportive';
      if (value > 1.0) return 'SURGING -- liquidity drain, bearish crypto';
      if (value > 0.5) return 'FIRM -- headwind building';
      return 'NEUTRAL -- no strong signal';
    }
    case 'Gold Z': {
      if (value > 1.0) return 'SURGING -- safe haven demand high';
      if (value > 0.5) return 'RISING -- inflation hedge bid';
      if (value < -0.5) return 'FALLING -- risk-on environment';
      return 'STABLE -- balanced';
    }
    case 'Oil Z': {
      if (value > 1.5) return 'ELEVATED -- stagflation watch';
      if (value > 0.5) return 'RISING -- inflationary pressure';
      if (value < -0.5) return 'LOW -- energy costs easing';
      return 'NORMAL -- no inflation pressure';
    }
    case 'Fear & Greed': {
      if (value < 15) return 'EXTREME FEAR -- contrarian buy signal';
      if (value < 25) return 'FEAR -- historically a buy zone';
      if (value < 45) return 'CAUTIOUS -- sentiment weak';
      if (value < 55) return 'NEUTRAL -- balanced sentiment';
      if (value < 75) return 'GREED -- risk appetite expanding';
      return 'EXTREME GREED -- overheated, caution';
    }
    case 'BTC.D': {
      if (value > 55) return 'HIGH -- flight to quality within crypto';
      if (value < 45) return 'LOW -- alt rotation, risk-on crypto';
      return 'MODERATE -- balanced allocation';
    }
    case 'USDT.D': {
      if (value > 7.5) return 'HIGH -- capital fleeing to stables';
      if (value < 5.0) return 'LOW -- stables deploying into crypto';
      return 'NORMAL -- balanced flow';
    }
    case 'USDC.D': {
      if (value > 3.0) return 'ELEVATED -- institutional de-risking';
      if (value < 1.5) return 'LOW -- capital deployed in market';
      return 'NORMAL -- steady';
    }
    case 'Yield Curve': {
      if (value < -0.5) return 'DEEPLY INVERTED -- recession signal';
      if (value < 0) return 'INVERTED -- caution warranted';
      if (value > 0.5) return 'NORMAL -- healthy economy signal';
      return 'FLAT -- transition zone';
    }
    default:
      return '--';
  }
}

export default function MacroEnvironment({ macro, oracle }: MacroEnvironmentProps) {
  const [expandedInfo, setExpandedInfo] = useState<string | null>(null);

  if (!macro) return null;

  const items = [
    { label: 'Fear & Greed', value: macro.fear_greed, fmt: (v: number) => `${v.toFixed(0)}`, sub: fearGreedLabel(macro.fear_greed), color: fgColor(macro.fear_greed) },
    { label: 'BTC.D', value: macro.btc_dominance, fmt: (v: number) => `${v.toFixed(1)}%`, sub: 'Dominance', color: 'text-amber-400' },
    { label: 'USDT.D', value: macro.usdt_dominance, fmt: (v: number) => `${v.toFixed(2)}%`, sub: 'Tether', color: 'text-cyan-400' },
    { label: 'USDC.D', value: macro.usdc_dominance, fmt: (v: number) => `${v.toFixed(2)}%`, sub: 'Circle', color: 'text-blue-400' },
    { label: 'VIX Z', value: macro.vix_z, fmt: (v: number) => v.toFixed(2), sub: zLabel(macro.vix_z), color: zColor(macro.vix_z) },
    { label: 'DXY Z', value: macro.dxy_z, fmt: (v: number) => v.toFixed(2), sub: zLabel(macro.dxy_z), color: zColor(macro.dxy_z) },
    { label: 'Gold Z', value: macro.gold_z, fmt: (v: number) => v.toFixed(2), sub: zLabel(macro.gold_z), color: zColor(macro.gold_z) },
    { label: 'Oil Z', value: macro.oil_z, fmt: (v: number) => v.toFixed(2), sub: zLabel(macro.oil_z), color: zColor(macro.oil_z) },
    { label: 'Yield Curve', value: macro.yield_curve, fmt: (v: number) => `${v.toFixed(2)}%`, sub: macro.yield_curve != null && macro.yield_curve < 0 ? 'Inverted' : 'Normal', color: macro.yield_curve != null && macro.yield_curve < 0 ? 'text-rose-400' : 'text-emerald-400' },
  ];

  return (
    <GlassCard>
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">Macro Environment</div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {items.map((item) => {
          const trafficLight = getTrafficLight(item.label, item.value ?? undefined);
          const stateLabel = getStateLabel(item.label, item.value ?? undefined);
          const oracleMacroKey = ORACLE_MACRO_KEYS[item.label];
          const oracleMacroItem = oracleMacroKey ? oracle?.macro_summary?.[oracleMacroKey] : undefined;

          return (
            <div key={item.label} className="bg-white/[0.02] rounded-xl border border-white/[0.04] p-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <span className={`inline-block w-2 h-2 rounded-full flex-shrink-0 ${trafficDotClass(trafficLight)}`} />
                  <div className="text-[10px] text-slate-600 uppercase">{item.label}</div>
                </div>
                {MACRO_INFO[item.label] && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setExpandedInfo(expandedInfo === item.label ? null : item.label);
                    }}
                    className="text-slate-600 hover:text-slate-400 transition-colors"
                    title="Learn more"
                  >
                    <Info className="w-3 h-3" />
                  </button>
                )}
              </div>
              <div className={`text-lg font-bold font-mono ${item.color}`}>
                {item.value != null ? item.fmt(item.value) : '--'}
              </div>
              {/* Human-readable state description */}
              <div className="text-[10px] text-slate-500 leading-snug">
                {stateLabel}
              </div>
              {/* Oracle macro_summary impact subtitle */}
              {oracleMacroItem?.impact && (
                <div className="text-[10px] text-slate-400 mt-0.5 leading-snug">
                  {oracleMacroItem.impact}
                </div>
              )}
              {expandedInfo === item.label && MACRO_INFO[item.label] && (
                <div className="mt-2 text-[10px] text-slate-500 leading-relaxed border-t border-white/[0.05] pt-2">
                  {MACRO_INFO[item.label]}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </GlassCard>
  );
}

function fgColor(v?: number): string {
  if (v == null) return 'text-slate-400';
  if (v < 25) return 'text-rose-400';
  if (v < 45) return 'text-orange-400';
  if (v < 55) return 'text-amber-400';
  if (v < 75) return 'text-emerald-400';
  return 'text-emerald-300';
}

function zColor(v?: number): string {
  if (v == null) return 'text-slate-400';
  const a = Math.abs(v);
  if (a < 1) return 'text-slate-300';
  if (a < 2) return 'text-amber-400';
  return 'text-rose-400';
}

function zLabel(v?: number): string {
  if (v == null) return '--';
  const a = Math.abs(v);
  if (a < 1) return 'Normal';
  if (a < 2) return 'Elevated';
  return 'Extreme';
}
