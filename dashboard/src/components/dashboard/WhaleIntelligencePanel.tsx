import { useState } from 'react';
import GlassCard from '../ui/GlassCard';
import type { WhaleIntelligenceData, OracleData } from '../../api/types';

interface WhaleIntelligencePanelProps {
  whale: WhaleIntelligenceData | null | undefined;
  oracle?: OracleData | null;
}

// -- Info explanations: what each metric means + how the engine uses it --
const WHALE_INFO: Record<string, { what: string; engine: string }> = {
  'OI Change (4H)': {
    what: 'Percentage change in total Open Interest over the last 4 hours. Rising OI means traders are opening NEW positions (conviction). Falling OI means traders are closing positions (unwinding).',
    engine: 'Used in whale conflict penalty: OI declining below -5% flags institutional exit. Also feeds the derivatives_heat CMI sub-component as "OI momentum".',
  },
  'OI Change (24H)': {
    what: 'Percentage change in Open Interest over 24 hours. Shows the broader trend of position building or unwinding across the market.',
    engine: 'Provides context for the 4H signal. A declining 24H trend with declining 4H confirms sustained institutional unwinding.',
  },
  'Funding Rate': {
    what: 'The fee paid between longs and shorts every 8 hours. Positive = longs pay shorts (market is long-heavy). Negative = shorts pay longs (market is short-heavy).',
    engine: 'Converted to a Z-score (funding_Z). When funding_Z > 2.0, longs are overcrowded — the engine applies a conflict penalty to long signals. Extreme negative funding boosts short-squeeze archetypes.',
  },
  'Funding Z': {
    what: 'Z-score of the funding rate relative to its recent history. Values above +2 = extreme long crowding. Below -2 = extreme short crowding.',
    engine: 'Primary overcrowding signal. Z > 2.0 triggers whale conflict for longs. Z < -2.0 triggers conflict for shorts. Also used as a hard gate for funding_divergence and long_squeeze archetypes.',
  },
  'L/S Ratio': {
    what: 'Long/Short ratio extreme indicator. Positive values mean more longs than shorts. Values > 2.0 indicate extreme long positioning, < -2.0 extreme short positioning.',
    engine: 'Used in whale conflict penalty: extreme L/S ratio flags crowded positioning. Also a hard gate for long_squeeze archetype (requires L/S > 1.5 to confirm longs are overcrowded).',
  },
  'Taker Imbalance': {
    what: 'Net taker buy/sell imbalance. Positive = more aggressive buying (taker buys). Negative = more aggressive selling (taker sells). Shows who is crossing the spread.',
    engine: 'Taker imbalance below -0.5 flags aggressive selling (whale conflict for longs). Feeds the "taker_conviction" sub-component of derivatives_heat in the CMI.',
  },
  'OI/Price Divergence': {
    what: 'Detects when price and Open Interest move in opposite directions. Bullish divergence: price falls but OI rises (new positions at lower prices = accumulation). Bearish: price rises but OI falls (smart money exiting).',
    engine: 'Powers the oi_divergence archetype — detects hollow moves where price rises but institutional conviction is declining. This is the only archetype that reads whale EXIT signals.',
  },
  'Funding/OI Divergence': {
    what: 'Detects divergence between funding rate and Open Interest direction. Value of +1 = bullish divergence (shorts building into rising OI = squeeze setup). Value of -1 = bearish divergence.',
    engine: 'Hard gate for funding_divergence archetype (requires +1 for squeeze setups). Also used in long_squeeze archetype (requires -1 to confirm smart longs exiting).',
  },
  'Derivatives Heat': {
    what: 'Composite score (0-1) measuring institutional market conviction. Combines OI momentum (40%), funding health (30%), and taker conviction (30%). High = hot market, low = cold/uncertain.',
    engine: 'Designed as a CMI sub-component for the dynamic threshold. Currently DISABLED (weight=0.0) because we only have 2 years of OI data. Will be enabled when coverage reaches 3+ years. Computed but not factored into regime detection yet.',
  },
};

// Sentiment badge styling
function sentimentBadge(sentiment?: string) {
  switch (sentiment) {
    case 'strongly_bullish':
      return { label: 'Strongly Bullish', bg: 'bg-emerald-500/20', text: 'text-emerald-400', dot: 'bg-emerald-400' };
    case 'bullish':
      return { label: 'Bullish', bg: 'bg-emerald-500/10', text: 'text-emerald-400', dot: 'bg-emerald-400' };
    case 'neutral':
      return { label: 'Neutral', bg: 'bg-slate-500/10', text: 'text-slate-400', dot: 'bg-slate-400' };
    case 'bearish':
      return { label: 'Bearish', bg: 'bg-amber-500/10', text: 'text-amber-400', dot: 'bg-amber-400' };
    case 'strongly_bearish':
      return { label: 'Strongly Bearish', bg: 'bg-rose-500/20', text: 'text-rose-400', dot: 'bg-rose-400' };
    default:
      return { label: 'No Data', bg: 'bg-slate-500/10', text: 'text-slate-600', dot: 'bg-slate-600' };
  }
}

function conflictColor(count?: number) {
  if (!count || count === 0) return 'text-emerald-400';
  if (count === 1) return 'text-amber-400';
  if (count === 2) return 'text-orange-400';
  return 'text-rose-400';
}

function oiColor(val?: number | null): string {
  if (val == null) return 'text-slate-500';
  if (val > 0.03) return 'text-emerald-400';
  if (val > 0) return 'text-emerald-400/70';
  if (val > -0.03) return 'text-amber-400';
  return 'text-rose-400';
}

function fundingZColor(val?: number | null): string {
  if (val == null) return 'text-slate-500';
  if (Math.abs(val) < 1.0) return 'text-emerald-400';
  if (Math.abs(val) < 2.0) return 'text-amber-400';
  return 'text-rose-400';
}

function takerColor(val?: number | null): string {
  if (val == null) return 'text-slate-500';
  if (val > 0.3) return 'text-emerald-400';
  if (val > -0.3) return 'text-slate-400';
  return 'text-rose-400';
}

function heatColor(val?: number | null): string {
  if (val == null) return 'text-slate-500';
  if (val > 0.65) return 'text-emerald-400';
  if (val > 0.45) return 'text-amber-400';
  return 'text-rose-400';
}

function fmtPct(val?: number | null, decimals = 2): string {
  if (val == null) return '--';
  return `${(val * 100).toFixed(decimals)}%`;
}

function fmtScore(val?: number | null): string {
  if (val == null) return '--';
  return val.toFixed(2);
}

export default function WhaleIntelligencePanel({ whale }: WhaleIntelligencePanelProps) {
  const [expandedInfo, setExpandedInfo] = useState<string | null>(null);

  if (!whale) {
    return (
      <GlassCard>
        <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">Whale Intelligence</div>
        <div className="text-sm text-slate-600">Waiting for institutional data...</div>
      </GlassCard>
    );
  }

  const raw = whale.raw ?? {};
  const derived = whale.derived ?? {};
  const conflict = whale.conflict ?? {};
  const badge = sentimentBadge(whale.sentiment);

  const toggleInfo = (key: string) => {
    setExpandedInfo(expandedInfo === key ? null : key);
  };

  // Define metric cards
  const rawItems = [
    { label: 'OI Change (4H)', value: raw.oi_change_4h, fmt: fmtPct, color: oiColor(raw.oi_change_4h),
      state: raw.oi_change_4h == null ? 'No data' : raw.oi_change_4h > 0.02 ? 'Positions opening' : raw.oi_change_4h < -0.03 ? 'Positions closing' : 'Stable' },
    { label: 'OI Change (24H)', value: raw.oi_change_24h, fmt: fmtPct, color: oiColor(raw.oi_change_24h),
      state: raw.oi_change_24h == null ? 'No data' : raw.oi_change_24h > 0.05 ? 'Strong accumulation' : raw.oi_change_24h < -0.05 ? 'Strong unwinding' : 'Normal range' },
    { label: 'Funding Rate', value: raw.funding_rate, fmt: (v: number | null | undefined) => v == null ? '--' : `${(v * 100).toFixed(4)}%`, color: fundingZColor(raw.funding_z),
      state: raw.funding_rate == null ? 'No data' : raw.funding_rate > 0.0001 ? 'Longs paying' : raw.funding_rate < -0.0001 ? 'Shorts paying' : 'Balanced' },
    { label: 'Funding Z', value: raw.funding_z, fmt: fmtScore, color: fundingZColor(raw.funding_z),
      state: raw.funding_z == null ? 'No data' : Math.abs(raw.funding_z) > 2 ? 'EXTREME crowding' : Math.abs(raw.funding_z) > 1 ? 'Elevated' : 'Normal range' },
    { label: 'L/S Ratio', value: raw.ls_ratio_extreme, fmt: fmtScore, color: raw.ls_ratio_extreme != null && Math.abs(raw.ls_ratio_extreme) > 2 ? 'text-rose-400' : raw.ls_ratio_extreme != null && Math.abs(raw.ls_ratio_extreme) > 1 ? 'text-amber-400' : 'text-slate-400',
      state: raw.ls_ratio_extreme == null ? 'No data' : raw.ls_ratio_extreme > 2 ? 'Longs overcrowded' : raw.ls_ratio_extreme < -2 ? 'Shorts overcrowded' : 'Balanced' },
    { label: 'Taker Imbalance', value: raw.taker_imbalance, fmt: fmtScore, color: takerColor(raw.taker_imbalance),
      state: raw.taker_imbalance == null ? 'No data' : raw.taker_imbalance > 0.3 ? 'Aggressive buying' : raw.taker_imbalance < -0.3 ? 'Aggressive selling' : 'Balanced flow' },
  ];

  const derivedItems = [
    { label: 'OI/Price Divergence', value: derived.oi_price_divergence, fmt: (v: number | null | undefined) => v == null ? '--' : v === 1 ? 'Bullish' : v === -1 ? 'Bearish' : 'None',
      color: derived.oi_price_divergence === 1 ? 'text-emerald-400' : derived.oi_price_divergence === -1 ? 'text-rose-400' : 'text-slate-400',
      state: derived.oi_price_divergence == null ? 'No data' : derived.oi_price_divergence === 1 ? 'Price down, OI up (accumulation)' : derived.oi_price_divergence === -1 ? 'Price up, OI down (distribution)' : 'Price and OI aligned' },
    { label: 'Funding/OI Divergence', value: derived.funding_oi_divergence, fmt: (v: number | null | undefined) => v == null ? '--' : v === 1 ? 'Bullish' : v === -1 ? 'Bearish' : 'None',
      color: derived.funding_oi_divergence === 1 ? 'text-emerald-400' : derived.funding_oi_divergence === -1 ? 'text-rose-400' : 'text-slate-400',
      state: derived.funding_oi_divergence == null ? 'No data' : derived.funding_oi_divergence === 1 ? 'Shorts building = squeeze fuel' : derived.funding_oi_divergence === -1 ? 'Smart longs exiting' : 'No divergence' },
    { label: 'Derivatives Heat', value: derived.derivatives_heat, fmt: fmtScore, color: heatColor(derived.derivatives_heat),
      state: derived.derivatives_heat == null ? 'No data' : derived.derivatives_heat > 0.65 ? 'Hot: strong institutional conviction' : derived.derivatives_heat > 0.45 ? 'Warm: moderate activity' : 'Cold: low conviction' },
  ];

  return (
    <GlassCard>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="text-xs text-slate-500 uppercase tracking-wider">Whale Intelligence</div>
          <div className={`${badge.bg} ${badge.text} text-[10px] font-semibold px-2.5 py-0.5 rounded-full flex items-center gap-1.5`}>
            <span className={`inline-block w-1.5 h-1.5 rounded-full ${badge.dot}`} />
            {badge.label}
          </div>
        </div>
        {/* Conflict indicator */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-600 uppercase">Whale Conflicts</span>
          <span className={`text-lg font-bold font-mono ${conflictColor(conflict.count)}`}>
            {conflict.count ?? 0}/4
          </span>
          {conflict.penalty_multiplier != null && conflict.penalty_multiplier < 1.0 && (
            <span className="text-[10px] text-rose-400/70">
              ({((1 - conflict.penalty_multiplier) * 100).toFixed(0)}% penalty)
            </span>
          )}
        </div>
      </div>

      {/* Explanation banner */}
      <div className="text-[10px] text-slate-500 leading-relaxed mb-4 bg-white/[0.02] rounded-lg border border-white/[0.04] px-3 py-2">
        Real-time derivatives data (OKX public API) showing where institutional money is flowing. The engine checks these signals before every trade — when whale data conflicts with a signal's direction, the fusion score is penalized.
      </div>

      {/* Conflict signals strip */}
      {conflict.signals && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
          {[
            { key: 'funding_overcrowded', label: 'Funding Crowded', desc: 'Funding Z > 2.0' },
            { key: 'oi_declining', label: 'OI Declining', desc: 'OI 4H < -5%' },
            { key: 'aggressive_selling', label: 'Taker Selling', desc: 'Imbalance < -0.5' },
            { key: 'ls_ratio_extreme', label: 'L/S Extreme', desc: 'Ratio > 2.0' },
          ].map(({ key, label, desc }) => {
            const active = (conflict.signals as Record<string, boolean>)?.[key] ?? false;
            return (
              <div
                key={key}
                className={`rounded-lg border px-2.5 py-1.5 text-center ${
                  active
                    ? 'border-rose-500/30 bg-rose-500/10'
                    : 'border-white/[0.04] bg-white/[0.02]'
                }`}
              >
                <div className={`text-[10px] font-semibold ${active ? 'text-rose-400' : 'text-slate-500'}`}>
                  {label}
                </div>
                <div className="text-[9px] text-slate-600">{desc}</div>
              </div>
            );
          })}
        </div>
      )}

      {/* Raw data section */}
      <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2 mt-2">
        Raw Data
        <span className="normal-case text-slate-600 ml-2">From OKX derivatives</span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-4">
        {rawItems.map((item) => (
          <div key={item.label} className="bg-white/[0.02] rounded-xl border border-white/[0.04] p-3">
            <div className="flex items-center justify-between">
              <div className="text-[10px] text-slate-600 uppercase leading-tight">{item.label}</div>
              {WHALE_INFO[item.label] && (
                <button
                  onClick={() => toggleInfo(item.label)}
                  className="text-slate-600 hover:text-cyan-400 text-[10px] transition-colors w-4 h-4 flex items-center justify-center"
                >
                  ?
                </button>
              )}
            </div>
            <div className={`text-lg font-bold font-mono ${item.color} mt-1`}>
              {item.value != null ? item.fmt(item.value) : '--'}
            </div>
            <div className="text-[10px] text-slate-500 leading-snug">{item.state}</div>
            {expandedInfo === item.label && WHALE_INFO[item.label] && (
              <div className="mt-2 text-[10px] text-slate-500 border-t border-white/[0.05] pt-2 space-y-1.5">
                <div><span className="text-slate-400 font-semibold">What it means:</span> {WHALE_INFO[item.label].what}</div>
                <div><span className="text-cyan-400/70 font-semibold">Engine uses it for:</span> {WHALE_INFO[item.label].engine}</div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Derived signals section */}
      <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">
        Engine-Derived Signals
        <span className="normal-case text-slate-600 ml-2">What the engine computes from raw data</span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
        {derivedItems.map((item) => (
          <div key={item.label} className="bg-white/[0.02] rounded-xl border border-white/[0.04] p-3">
            <div className="flex items-center justify-between">
              <div className="text-[10px] text-slate-600 uppercase">{item.label}</div>
              {WHALE_INFO[item.label] && (
                <button
                  onClick={() => toggleInfo(item.label)}
                  className="text-slate-600 hover:text-cyan-400 text-[10px] transition-colors w-4 h-4 flex items-center justify-center"
                >
                  ?
                </button>
              )}
            </div>
            <div className={`text-lg font-bold font-mono ${item.color} mt-1`}>
              {item.value != null ? item.fmt(item.value) : '--'}
            </div>
            <div className="text-[10px] text-slate-500 leading-snug">{item.state}</div>
            {expandedInfo === item.label && WHALE_INFO[item.label] && (
              <div className="mt-2 text-[10px] text-slate-500 border-t border-white/[0.05] pt-2 space-y-1.5">
                <div><span className="text-slate-400 font-semibold">What it means:</span> {WHALE_INFO[item.label].what}</div>
                <div><span className="text-cyan-400/70 font-semibold">Engine uses it for:</span> {WHALE_INFO[item.label].engine}</div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Derivatives heat sub-components (mini bar chart) */}
      {derived.derivatives_heat != null && (
        <div className="bg-white/[0.02] rounded-xl border border-white/[0.04] p-3">
          <div className="flex items-center justify-between mb-2">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider">
              Derivatives Heat Breakdown
              <span className="normal-case text-slate-600 ml-2">How institutional conviction is scored</span>
            </div>
            <div className={`text-sm font-bold font-mono ${heatColor(derived.derivatives_heat)}`}>
              {fmtScore(derived.derivatives_heat)}
            </div>
          </div>
          <div className="space-y-2">
            {[
              { label: 'OI Momentum', value: derived.oi_momentum, weight: '40%', desc: 'Rising OI = opening positions' },
              { label: 'Funding Health', value: derived.funding_health, weight: '30%', desc: 'Moderate = healthy, extreme = crowded' },
              { label: 'Taker Conviction', value: derived.taker_conviction, weight: '30%', desc: 'Net buying = conviction' },
            ].map((sub) => (
              <div key={sub.label} className="flex items-center gap-3">
                <div className="text-[10px] text-slate-500 w-28 shrink-0">{sub.label} <span className="text-slate-600">({sub.weight})</span></div>
                <div className="flex-1 h-2 bg-white/[0.04] rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      sub.value != null && sub.value > 0.65 ? 'bg-emerald-500/60' :
                      sub.value != null && sub.value > 0.45 ? 'bg-amber-500/60' :
                      'bg-rose-500/40'
                    }`}
                    style={{ width: `${(sub.value ?? 0.5) * 100}%` }}
                  />
                </div>
                <div className={`text-[10px] font-mono w-10 text-right ${heatColor(sub.value)}`}>
                  {sub.value != null ? sub.value.toFixed(2) : '--'}
                </div>
              </div>
            ))}
          </div>
          {whale.cmi_status && (
            <div className="mt-2 text-[9px] text-slate-600 italic">
              CMI weight: {whale.cmi_status.derivatives_heat_weight}% — {whale.cmi_status.note}
            </div>
          )}
        </div>
      )}
    </GlassCard>
  );
}
