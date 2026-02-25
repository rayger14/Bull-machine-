import { useState } from 'react';
import { Info, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import type { MacroCorrelations as MacroCorrType } from '../../api/types';

interface MacroCorrelationsProps {
  data: MacroCorrType | null | undefined;
}

// Human-readable indicator metadata
const INDICATOR_META: Record<string, {
  name: string;
  desc: string;
  inverse: boolean; // true if positive correlation is bearish for BTC
}> = {
  dxy_btc: {
    name: 'Dollar Index (DXY)',
    desc: 'US dollar strength. A rising dollar typically pressures risk assets including BTC.',
    inverse: true,
  },
  vix_btc: {
    name: 'Volatility Index (VIX)',
    desc: 'Equity market fear gauge. Spikes often cause "sell everything" risk-off across crypto.',
    inverse: true,
  },
  gold_btc: {
    name: 'Gold',
    desc: 'Safe-haven asset. Both BTC and gold can rally together during currency debasement fears.',
    inverse: false,
  },
  oil_btc: {
    name: 'Oil (WTI Crude)',
    desc: 'Energy prices. Rising oil can signal inflation (mixed for BTC) or demand growth (positive).',
    inverse: false,
  },
  fg_btc: {
    name: 'Fear & Greed Index',
    desc: 'Crypto market sentiment (0=extreme fear, 100=extreme greed). Tracks crowd psychology.',
    inverse: false,
  },
  btc_d_btc: {
    name: 'BTC Dominance',
    desc: 'BTC market cap share. Rising dominance means capital flowing from altcoins into BTC.',
    inverse: false,
  },
  usdt_d_btc: {
    name: 'USDT Dominance',
    desc: 'Tether market share. Rising USDT.D means traders are moving to stablecoins (risk-off).',
    inverse: true,
  },
  usdc_d_btc: {
    name: 'USDC Dominance',
    desc: 'USDC market share. Similar to USDT.D — rising means capital exiting risk assets.',
    inverse: true,
  },
};

function strengthLabel(v: number | null): { text: string; color: string } {
  if (v == null) return { text: 'No data', color: 'text-slate-600' };
  const a = Math.abs(v);
  if (a < 0.1) return { text: 'None', color: 'text-slate-600' };
  if (a < 0.3) return { text: 'Weak', color: 'text-slate-400' };
  if (a < 0.5) return { text: 'Moderate', color: 'text-amber-400' };
  if (a < 0.7) return { text: 'Strong', color: 'text-orange-400' };
  return { text: 'Very strong', color: 'text-rose-400' };
}

function interpretation(key: string, v: number | null): string {
  if (v == null) return 'Awaiting data';
  const a = Math.abs(v);
  if (a < 0.1) return 'Moving independently of BTC right now';

  const meta = INDICATOR_META[key];
  const direction = v > 0 ? 'positive' : 'negative';
  const isExpected = meta?.inverse ? v < 0 : v > 0;

  if (a < 0.3) {
    return isExpected
      ? 'Behaving as expected — normal regime'
      : 'Slightly unusual relationship — worth monitoring';
  }
  if (a < 0.5) {
    return isExpected
      ? `Moderate ${direction} link — traditional relationship holding`
      : `Unusual ${direction} correlation — regime may be shifting`;
  }
  return isExpected
    ? `Strong ${direction} coupling — macro driving crypto`
    : `Abnormal ${direction} correlation — watch for regime change`;
}

function CorrBar({ value }: { value: number | null }) {
  if (value == null) return <div className="h-2 w-full bg-slate-800/50 rounded-full" />;
  // Bar from center: negative goes left, positive goes right
  const pct = Math.min(Math.abs(value) * 100, 100);
  const isNeg = value < 0;
  return (
    <div className="relative h-2 w-full bg-slate-800/50 rounded-full overflow-hidden">
      {/* Center marker */}
      <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-700" />
      {/* Bar */}
      <div
        className="absolute top-0 bottom-0 rounded-full transition-all duration-300"
        style={{
          background: isNeg
            ? 'linear-gradient(90deg, #f87171, #fb923c)'
            : 'linear-gradient(90deg, #34d399, #06b6d4)',
          width: `${pct / 2}%`,
          ...(isNeg
            ? { right: '50%' }
            : { left: '50%' }),
        }}
      />
    </div>
  );
}

export default function MacroCorrelations({ data }: MacroCorrelationsProps) {
  const [showInfo, setShowInfo] = useState(false);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  if (!data) return null;

  const w20 = data.window_20 ?? {};
  const w60 = data.window_60 ?? {};
  const nBars = data.n_bars;
  const nMacroBars = data.n_macro_bars;
  const minBars20 = data.min_bars_20 ?? 21;
  const minBars60 = data.min_bars_60 ?? 61;

  const isAwaitingMacro = data.regime === 'awaiting_macro';
  const isInsufficient = data.regime === 'insufficient_data' || (Object.keys(w20).length === 0 && Object.keys(w60).length === 0 && !isAwaitingMacro);

  if (isInsufficient || isAwaitingMacro) {
    const pct20 = nBars != null ? Math.min((nBars / minBars20) * 100, 100) : 0;
    const pct60 = nBars != null ? Math.min((nBars / minBars60) * 100, 100) : 0;

    return (
      <GlassCard>
        <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">Macro Correlations</div>
        <div className="space-y-3 py-2">
          <p className="text-xs text-slate-400 leading-relaxed">
            Tracks how sensitive BTC is to traditional markets right now. When correlations spike, macro is driving crypto.
          </p>
          <div className="bg-white/[0.02] rounded-lg px-3 py-2 border border-white/[0.04]">
            {isAwaitingMacro ? (
              <>
                <p className="text-xs text-cyan-400/80 font-medium mb-1">BTC Price Data Loaded</p>
                <p className="text-[11px] text-slate-500 leading-relaxed">
                  {nBars ?? 0} bars of BTC history loaded. Macro indicator data (DXY, VIX, Gold, Oil, F&G)
                  will be fetched from external APIs within the next hour.
                </p>
              </>
            ) : (
              <>
                <p className="text-xs text-amber-400/80 font-medium mb-1">Building Data</p>
                <p className="text-[11px] text-slate-500 leading-relaxed">
                  Collecting hourly data to compute meaningful correlations.
                  {nBars != null && ` ${nBars} bars so far.`}
                </p>
              </>
            )}
          </div>
          <div className="space-y-2">
            {[
              { label: 'Short-term (20h)', pct: isAwaitingMacro ? Math.min(((nMacroBars ?? 0) / minBars20) * 100, 100) : pct20, count: isAwaitingMacro ? nMacroBars ?? 0 : nBars ?? 0, need: minBars20 },
              { label: 'Medium-term (60h)', pct: isAwaitingMacro ? Math.min(((nMacroBars ?? 0) / minBars60) * 100, 100) : pct60, count: isAwaitingMacro ? nMacroBars ?? 0 : nBars ?? 0, need: minBars60 },
            ].map(({ label, pct: p, count, need }) => (
              <div key={label}>
                <div className="flex justify-between text-[10px] text-slate-600 mb-0.5">
                  <span>{label}</span>
                  <span>{count}/{need} bars</span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full rounded-full transition-all duration-500" style={{ width: `${p}%`, background: 'linear-gradient(90deg, #06b6d4, #10b981)' }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </GlassCard>
    );
  }

  // Order indicators by significance (highest absolute 20-bar correlation first)
  const orderedKeys = Object.keys(INDICATOR_META).filter(k => k in w20 || k in w60);
  orderedKeys.sort((a, b) => Math.abs(w20[b] ?? 0) - Math.abs(w20[a] ?? 0));

  return (
    <GlassCard>
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xs text-slate-500 uppercase tracking-wider">Macro Correlations</span>
        <button
          onClick={() => setShowInfo(!showInfo)}
          className="text-slate-600 hover:text-cyan-400 transition-colors"
        >
          <Info size={12} />
        </button>
        <span className="text-[10px] text-slate-600 ml-auto">
          {data.regime === 'stressed' ? (
            <span className="text-amber-400">Stressed regime</span>
          ) : (
            <span>Normal regime</span>
          )}
        </span>
      </div>

      {showInfo && (
        <div className="mb-3 bg-white/[0.02] rounded-lg px-3 py-2 border border-white/[0.04]">
          <p className="text-[11px] text-slate-400 leading-relaxed mb-1">
            Correlation measures how closely BTC moves with each macro indicator (-1 to +1).
          </p>
          <div className="text-[10px] text-slate-600 space-y-0.5">
            <p><span className="text-slate-400">Near 0:</span> BTC moving independently (good for diversification)</p>
            <p><span className="text-amber-400">0.3-0.5:</span> Moderate link — macro is influencing crypto</p>
            <p><span className="text-rose-400">&gt;0.5:</span> Strong coupling — macro is driving crypto price action</p>
          </div>
          <p className="text-[10px] text-slate-600 mt-1">
            <span className="text-slate-400">20h</span> = recent sensitivity, <span className="text-slate-400">60h</span> = medium-term trend
          </p>
        </div>
      )}

      <div className="space-y-1">
        {orderedKeys.map((key) => {
          const meta = INDICATOR_META[key];
          if (!meta) return null;
          const v20 = w20[key] ?? null;
          const v60 = w60[key] ?? null;
          const strength = strengthLabel(v20);
          const isExpanded = expandedRow === key;

          return (
            <div key={key}>
              <button
                className="w-full text-left px-2 py-2 rounded-lg hover:bg-white/[0.03] transition-colors"
                onClick={() => setExpandedRow(isExpanded ? null : key)}
              >
                <div className="flex items-center gap-3">
                  {/* Direction icon */}
                  <div className="w-4 shrink-0">
                    {v20 == null || Math.abs(v20) < 0.1 ? (
                      <Minus size={14} className="text-slate-600" />
                    ) : v20 > 0 ? (
                      <TrendingUp size={14} className="text-emerald-400" />
                    ) : (
                      <TrendingDown size={14} className="text-rose-400" />
                    )}
                  </div>

                  {/* Name + strength */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-300 truncate">{meta.name}</span>
                      <span className={`text-[10px] ${strength.color}`}>{strength.text}</span>
                    </div>
                    {/* Correlation bar */}
                    <div className="mt-1">
                      <CorrBar value={v20} />
                    </div>
                  </div>

                  {/* Values */}
                  <div className="flex gap-3 shrink-0 text-right">
                    <div>
                      <div className="text-[9px] text-slate-600">20h</div>
                      <div className="text-xs font-mono text-slate-300">
                        {v20 != null ? (v20 > 0 ? '+' : '') + v20.toFixed(2) : '--'}
                      </div>
                    </div>
                    <div>
                      <div className="text-[9px] text-slate-600">60h</div>
                      <div className="text-xs font-mono text-slate-400">
                        {v60 != null ? (v60 > 0 ? '+' : '') + v60.toFixed(2) : '--'}
                      </div>
                    </div>
                  </div>
                </div>
              </button>

              {/* Expanded detail */}
              {isExpanded && (
                <div className="px-2 pb-2 ml-7">
                  <div className="bg-white/[0.02] rounded-lg px-3 py-2 border border-white/[0.04] space-y-1.5">
                    <p className="text-[11px] text-slate-400 leading-relaxed">{meta.desc}</p>
                    <p className="text-[11px] text-slate-500 leading-relaxed">
                      <span className="text-slate-400">Current reading:</span>{' '}
                      {interpretation(key, v20)}
                    </p>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </GlassCard>
  );
}
