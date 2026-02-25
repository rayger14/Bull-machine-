import { useState } from 'react';
import { Info } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import type { CointegrationData } from '../../api/types';

interface CointegrationTableProps {
  data: CointegrationData | null | undefined;
}

const PAIR_NAMES: Record<string, { name: string; desc: string }> = {
  'BTC-DXY': {
    name: 'BTC vs Dollar Index',
    desc: 'When cointegrated, BTC and the dollar have a mean-reverting spread — deviations signal opportunities.',
  },
  'BTC-Gold': {
    name: 'BTC vs Gold',
    desc: 'Both are "hard money" assets. Cointegration means they track each other and divergences tend to close.',
  },
  'BTC-VIX': {
    name: 'BTC vs VIX (Fear)',
    desc: 'Links crypto to equity market fear. Cointegration suggests BTC sells off predictably with VIX spikes.',
  },
  'BTC-Oil': {
    name: 'BTC vs Oil',
    desc: 'Connects crypto to energy/inflation cycle. Cointegration means they share an economic driver.',
  },
  'BTC-USDT.D': {
    name: 'BTC vs USDT Dominance',
    desc: 'Inverse risk-on/risk-off pair. USDT.D rises when traders flee to safety — cointegration means this is predictable.',
  },
  'BTC-BTC.D': {
    name: 'BTC vs BTC Dominance',
    desc: 'BTC dominance vs BTC price. Cointegration suggests capital rotation between BTC and altcoins is mean-reverting.',
  },
};

function signalText(p: { cointegrated?: boolean; signal?: string; current_zscore?: number | null; stability?: string }): { text: string; color: string } {
  if (p.stability === 'insufficient_data' || p.signal?.includes('failed')) {
    return { text: 'Insufficient variance in data', color: 'text-slate-600' };
  }
  if (!p.cointegrated) {
    return { text: 'No statistical link detected', color: 'text-slate-500' };
  }
  const z = p.current_zscore ?? 0;
  if (Math.abs(z) > 2) {
    return { text: z > 0 ? 'Spread stretched high — potential mean reversion down' : 'Spread stretched low — potential mean reversion up', color: 'text-amber-400' };
  }
  return { text: 'Cointegrated — spread within normal range', color: 'text-emerald-400' };
}

export default function CointegrationTable({ data }: CointegrationTableProps) {
  const [showInfo, setShowInfo] = useState(false);

  if (!data) return null;

  const hasPairs = data.pairs && data.pairs.length > 0;
  const hasOpp = data.has_opportunity || (data.pairs?.some(p => p.has_opportunity) ?? false);

  // Filter out pairs with failed regressions for cleaner display
  const validPairs = data.pairs?.filter(p => p.stability !== 'insufficient_data' && !p.signal?.includes('failed')) ?? [];
  const failedPairs = data.pairs?.filter(p => p.stability === 'insufficient_data' || p.signal?.includes('failed')) ?? [];

  return (
    <GlassCard>
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xs text-slate-500 uppercase tracking-wider">Cointegration Analysis</span>
        <button
          onClick={() => setShowInfo(!showInfo)}
          className="text-slate-600 hover:text-cyan-400 transition-colors"
        >
          <Info size={12} />
        </button>
        {hasOpp && <Badge variant="green">Opportunity</Badge>}
        <span className="text-[10px] text-slate-600 ml-auto">
          {data.n_bars_available ?? '--'}/{data.min_bars_required ?? '--'} bars
        </span>
      </div>

      {showInfo && (
        <div className="mb-3 bg-white/[0.02] rounded-lg px-3 py-2 border border-white/[0.04] space-y-1.5">
          <p className="text-[11px] text-slate-400 leading-relaxed">
            Cointegration tests whether BTC and a macro indicator are mathematically linked over time.
            Unlike correlation (which measures direction), cointegration finds pairs that <span className="text-slate-300">always return to equilibrium</span>.
          </p>
          <p className="text-[11px] text-slate-500">
            When a cointegrated pair's spread deviates beyond 2 standard deviations, it historically snaps back — creating a tradeable signal.
          </p>
          <div className="text-[10px] text-slate-600 space-y-0.5">
            <p><span className="text-emerald-400">Cointegrated:</span> Statistical link confirmed (p &lt; 0.05). Spread deviations tend to revert.</p>
            <p><span className="text-slate-400">Half-life:</span> How quickly the spread reverts (shorter = faster mean reversion).</p>
            <p><span className="text-amber-400">Z-Score &gt; 2:</span> Spread is stretched — potential reversion trade opportunity.</p>
          </div>
        </div>
      )}

      {hasPairs ? (
        <div className="space-y-2">
          {/* Valid pairs with results */}
          {validPairs.map((p) => {
            const meta = PAIR_NAMES[p.pair ?? ''] ?? { name: p.pair ?? 'Unknown', desc: '' };
            const sig = signalText(p);
            return (
              <div key={p.pair} className="bg-white/[0.02] rounded-lg px-3 py-2.5 border border-white/[0.04]">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs text-slate-300">{meta.name}</span>
                  {p.cointegrated ? (
                    <Badge variant="green">Linked</Badge>
                  ) : (
                    <Badge variant="neutral">Independent</Badge>
                  )}
                </div>
                <p className={`text-[11px] ${sig.color} mb-1.5`}>{sig.text}</p>
                {p.cointegrated && (
                  <div className="flex gap-4 text-[10px]">
                    {(p.half_life_hours ?? p.half_life) != null && (
                      <div>
                        <span className="text-slate-600">Half-life: </span>
                        <span className="text-slate-400 font-mono">{(p.half_life_hours ?? p.half_life)!.toFixed(0)}h</span>
                      </div>
                    )}
                    {p.current_zscore != null && (
                      <div>
                        <span className="text-slate-600">Z-Score: </span>
                        <span className={`font-mono ${Math.abs(p.current_zscore) > 2 ? 'text-amber-400' : 'text-slate-400'}`}>
                          {p.current_zscore.toFixed(2)}
                        </span>
                      </div>
                    )}
                    {p.p_value != null && (
                      <div>
                        <span className="text-slate-600">p-value: </span>
                        <span className="text-slate-500 font-mono">{p.p_value.toFixed(3)}</span>
                      </div>
                    )}
                  </div>
                )}
                {!p.cointegrated && p.p_value != null && (
                  <div className="text-[10px] text-slate-600">
                    p-value: <span className="font-mono">{p.p_value.toFixed(3)}</span>
                    {p.p_value < 0.2 && <span className="text-slate-500 ml-1">(approaching significance)</span>}
                  </div>
                )}
              </div>
            );
          })}

          {/* Failed pairs summary (collapsed) */}
          {failedPairs.length > 0 && (
            <div className="text-[10px] text-slate-600 px-1 pt-1">
              {failedPairs.map(p => PAIR_NAMES[p.pair ?? '']?.name ?? p.pair).join(', ')} — insufficient
              variance for regression (constant values from snapshot data, will improve with live data)
            </div>
          )}
        </div>
      ) : (
        <div className="text-xs text-slate-600 text-center py-4">
          No cointegration pairs available yet
        </div>
      )}
    </GlassCard>
  );
}
