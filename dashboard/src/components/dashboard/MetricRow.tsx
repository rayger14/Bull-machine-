import { TrendingUp, Shield, Wallet, BarChart3 } from 'lucide-react';
import StatCard from '../ui/StatCard';
import Badge from '../ui/Badge';
import { fmt, fmtUsd, fmtPct } from '../../utils/format';
import type { Heartbeat, Performance } from '../../api/types';

interface MetricRowProps {
  hb: Heartbeat;
  performance?: Performance | null;
}

export default function MetricRow({ hb, performance }: MetricRowProps) {
  const regimeVariant = (r?: string) => {
    if (r === 'bull') return 'green';
    if (r === 'bear' || r === 'crisis') return 'red';
    if (r === 'stagflation') return 'orange';
    if (r === 'neutral') return 'yellow';
    return 'cyan';
  };

  // Prefer performance object (from /api/status), fall back to heartbeat
  const pf = (performance?.profit_factor as number | undefined) ?? hb.profit_factor;
  const wr = (performance?.win_rate_pct as number | undefined) ?? hb.win_rate;
  const returnPct = (performance?.total_return_pct as number | undefined) ?? hb.return_pct;

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <StatCard
        label="BTC Price"
        value={`$${fmt(hb.btc_price, 0)}`}
        valueColor="text-amber-400"
        icon={<TrendingUp className="w-4 h-4" />}
        accent
      />
      <div className="bg-white/[0.03] backdrop-blur-xl border border-white/[0.08] rounded-2xl p-4 hover:bg-white/[0.06] hover:border-white/[0.15] transition-all duration-300">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs text-slate-500 uppercase tracking-wider">Regime</span>
          <Shield className="w-4 h-4 text-slate-500" />
        </div>
        <div className="flex items-center gap-2 mt-2">
          <Badge variant={regimeVariant(hb.regime)}>
            {(hb.regime || 'unknown').replace('_', ' ').toUpperCase()}
          </Badge>
          {hb.leverage && (
            <span className="text-xs text-slate-500">{hb.leverage}x</span>
          )}
        </div>
        <div className="text-xs text-slate-500 mt-2">
          Threshold: {hb.threshold?.toFixed(2) ?? '--'}
        </div>
      </div>
      <StatCard
        label="Portfolio"
        value={fmtUsd(hb.equity, 0)}
        sub={returnPct != null ? `${returnPct > 0 ? '+' : ''}${fmtPct(returnPct)}` : undefined}
        valueColor={returnPct != null && returnPct >= 0 ? 'text-emerald-400' : 'text-rose-400'}
        icon={<Wallet className="w-4 h-4" />}
      />
      <StatCard
        label="Profit Factor"
        value={pf != null ? pf.toFixed(2) : '--'}
        sub={wr != null ? `WR: ${fmtPct(wr)}` : undefined}
        valueColor={pf != null && pf >= 1.0 ? 'text-emerald-400' : 'text-rose-400'}
        icon={<BarChart3 className="w-4 h-4" />}
      />
    </div>
  );
}
