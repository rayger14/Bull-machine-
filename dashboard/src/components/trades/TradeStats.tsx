import { useMemo } from 'react';
import GlassCard from '../ui/GlassCard';
import type { Trade } from '../../api/types';

interface TradeStatsProps {
  trades: Trade[];
}

export default function TradeStats({ trades }: TradeStatsProps) {
  const stats = useMemo(() => {
    if (!trades.length) return null;
    const wins = trades.filter(t => (t.pnl ?? 0) > 0);
    const losses = trades.filter(t => (t.pnl ?? 0) <= 0);
    const grossProfit = wins.reduce((s, t) => s + (t.pnl ?? 0), 0);
    const grossLoss = Math.abs(losses.reduce((s, t) => s + (t.pnl ?? 0), 0));
    return {
      total: trades.length,
      winners: wins.length,
      losers: losses.length,
      winRate: (wins.length / trades.length * 100).toFixed(1),
      pf: grossLoss > 0 ? (grossProfit / grossLoss).toFixed(2) : 'Inf',
      totalPnl: trades.reduce((s, t) => s + (t.pnl ?? 0), 0).toFixed(2),
      avgWin: wins.length ? (grossProfit / wins.length).toFixed(2) : '0',
      avgLoss: losses.length ? (grossLoss / losses.length).toFixed(2) : '0',
      avgDuration: (trades.reduce((s, t) => s + (t.duration_hours ?? 0), 0) / trades.length).toFixed(1),
    };
  }, [trades]);

  if (!stats) {
    return (
      <GlassCard>
        <div className="text-center text-slate-600 py-8">No trades yet</div>
      </GlassCard>
    );
  }

  const items = [
    { label: 'Total', value: stats.total, color: 'text-slate-200' },
    { label: 'Winners', value: stats.winners, color: 'text-emerald-400' },
    { label: 'Losers', value: stats.losers, color: 'text-rose-400' },
    { label: 'Win Rate', value: stats.winRate + '%', color: parseFloat(stats.winRate) >= 50 ? 'text-emerald-400' : 'text-rose-400' },
    { label: 'Profit Factor', value: stats.pf, color: parseFloat(stats.pf) >= 1 ? 'text-emerald-400' : 'text-rose-400' },
    { label: 'Net PnL', value: '$' + stats.totalPnl, color: parseFloat(stats.totalPnl) >= 0 ? 'text-emerald-400' : 'text-rose-400' },
    { label: 'Avg Win', value: '$' + stats.avgWin, color: 'text-emerald-400' },
    { label: 'Avg Loss', value: '$' + stats.avgLoss, color: 'text-rose-400' },
  ];

  return (
    <GlassCard>
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">Trade Summary</div>
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
        {items.map((item) => (
          <div key={item.label} className="text-center">
            <div className={`text-lg font-bold font-mono ${item.color}`}>{item.value}</div>
            <div className="text-[10px] text-slate-600">{item.label}</div>
          </div>
        ))}
      </div>
    </GlassCard>
  );
}
