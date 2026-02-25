import { useMemo } from 'react';
import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import type { Trade } from '../../api/types';

interface TradesByArchetypeProps {
  trades: Trade[];
}

export default function TradesByArchetype({ trades }: TradesByArchetypeProps) {
  const rows = useMemo(() => {
    const map: Record<string, { trades: number; wins: number; totalPnl: number; grossProfit: number; grossLoss: number; totalDuration: number }> = {};
    trades.forEach(t => {
      const a = t.archetype ?? 'unknown';
      if (!map[a]) map[a] = { trades: 0, wins: 0, totalPnl: 0, grossProfit: 0, grossLoss: 0, totalDuration: 0 };
      map[a].trades++;
      const pnl = t.pnl ?? 0;
      if (pnl > 0) { map[a].wins++; map[a].grossProfit += pnl; }
      else { map[a].grossLoss += Math.abs(pnl); }
      map[a].totalPnl += pnl;
      map[a].totalDuration += t.duration_hours ?? 0;
    });
    return Object.entries(map)
      .map(([name, a]) => ({
        name,
        trades: a.trades,
        winRate: (a.wins / a.trades * 100).toFixed(0),
        pf: a.grossLoss > 0 ? (a.grossProfit / a.grossLoss).toFixed(2) : 'Inf',
        totalPnl: a.totalPnl.toFixed(2),
        avgDuration: (a.totalDuration / a.trades).toFixed(1),
      }))
      .sort((a, b) => parseFloat(b.totalPnl) - parseFloat(a.totalPnl));
  }, [trades]);

  if (rows.length === 0) return null;

  return (
    <GlassCard>
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">Performance by Archetype</div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-600 border-b border-white/[0.05]">
            <th className="text-left py-2 font-medium">Archetype</th>
            <th className="text-right py-2 font-medium">Trades</th>
            <th className="text-right py-2 font-medium">Win Rate</th>
            <th className="text-right py-2 font-medium">PF</th>
            <th className="text-right py-2 font-medium">PnL</th>
            <th className="text-right py-2 font-medium">Avg Duration</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name} className="border-b border-white/[0.03] hover:bg-white/[0.02]">
              <td className="py-2">
                <Badge variant="cyan">{r.name.replace(/_/g, ' ')}</Badge>
              </td>
              <td className="py-2 text-right font-mono text-slate-400">{r.trades}</td>
              <td className="py-2 text-right font-mono text-slate-300">{r.winRate}%</td>
              <td className={`py-2 text-right font-mono ${parseFloat(r.pf) >= 1 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {r.pf}
              </td>
              <td className={`py-2 text-right font-mono ${parseFloat(r.totalPnl) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                ${r.totalPnl}
              </td>
              <td className="py-2 text-right font-mono text-slate-500">{r.avgDuration}h</td>
            </tr>
          ))}
        </tbody>
      </table>
    </GlassCard>
  );
}
