import { useMemo } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from 'recharts';
import GlassCard from '../ui/GlassCard';
import type { FactorAttrBucket } from '../../api/types';

interface FactorAttributionProps {
  summary: Record<string, FactorAttrBucket> | null | undefined;
}

const factorColors: Record<string, string> = {
  technical: '#34d399',
  liquidity: '#3b82f6',
  macro: '#fb923c',
  regime: '#8b5cf6',
};

export default function FactorAttribution({ summary }: FactorAttributionProps) {
  const chartData = useMemo(() => {
    if (!summary) return [];
    const factors = ['technical', 'liquidity', 'macro', 'regime'];
    return [
      {
        name: 'Overall',
        ...Object.fromEntries(factors.map(f => [f, (summary[f]?.avg_pct ?? 0) * 100])),
      },
      {
        name: 'Winners',
        ...Object.fromEntries(factors.map(f => [f, (summary[f]?.win_contribution ?? 0) * 100])),
      },
      {
        name: 'Losers',
        ...Object.fromEntries(factors.map(f => [f, (summary[f]?.loss_contribution ?? 0) * 100])),
      },
    ];
  }, [summary]);

  if (chartData.length === 0) return null;

  return (
    <GlassCard>
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">Factor Attribution</div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <XAxis dataKey="name" tick={{ fill: '#6b7a8d', fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: '#6b7a8d', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${v.toFixed(0)}%`} />
          <Tooltip
            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, fontSize: 11 }}
            formatter={(v: number | undefined) => `${(v ?? 0).toFixed(1)}%`}
          />
          <Legend wrapperStyle={{ fontSize: 11, color: '#6b7a8d' }} />
          <Bar dataKey="technical" stackId="a" fill={factorColors.technical} name="Technical" radius={[0, 0, 0, 0]} />
          <Bar dataKey="liquidity" stackId="a" fill={factorColors.liquidity} name="Liquidity" />
          <Bar dataKey="macro" stackId="a" fill={factorColors.macro} name="Macro" />
          <Bar dataKey="regime" stackId="a" fill={factorColors.regime} name="Regime" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </GlassCard>
  );
}
