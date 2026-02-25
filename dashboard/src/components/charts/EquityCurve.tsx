import { useMemo } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from 'recharts';
import GlassCard from '../ui/GlassCard';
import { colors } from '../../theme/colors';
import type { EquityRow } from '../../api/types';

interface EquityCurveProps {
  data: EquityRow[] | undefined;
}

export default function EquityCurve({ data }: EquityCurveProps) {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];
    return data.map((r) => ({
      label: shortLabel(r.timestamp),
      equity: parseFloat(r.equity),
      btc: parseFloat(r.btc_price),
      threshold: parseFloat(r.threshold ?? '0'),
      regime: r.regime ?? 'neutral',
    }));
  }, [data]);

  if (chartData.length === 0) {
    return (
      <GlassCard>
        <div className="text-xs text-slate-500 mb-2">Equity Curve</div>
        <div className="h-64 flex items-center justify-center text-slate-600 text-sm">No data yet</div>
      </GlassCard>
    );
  }

  return (
    <GlassCard>
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">Equity Curve</div>
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={chartData} margin={{ top: 5, right: 50, left: 10, bottom: 5 }}>
          <defs>
            <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={colors.chart.equity} stopOpacity={0.1} />
              <stop offset="95%" stopColor={colors.chart.equity} stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="label"
            tick={{ fill: '#3d4f63', fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: '#1e2738' }}
            interval="preserveStartEnd"
          />
          <YAxis
            yAxisId="eq"
            tick={{ fill: '#6b7a8d', fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v: number) => `$${(v/1000).toFixed(1)}k`}
          />
          <YAxis
            yAxisId="btc"
            orientation="right"
            tick={{ fill: '#fbbf24', fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v: number) => `$${(v/1000).toFixed(0)}k`}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, fontSize: 11 }}
            labelStyle={{ color: '#94a3b8' }}
          />
          <Legend
            wrapperStyle={{ fontSize: 11, color: '#6b7a8d' }}
          />
          <Area
            yAxisId="eq"
            type="monotone"
            dataKey="equity"
            stroke={colors.chart.equity}
            fill="url(#eqGrad)"
            strokeWidth={2}
            dot={false}
            name="Equity"
          />
          <Line
            yAxisId="btc"
            type="monotone"
            dataKey="btc"
            stroke={colors.chart.btc}
            strokeWidth={1}
            dot={false}
            name="BTC"
          />
          <Line
            yAxisId="eq"
            type="monotone"
            dataKey="threshold"
            stroke={colors.chart.threshold}
            strokeWidth={1.5}
            strokeDasharray="4 3"
            dot={false}
            name="Threshold"
            hide
          />
        </ComposedChart>
      </ResponsiveContainer>
    </GlassCard>
  );
}

function shortLabel(ts: string): string {
  const d = new Date(ts);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/Los_Angeles' });
}
