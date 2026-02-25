import { useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from 'recharts';
import GlassCard from '../ui/GlassCard';
import { colors } from '../../theme/colors';
import type { EquityRow } from '../../api/types';

interface CMIHistoryChartProps {
  data: EquityRow[] | undefined;
}

export default function CMIHistoryChart({ data }: CMIHistoryChartProps) {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];
    return data.map((r) => ({
      label: shortLabel(r.timestamp),
      risk_temp: parseFloat(r.risk_temp ?? '0'),
      instability: parseFloat(r.instability ?? '0'),
      crisis_prob: parseFloat(r.crisis_prob ?? '0'),
      threshold: parseFloat(r.threshold ?? '0'),
    }));
  }, [data]);

  if (chartData.length === 0) return null;

  return (
    <GlassCard>
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">CMI Components</div>
      <ResponsiveContainer width="100%" height={240}>
        <AreaChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <XAxis
            dataKey="label"
            tick={{ fill: '#3d4f63', fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: '#1e2738' }}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fill: '#6b7a8d', fontSize: 10 }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, fontSize: 11 }}
            labelStyle={{ color: '#94a3b8' }}
          />
          <Legend wrapperStyle={{ fontSize: 11, color: '#6b7a8d' }} />
          <Area
            type="monotone"
            dataKey="risk_temp"
            stroke={colors.chart.riskTemp}
            fill={colors.chart.riskTemp}
            fillOpacity={0.06}
            strokeWidth={1.5}
            dot={false}
            name="Risk Temp"
          />
          <Area
            type="monotone"
            dataKey="instability"
            stroke={colors.chart.instability}
            fill={colors.chart.instability}
            fillOpacity={0.06}
            strokeWidth={1.5}
            dot={false}
            name="Instability"
          />
          <Area
            type="monotone"
            dataKey="crisis_prob"
            stroke={colors.chart.crisisProb}
            fill={colors.chart.crisisProb}
            fillOpacity={0.06}
            strokeWidth={1.5}
            dot={false}
            name="Crisis Prob"
          />
          <Line
            type="monotone"
            dataKey="threshold"
            stroke={colors.chart.threshold}
            strokeWidth={1.5}
            strokeDasharray="4 3"
            dot={false}
            name="Threshold"
          />
        </AreaChart>
      </ResponsiveContainer>
    </GlassCard>
  );
}

function shortLabel(ts: string): string {
  const d = new Date(ts);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/Los_Angeles' });
}
