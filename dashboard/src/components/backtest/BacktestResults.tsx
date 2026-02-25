import { useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
} from 'recharts';
import GlassCard from '../ui/GlassCard';
import type { BacktestResult } from '../../api/types';

interface BacktestResultsProps {
  result: BacktestResult;
}

export default function BacktestResults({ result }: BacktestResultsProps) {
  const { stats, breakdown, equity } = result;
  const [showColumns, setShowColumns] = useState(false);

  const equityData = useMemo(() => {
    return equity.map((r) => ({
      label: new Date(r.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: '2-digit', timeZone: 'America/Los_Angeles' }),
      equity: parseFloat(r.equity),
    }));
  }, [equity]);

  const statItems = [
    { label: 'Total Trades', value: stats.total_trades },
    { label: 'Win Rate', value: stats.win_rate != null ? `${Number(stats.win_rate).toFixed(1)}%` : '--' },
    { label: 'Profit Factor', value: stats.profit_factor != null ? Number(stats.profit_factor).toFixed(2) : '--' },
    { label: 'Return', value: stats.total_return_pct != null ? `${Number(stats.total_return_pct).toFixed(1)}%` : '--' },
    { label: 'Max DD', value: stats.max_drawdown_pct != null ? `${Number(stats.max_drawdown_pct).toFixed(1)}%` : '--' },
    { label: 'Sharpe', value: stats.sharpe_ratio != null ? Number(stats.sharpe_ratio).toFixed(2) : '--' },
  ];

  return (
    <div className="space-y-4">
      {/* Stats grid */}
      <GlassCard>
        <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">Backtest Results</div>
        <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
          {statItems.map((item) => (
            <div key={item.label} className="text-center">
              <div className="text-lg font-bold font-mono text-slate-200">{item.value ?? '--'}</div>
              <div className="text-[10px] text-slate-600">{item.label}</div>
            </div>
          ))}
        </div>
      </GlassCard>

      {/* Equity curve */}
      {equityData.length > 0 && (
        <GlassCard>
          <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">Equity Curve</div>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={equityData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
              <defs>
                <linearGradient id="btEqGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#4da6ff" stopOpacity={0.1} />
                  <stop offset="95%" stopColor="#4da6ff" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="label" tick={{ fill: '#3d4f63', fontSize: 10 }} tickLine={false} axisLine={{ stroke: '#1e2738' }} interval="preserveStartEnd" />
              <YAxis tick={{ fill: '#6b7a8d', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={(v: number) => `$${(v / 1000).toFixed(1)}k`} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, fontSize: 11 }} />
              <Area type="monotone" dataKey="equity" stroke="#4da6ff" fill="url(#btEqGrad)" strokeWidth={2} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </GlassCard>
      )}

      {/* Archetype breakdown */}
      {breakdown.length > 0 && (
        <GlassCard>
          <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">Archetype Breakdown</div>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-600 border-b border-white/[0.05]">
                {Object.keys(breakdown[0]).map((k) => (
                  <th key={k} className="text-left py-1.5 font-medium">{k.replace(/_/g, ' ')}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {breakdown.map((row, i) => (
                <tr key={i} className="border-b border-white/[0.03]">
                  {Object.values(row).map((v, j) => (
                    <td key={j} className="py-1.5 text-slate-400 font-mono">{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </GlassCard>
      )}

      {/* Feature Store Metadata */}
      {result.feature_store && result.feature_store.columns > 0 && (
        <GlassCard>
          <div className="flex items-center justify-between mb-3">
            <div className="text-xs text-slate-500 uppercase tracking-wider">Feature Store</div>
            <div className="flex items-center gap-2 text-[10px] text-slate-600">
              <span className="font-mono">{result.feature_store.columns} cols</span>
              <span>&times;</span>
              <span className="font-mono">{result.feature_store.rows.toLocaleString()} rows</span>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3 text-xs mb-3">
            <div>
              <span className="text-slate-600">File: </span>
              <span className="text-slate-400 font-mono text-[10px] break-all">
                {result.feature_store.path.split('/').slice(-1)[0]}
              </span>
            </div>
            {result.feature_store.date_range && (
              <div>
                <span className="text-slate-600">Range: </span>
                <span className="text-slate-400 font-mono text-[10px]">{result.feature_store.date_range}</span>
              </div>
            )}
          </div>
          <button
            onClick={() => setShowColumns(!showColumns)}
            className="text-[10px] text-blue-400/70 hover:text-blue-400 transition-colors cursor-pointer"
          >
            {showColumns ? 'Hide' : 'Show'} all {result.feature_store.columns} columns
          </button>
          {showColumns && (
            <div className="mt-2 p-2 bg-white/[0.02] rounded border border-white/[0.04] max-h-60 overflow-y-auto">
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-x-3 gap-y-0.5">
                {result.feature_store.column_list.map((col) => (
                  <span key={col} className="text-[9px] font-mono text-slate-500 truncate">{col}</span>
                ))}
              </div>
            </div>
          )}
        </GlassCard>
      )}

      {/* Backtest Parameters */}
      {result.params && (
        <GlassCard>
          <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">Parameters Used</div>
          <div className="grid grid-cols-3 sm:grid-cols-6 gap-3 text-xs">
            {result.params.capital != null && (
              <div className="text-center">
                <div className="text-sm font-bold font-mono text-slate-300">${Number(result.params.capital).toLocaleString()}</div>
                <div className="text-[10px] text-slate-600">Capital</div>
              </div>
            )}
            {result.params.leverage != null && (
              <div className="text-center">
                <div className="text-sm font-bold font-mono text-slate-300">{result.params.leverage}x</div>
                <div className="text-[10px] text-slate-600">Leverage</div>
              </div>
            )}
            {result.params.commission != null && (
              <div className="text-center">
                <div className="text-sm font-bold font-mono text-slate-300">{(Number(result.params.commission) * 10000).toFixed(0)} bps</div>
                <div className="text-[10px] text-slate-600">Commission</div>
              </div>
            )}
            {result.params.slippage != null && (
              <div className="text-center">
                <div className="text-sm font-bold font-mono text-slate-300">{result.params.slippage} bps</div>
                <div className="text-[10px] text-slate-600">Slippage</div>
              </div>
            )}
            {result.params.start_date && (
              <div className="text-center">
                <div className="text-sm font-bold font-mono text-slate-300">{result.params.start_date}</div>
                <div className="text-[10px] text-slate-600">Start</div>
              </div>
            )}
            {result.params.end_date && (
              <div className="text-center">
                <div className="text-sm font-bold font-mono text-slate-300">{result.params.end_date}</div>
                <div className="text-[10px] text-slate-600">End</div>
              </div>
            )}
          </div>
        </GlassCard>
      )}
    </div>
  );
}
