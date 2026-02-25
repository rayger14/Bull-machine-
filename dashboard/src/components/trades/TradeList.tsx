import { useState, useMemo } from 'react';
import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import FilterPills from '../ui/FilterPills';
import { CounterfactualPanel } from './CounterfactualPanel';
import { fmtUsd, shortDate } from '../../utils/format';
import type { Trade } from '../../api/types';

interface TradeListProps {
  trades: Trade[];
}

const statusFilters = [
  { value: 'all', label: 'All' },
  { value: 'winners', label: 'Winners' },
  { value: 'losers', label: 'Losers' },
];

const sortOptions = [
  { value: 'time_desc', label: 'Newest' },
  { value: 'time_asc', label: 'Oldest' },
  { value: 'pnl_desc', label: 'Best PnL' },
  { value: 'pnl_asc', label: 'Worst PnL' },
];

const COL_COUNT = 11;

function computeRiskReward(trade: Trade): string {
  const entry = trade.entry_price;
  const sl = trade.stop_loss;
  const tp = trade.take_profit;
  if (entry == null || sl == null || tp == null) return '-';
  const risk = Math.abs(entry - sl);
  if (risk === 0) return '-';
  const reward = Math.abs(tp - entry);
  return (reward / risk).toFixed(2);
}

function TradeDetail({ trade }: { trade: Trade }) {
  return (
    <tr>
      <td colSpan={COL_COUNT} className="px-3 py-3 bg-gray-800/50 border-t border-gray-700">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-2 font-semibold">Trade Attribution</div>
        <div className="grid grid-cols-3 gap-x-8 gap-y-2 text-xs">
          {/* Column 1: Price levels */}
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-500">Stop Loss</span>
              <span className="font-mono text-slate-300">
                {trade.stop_loss != null ? `$${trade.stop_loss.toFixed(2)}` : '-'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Take Profit</span>
              <span className="font-mono text-slate-300">
                {trade.take_profit != null ? `$${trade.take_profit.toFixed(2)}` : '-'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Risk:Reward</span>
              <span className="font-mono text-cyan-400">{computeRiskReward(trade)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">ATR at Entry</span>
              <span className="font-mono text-slate-300">
                {trade.atr_at_entry != null ? trade.atr_at_entry.toFixed(2) : '-'}
              </span>
            </div>
          </div>

          {/* Column 2: Position sizing */}
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-500">Position Size</span>
              <span className="font-mono text-slate-300">
                {trade.position_size_usd != null ? fmtUsd(trade.position_size_usd) : '-'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Leverage</span>
              <span className="font-mono text-slate-300">
                {trade.leverage_applied != null ? `${trade.leverage_applied.toFixed(1)}x` : '-'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Threshold at Entry</span>
              <span className="font-mono text-slate-300">
                {trade.threshold_at_entry != null ? trade.threshold_at_entry.toFixed(3) : (trade.threshold != null ? trade.threshold.toFixed(3) : '-')}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Threshold Margin</span>
              <span className={`font-mono ${
                (trade.threshold_margin ?? 0) > 0.05 ? 'text-emerald-400' :
                (trade.threshold_margin ?? 0) > 0 ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {trade.threshold_margin != null ? `+${trade.threshold_margin.toFixed(3)}` : '-'}
              </span>
            </div>
          </div>

          {/* Column 3: CMI components */}
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-500">Risk Temp</span>
              <span className="font-mono text-slate-300">
                {trade.risk_temp_at_entry != null ? trade.risk_temp_at_entry.toFixed(3) : '-'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Instability</span>
              <span className="font-mono text-slate-300">
                {trade.instability_at_entry != null ? trade.instability_at_entry.toFixed(3) : '-'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Crisis Prob</span>
              <span className={`font-mono ${
                (trade.crisis_prob_at_entry ?? 0) > 0.5 ? 'text-rose-400' :
                (trade.crisis_prob_at_entry ?? 0) > 0.3 ? 'text-yellow-400' : 'text-slate-300'
              }`}>
                {trade.crisis_prob_at_entry != null ? trade.crisis_prob_at_entry.toFixed(3) : '-'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Exit Reason</span>
              <span className="font-mono text-slate-300 truncate max-w-[140px]" title={trade.exit_reason || ''}>
                {trade.exit_reason || '-'}
              </span>
            </div>
          </div>
        </div>
        {trade.counterfactual && (
          <CounterfactualPanel data={trade.counterfactual} />
        )}
      </td>
    </tr>
  );
}

export default function TradeList({ trades }: TradeListProps) {
  const [filter, setFilter] = useState('all');
  const [sort, setSort] = useState('time_desc');
  const [archFilter, setArchFilter] = useState('all');
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  const archetypes = useMemo(() => {
    const s = new Set(trades.map(t => t.archetype ?? 'unknown'));
    return ['all', ...Array.from(s).sort()];
  }, [trades]);

  const filtered = useMemo(() => {
    let t = [...trades];
    if (filter === 'winners') t = t.filter(x => (x.pnl ?? 0) > 0);
    if (filter === 'losers') t = t.filter(x => (x.pnl ?? 0) <= 0);
    if (archFilter !== 'all') t = t.filter(x => x.archetype === archFilter);
    if (sort === 'time_desc') t.sort((a, b) => (b.timestamp_entry ?? '').localeCompare(a.timestamp_entry ?? ''));
    else if (sort === 'time_asc') t.sort((a, b) => (a.timestamp_entry ?? '').localeCompare(b.timestamp_entry ?? ''));
    else if (sort === 'pnl_desc') t.sort((a, b) => (b.pnl ?? 0) - (a.pnl ?? 0));
    else if (sort === 'pnl_asc') t.sort((a, b) => (a.pnl ?? 0) - (b.pnl ?? 0));
    return t;
  }, [trades, filter, sort, archFilter]);

  return (
    <GlassCard>
      <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
        <span className="text-xs text-slate-500 uppercase tracking-wider">Trade History ({filtered.length})</span>
        <div className="flex gap-2 flex-wrap">
          <FilterPills options={statusFilters} selected={filter} onChange={setFilter} />
          <FilterPills options={sortOptions} selected={sort} onChange={setSort} />
        </div>
      </div>
      {archetypes.length > 2 && (
        <div className="mb-3">
          <FilterPills
            options={archetypes.map(a => ({ value: a, label: a === 'all' ? 'All Archetypes' : a.replace(/_/g, ' ') }))}
            selected={archFilter}
            onChange={setArchFilter}
          />
        </div>
      )}
      <div className="max-h-[600px] overflow-y-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-[#030712] z-10">
            <tr className="text-slate-600 border-b border-white/[0.05]">
              <th className="text-left px-3 py-2 font-medium">Time</th>
              <th className="text-left px-3 py-2 font-medium">Archetype</th>
              <th className="text-left px-3 py-2 font-medium">Regime</th>
              <th className="text-right px-3 py-2 font-medium">Entry</th>
              <th className="text-right px-3 py-2 font-medium">Exit</th>
              <th className="text-right px-3 py-2 font-medium">PnL</th>
              <th className="text-right px-3 py-2 font-medium">Duration</th>
              <th className="text-right px-3 py-2 font-medium">Fusion</th>
              <th className="text-left px-3 py-2 font-medium">Exit Reason</th>
              <th className="text-right px-3 py-2 font-medium">Threshold</th>
              <th className="text-right px-3 py-2 font-medium">Margin</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((t, i) => {
              const isExpanded = expandedIdx === i;
              return (
                <>
                  <tr
                    key={`row-${i}`}
                    className={`border-b border-white/[0.03] hover:bg-white/[0.02] cursor-pointer ${isExpanded ? 'bg-white/[0.02]' : ''}`}
                    onClick={() => setExpandedIdx(isExpanded ? null : i)}
                  >
                    <td className="px-3 py-2 text-slate-400">{shortDate(t.timestamp_entry)}</td>
                    <td className="px-3 py-2">
                      <Badge variant="cyan">{(t.archetype ?? '').replace(/_/g, ' ')}</Badge>
                    </td>
                    <td className="px-3 py-2">
                      <span className={`text-xs font-medium ${
                        t.regime === 'bull' ? 'text-emerald-400' :
                        t.regime === 'bear' ? 'text-rose-400' :
                        'text-slate-400'
                      }`}>
                        {t.regime || '-'}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-slate-300">${(t.entry_price ?? 0).toFixed(0)}</td>
                    <td className="px-3 py-2 text-right font-mono text-slate-300">${(t.exit_price ?? 0).toFixed(0)}</td>
                    <td className={`px-3 py-2 text-right font-mono ${(t.pnl ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {fmtUsd(t.pnl)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-slate-500">{(t.duration_hours ?? 0).toFixed(1)}h</td>
                    <td className="px-3 py-2 text-right font-mono text-cyan-400">{t.fusion_score?.toFixed(3)}</td>
                    <td className="px-3 py-2 text-xs text-gray-400 max-w-[120px] truncate" title={t.exit_reason || ''}>
                      {t.exit_reason || '-'}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-slate-400">
                      {(t.threshold_at_entry ?? t.threshold) != null ? (t.threshold_at_entry ?? t.threshold)!.toFixed(3) : '-'}
                    </td>
                    <td className={`px-3 py-2 text-right font-mono ${
                      (t.threshold_margin ?? 0) > 0.05 ? 'text-emerald-400' :
                      (t.threshold_margin ?? 0) > 0 ? 'text-yellow-400' : 'text-red-400'
                    }`}>
                      {t.threshold_margin != null ? t.threshold_margin.toFixed(3) : '-'}
                    </td>
                  </tr>
                  {isExpanded && <TradeDetail key={`detail-${i}`} trade={t} />}
                </>
              );
            })}
          </tbody>
        </table>
        {filtered.length === 0 && (
          <div className="text-center text-slate-600 py-8">No trades match filters</div>
        )}
      </div>
    </GlassCard>
  );
}
