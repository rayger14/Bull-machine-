import { useState, useMemo, Fragment } from 'react';
import { ChevronRight, ChevronDown } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import FilterPills from '../ui/FilterPills';
import { CounterfactualPanel } from './CounterfactualPanel';
import { fmtUsd, shortDate } from '../../utils/format';
import type { Trade } from '../../api/types';

interface TradeListProps {
  trades: Trade[];
}

/** A group of exit rows sharing the same position_id (or a single ungrouped trade). */
interface PositionGroup {
  positionId: string;
  archetype: string;
  direction: string;
  regime: string;
  entryPrice: number;
  entryTime: string;
  fusionScore: number | undefined;
  threshold: number | undefined;
  thresholdAtEntry: number | undefined;
  thresholdMargin: number | undefined;
  totalPnl: number;
  totalDurationHours: number;
  exits: Trade[];
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

const COL_COUNT = 12;

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
  const [expandedPositions, setExpandedPositions] = useState<Set<string>>(new Set());
  const [expandedDetail, setExpandedDetail] = useState<string | null>(null);

  const archetypes = useMemo(() => {
    const s = new Set(trades.map(t => t.archetype ?? 'unknown'));
    return ['all', ...Array.from(s).sort()];
  }, [trades]);

  /** Group trades by position_id. Trades without position_id get a unique synthetic id. */
  const groups = useMemo(() => {
    const groupMap = new Map<string, Trade[]>();
    let syntheticIdx = 0;

    for (const t of trades) {
      const pid = t.position_id || `__solo_${syntheticIdx++}`;
      const existing = groupMap.get(pid);
      if (existing) {
        existing.push(t);
      } else {
        groupMap.set(pid, [t]);
      }
    }

    const result: PositionGroup[] = [];
    for (const [positionId, exits] of groupMap.entries()) {
      // Sort exits by exit time
      exits.sort((a, b) => (a.timestamp_exit ?? '').localeCompare(b.timestamp_exit ?? ''));

      const first = exits[0];
      const lastExit = exits[exits.length - 1];
      const totalPnl = exits.reduce((s, t) => s + (t.pnl ?? 0), 0);

      // Duration: from entry to the last exit
      let totalDurationHours = 0;
      if (first.timestamp_entry && lastExit.timestamp_exit) {
        const entryMs = new Date(first.timestamp_entry).getTime();
        const exitMs = new Date(lastExit.timestamp_exit).getTime();
        totalDurationHours = (exitMs - entryMs) / (1000 * 60 * 60);
      } else {
        // Fallback: sum of individual durations (for legacy data)
        totalDurationHours = exits.reduce((s, t) => Math.max(s, t.duration_hours ?? 0), 0);
      }

      result.push({
        positionId,
        archetype: first.archetype ?? 'unknown',
        direction: first.direction ?? 'long',
        regime: first.regime ?? '-',
        entryPrice: first.entry_price ?? 0,
        entryTime: first.timestamp_entry ?? '',
        fusionScore: first.fusion_score,
        threshold: first.threshold,
        thresholdAtEntry: first.threshold_at_entry,
        thresholdMargin: first.threshold_margin,
        totalPnl,
        totalDurationHours,
        exits,
      });
    }

    return result;
  }, [trades]);

  /** Filter and sort groups */
  const filtered = useMemo(() => {
    let g = [...groups];
    if (filter === 'winners') g = g.filter(x => x.totalPnl > 0);
    if (filter === 'losers') g = g.filter(x => x.totalPnl <= 0);
    if (archFilter !== 'all') g = g.filter(x => x.archetype === archFilter);
    if (sort === 'time_desc') g.sort((a, b) => b.entryTime.localeCompare(a.entryTime));
    else if (sort === 'time_asc') g.sort((a, b) => a.entryTime.localeCompare(b.entryTime));
    else if (sort === 'pnl_desc') g.sort((a, b) => b.totalPnl - a.totalPnl);
    else if (sort === 'pnl_asc') g.sort((a, b) => a.totalPnl - b.totalPnl);
    return g;
  }, [groups, filter, sort, archFilter]);

  const totalExitRows = useMemo(() => filtered.reduce((s, g) => s + g.exits.length, 0), [filtered]);

  const togglePosition = (pid: string) => {
    setExpandedPositions(prev => {
      const next = new Set(prev);
      if (next.has(pid)) {
        next.delete(pid);
      } else {
        next.add(pid);
      }
      return next;
    });
  };

  const toggleDetail = (key: string) => {
    setExpandedDetail(prev => prev === key ? null : key);
  };

  return (
    <GlassCard>
      <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
        <span className="text-xs text-slate-500 uppercase tracking-wider">
          Trade History ({filtered.length} positions, {totalExitRows} exits)
        </span>
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
              <th className="w-6 px-1 py-2"></th>
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
            {filtered.map((group) => {
              const isMultiExit = group.exits.length > 1;
              const isExpanded = expandedPositions.has(group.positionId);
              const detailKey = `detail-${group.positionId}`;
              const isDetailExpanded = expandedDetail === detailKey;
              const thresholdVal = group.thresholdAtEntry ?? group.threshold;

              return (
                <Fragment key={group.positionId}>
                  {/* Parent row: position summary */}
                  <tr
                    className={`border-b border-white/[0.03] hover:bg-white/[0.02] cursor-pointer ${isExpanded ? 'bg-white/[0.03]' : ''}`}
                    onClick={() => {
                      if (isMultiExit) {
                        togglePosition(group.positionId);
                      } else {
                        toggleDetail(detailKey);
                      }
                    }}
                  >
                    <td className="px-1 py-2 text-slate-500">
                      {isMultiExit ? (
                        isExpanded
                          ? <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
                          : <ChevronRight className="w-3.5 h-3.5 text-slate-500" />
                      ) : (
                        <span className="w-3.5 h-3.5 inline-block" />
                      )}
                    </td>
                    <td className="px-3 py-2 text-slate-400 font-medium">{shortDate(group.entryTime)}</td>
                    <td className="px-3 py-2">
                      <Badge variant="cyan">{group.archetype.replace(/_/g, ' ')}</Badge>
                      {isMultiExit && (
                        <span className="ml-1.5 text-[10px] text-slate-500 font-mono">
                          {group.exits.length} exits
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-2">
                      <span className={`text-xs font-medium ${
                        group.regime === 'bull' ? 'text-emerald-400' :
                        group.regime === 'bear' ? 'text-rose-400' :
                        'text-slate-400'
                      }`}>
                        {group.regime || '-'}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-slate-300 font-medium">
                      ${group.entryPrice.toFixed(0)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-slate-500">
                      {isMultiExit ? '-' : `$${(group.exits[0].exit_price ?? 0).toFixed(0)}`}
                    </td>
                    <td className={`px-3 py-2 text-right font-mono font-semibold ${group.totalPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {fmtUsd(group.totalPnl)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-slate-500">
                      {group.totalDurationHours.toFixed(1)}h
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-cyan-400">
                      {group.fusionScore?.toFixed(3)}
                    </td>
                    <td className="px-3 py-2 text-xs text-gray-400 max-w-[120px] truncate" title={
                      isMultiExit
                        ? group.exits.map(e => e.exit_reason).filter(Boolean).join(', ')
                        : group.exits[0].exit_reason || ''
                    }>
                      {isMultiExit
                        ? `${group.exits.length} scale-outs`
                        : (group.exits[0].exit_reason || '-')
                      }
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-slate-400">
                      {thresholdVal != null ? thresholdVal.toFixed(3) : '-'}
                    </td>
                    <td className={`px-3 py-2 text-right font-mono ${
                      (group.thresholdMargin ?? 0) > 0.05 ? 'text-emerald-400' :
                      (group.thresholdMargin ?? 0) > 0 ? 'text-yellow-400' : 'text-red-400'
                    }`}>
                      {group.thresholdMargin != null ? group.thresholdMargin.toFixed(3) : '-'}
                    </td>
                  </tr>

                  {/* For single-exit trades, show detail panel inline */}
                  {!isMultiExit && isDetailExpanded && (
                    <TradeDetail key={`single-detail-${group.positionId}`} trade={group.exits[0]} />
                  )}

                  {/* Child rows: individual exits (only for multi-exit positions) */}
                  {isMultiExit && isExpanded && group.exits.map((exit, exitIdx) => {
                    const exitKey = `${group.positionId}-exit-${exitIdx}`;
                    const isExitDetailExpanded = expandedDetail === exitKey;
                    return (
                      <Fragment key={exitKey}>
                        <tr
                          className="border-b border-white/[0.02] hover:bg-white/[0.015] cursor-pointer bg-gray-900/30"
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleDetail(exitKey);
                          }}
                        >
                          <td className="px-1 py-1.5"></td>
                          <td className="px-3 py-1.5 text-slate-500 pl-6">
                            <span className="text-slate-600 mr-1">|_</span>
                            {shortDate(exit.timestamp_exit)}
                          </td>
                          <td className="px-3 py-1.5 text-slate-500 text-[10px] font-mono">
                            exit {exitIdx + 1}/{group.exits.length}
                          </td>
                          <td className="px-3 py-1.5"></td>
                          <td className="px-3 py-1.5 text-right font-mono text-slate-500">
                            ${group.entryPrice.toFixed(0)}
                          </td>
                          <td className="px-3 py-1.5 text-right font-mono text-slate-400">
                            ${(exit.exit_price ?? 0).toFixed(0)}
                          </td>
                          <td className={`px-3 py-1.5 text-right font-mono ${(exit.pnl ?? 0) >= 0 ? 'text-emerald-400/70' : 'text-rose-400/70'}`}>
                            {fmtUsd(exit.pnl)}
                          </td>
                          <td className="px-3 py-1.5 text-right font-mono text-slate-600">
                            {(exit.duration_hours ?? 0).toFixed(1)}h
                          </td>
                          <td className="px-3 py-1.5"></td>
                          <td className="px-3 py-1.5 text-[11px] text-slate-500 max-w-[120px] truncate" title={exit.exit_reason || ''}>
                            {exit.exit_reason || '-'}
                          </td>
                          <td className="px-3 py-1.5"></td>
                          <td className="px-3 py-1.5"></td>
                        </tr>
                        {isExitDetailExpanded && (
                          <TradeDetail key={`child-detail-${exitKey}`} trade={exit} />
                        )}
                      </Fragment>
                    );
                  })}
                </Fragment>
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
