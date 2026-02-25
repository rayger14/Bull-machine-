import GlassCard from '../ui/GlassCard';
import { uptimeStr, fmtPct, fmtUsd } from '../../utils/format';
import type { Heartbeat, Performance } from '../../api/types';

interface SessionInfoProps {
  hb: Heartbeat;
  performance?: Performance | null;
}

export default function SessionInfo({ hb, performance }: SessionInfoProps) {
  const maxDD = performance?.max_drawdown_pct as number | undefined;
  const netPnl = performance?.net_pnl_after_funding as number | undefined;

  return (
    <GlassCard>
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">Session Info</div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
        <div>
          <div className="text-slate-600">Uptime</div>
          <div className="text-slate-300 font-mono">{uptimeStr(hb.session_start)}</div>
        </div>
        <div>
          <div className="text-slate-600">Bars</div>
          <div className="text-slate-300 font-mono">{hb.bars_processed?.toLocaleString() ?? '--'}</div>
        </div>
        <div>
          <div className="text-slate-600">Max DD</div>
          <div className="text-rose-400 font-mono">{maxDD != null ? fmtPct(maxDD) : '--'}</div>
        </div>
        <div>
          <div className="text-slate-600">Net PnL</div>
          <div className={`font-mono ${(netPnl ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {netPnl != null ? fmtUsd(netPnl) : '--'}
          </div>
        </div>
        <div>
          <div className="text-slate-600">Signals</div>
          <div className="text-slate-300 font-mono">{hb.total_signals ?? '--'}</div>
        </div>
        <div>
          <div className="text-slate-600">Allocated</div>
          <div className="text-slate-300 font-mono">{hb.signals_allocated ?? '--'}</div>
        </div>
        <div>
          <div className="text-slate-600">Positions</div>
          <div className="text-slate-300 font-mono">{hb.positions ?? 0}</div>
        </div>
        <div>
          <div className="text-slate-600">Completed Trades</div>
          <div className="text-slate-300 font-mono">{hb.completed_trades ?? 0}</div>
        </div>
      </div>
    </GlassCard>
  );
}
