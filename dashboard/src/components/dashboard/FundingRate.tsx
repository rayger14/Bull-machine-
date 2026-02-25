import GlassCard from '../ui/GlassCard';
import type { FundingCardData } from '../../api/types';

interface FundingRateProps {
  funding: FundingCardData | null | undefined;
}

export default function FundingRate({ funding }: FundingRateProps) {
  if (!funding) return null;

  return (
    <GlassCard>
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">Funding Rate</div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="text-[10px] text-slate-600">Rate (bps)</div>
          <div className={`text-lg font-bold font-mono ${(funding.last_rate_bps ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {funding.last_rate_bps?.toFixed(2) ?? '--'}
          </div>
        </div>
        <div>
          <div className="text-[10px] text-slate-600">Annualized</div>
          <div className="text-lg font-bold font-mono text-slate-300">
            {funding.annualized_pct != null ? `${funding.annualized_pct.toFixed(1)}%` : '--'}
          </div>
        </div>
        <div>
          <div className="text-[10px] text-slate-600">Total Cost</div>
          <div className="text-sm font-mono text-slate-400">
            ${funding.total_cost_usd?.toFixed(2) ?? '--'}
          </div>
        </div>
        <div>
          <div className="text-[10px] text-slate-600">Funding Z</div>
          <div className={`text-sm font-mono ${Math.abs(funding.funding_z ?? 0) > 2 ? 'text-rose-400' : 'text-slate-400'}`}>
            {funding.funding_z?.toFixed(2) ?? '--'}
          </div>
        </div>
      </div>
    </GlassCard>
  );
}
