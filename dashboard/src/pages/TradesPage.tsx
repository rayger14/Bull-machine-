import { useTrades } from '../api/hooks/useTrades';
import { useStatus } from '../api/hooks/useStatus';
import TradeStats from '../components/trades/TradeStats';
import TradesByArchetype from '../components/trades/TradesByArchetype';
import FactorAttribution from '../components/trades/FactorAttribution';
import TradeList from '../components/trades/TradeList';

export default function TradesPage() {
  const { data: trades } = useTrades();
  const { data: status } = useStatus();

  const factorSummary = status?.heartbeat?.factor_attribution_summary;

  return (
    <div className="space-y-4">
      <TradeStats trades={trades ?? []} />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <TradesByArchetype trades={trades ?? []} />
        <FactorAttribution summary={factorSummary} />
      </div>
      <TradeList trades={trades ?? []} />
    </div>
  );
}
