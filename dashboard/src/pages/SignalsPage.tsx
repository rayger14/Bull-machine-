import { useSignalLog } from '../api/hooks/useSignalLog';
import { useCandleHistory } from '../api/hooks/useCandleHistory';
import BTCPriceChart from '../components/charts/BTCPriceChart';
import SignalFunnel from '../components/signals/SignalFunnel';
import SignalExplorer from '../components/signals/SignalExplorer';

export default function SignalsPage() {
  const { data: signals } = useSignalLog();
  const { data: candles } = useCandleHistory();

  return (
    <div className="space-y-4">
      <BTCPriceChart data={candles} />
      <SignalFunnel signals={signals ?? []} />
      <SignalExplorer signals={signals ?? []} />
    </div>
  );
}
