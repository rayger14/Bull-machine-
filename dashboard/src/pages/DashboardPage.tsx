import { useStatus } from '../api/hooks/useStatus';
import { useEquityHistory } from '../api/hooks/useEquityHistory';
import OraclePanel from '../components/dashboard/OraclePanel';
import MarketBriefing from '../components/dashboard/MarketBriefing';
import MetricRow from '../components/dashboard/MetricRow';
import ThresholdHero from '../components/dashboard/ThresholdHero';
import OpenPositions from '../components/dashboard/OpenPositions';
import LastSignalNarrative from '../components/dashboard/LastSignalNarrative';
import MacroEnvironment from '../components/dashboard/MacroEnvironment';
import MacroOutlook from '../components/dashboard/MacroOutlook';
import FundingRate from '../components/dashboard/FundingRate';
import StressScenarios from '../components/dashboard/StressScenarios';
import CointegrationTable from '../components/dashboard/CointegrationTable';
import MacroCorrelations from '../components/dashboard/MacroCorrelations';
import CapitalFlows from '../components/dashboard/CapitalFlows';
import WyckoffCycle from '../components/dashboard/WyckoffCycle';
import SessionInfo from '../components/dashboard/SessionInfo';
import PhantomTracker from '../components/dashboard/PhantomTracker';
import WhaleIntelligencePanel from '../components/dashboard/WhaleIntelligencePanel';

import EquityCurve from '../components/charts/EquityCurve';
import CMIHistoryChart from '../components/charts/CMIHistoryChart';

export default function DashboardPage() {
  const { data: status } = useStatus();
  const { data: equityData } = useEquityHistory();

  const hb = status?.heartbeat ?? {};
  const perf = status?.performance ?? null;
  const oracle = hb.oracle;

  return (
    <div className="space-y-4">
      {/* 0. Oracle Panel — master synthesis (top of page) */}
      <OraclePanel oracle={oracle} />

      {/* 1. Market Briefing — synthesized intelligence */}
      <MarketBriefing hb={hb} oracle={oracle} />

      {/* 2. Metric Row */}
      <MetricRow hb={hb} performance={perf} />

      {/* 3. Threshold Hero */}
      <ThresholdHero hb={hb} oracle={oracle} />

      {/* 4. Open Positions (moved up) */}
      <OpenPositions positions={hb.open_position_details} />

      {/* 5. Last Signal Narrative + Funding Rate */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <LastSignalNarrative narrative={hb.last_signal_narrative} oracle={oracle} />
        </div>
        <FundingRate funding={hb.funding} />
      </div>

      {/* 5.5. Whale / Institutional Intelligence */}
      <WhaleIntelligencePanel whale={hb.whale_intelligence} oracle={oracle} />

      {/* 6. Equity Curve + CMI History */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <EquityCurve data={equityData} />
        <CMIHistoryChart data={equityData} />
      </div>

      {/* 6.5. Phantom Trade Tracker — fusion score analysis */}
      <PhantomTracker
        phantom={hb.phantom_tracker}
        threshold={hb.threshold}
        winRate={hb.win_rate}
      />

      {/* 7. Capital Flows + Wyckoff Cycle */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <CapitalFlows flows={hb.capital_flows} oracle={oracle} />
        <WyckoffCycle wyckoff={hb.wyckoff} oracle={oracle} />
      </div>

      {/* 8. Macro Environment */}
      <MacroEnvironment macro={hb.macro} oracle={oracle} />

      {/* 9. Macro Outlook */}
      <MacroOutlook outlook={hb.macro_outlook} oracle={oracle} />

      {/* 10. Stress Scenarios + Cointegration Table */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <StressScenarios scenarios={hb.active_stress_scenarios} oracle={oracle} />
        <CointegrationTable data={hb.cointegration} />
      </div>

      {/* 11. Macro Correlations */}
      <MacroCorrelations data={hb.macro_correlations} />

      {/* 12. Session Info */}
      <SessionInfo hb={hb} performance={perf} />
    </div>
  );
}
