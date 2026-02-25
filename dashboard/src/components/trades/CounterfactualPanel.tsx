import { type TradeCounterfactual, type CounterfactualResult } from '../../api/types';

interface CounterfactualPanelProps {
  data: TradeCounterfactual;
}

export function CounterfactualPanel({ data }: CounterfactualPanelProps) {
  // Group scenarios by param_changed type
  const groups: Record<string, CounterfactualResult[]> = {};
  Object.values(data.scenarios).forEach(s => {
    const key = s.param_changed;
    if (!groups[key]) groups[key] = [];
    groups[key].push(s);
  });

  const paramLabels: Record<string, string> = {
    stop_loss: 'Stop Loss Sensitivity',
    take_profit: 'Take Profit Sensitivity',
    hold_time: 'Hold Time Sensitivity',
    scale_out: 'Scale-Out Strategy',
  };

  return (
    <div className="mt-3 pt-3 border-t border-gray-700/50">
      {/* Header with summary */}
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Counterfactual Analysis
        </h4>
        <div className="flex items-center gap-3">
          {data.was_optimal && (
            <span className="px-2 py-0.5 text-[10px] font-medium bg-emerald-500/20 text-emerald-400 rounded">
              OPTIMAL
            </span>
          )}
          {data.best_pnl_delta > 0 && (
            <span className="text-[10px] text-gray-500">
              Best alt: <span className="text-emerald-400">{data.best_scenario}</span>
              {' '}(+${data.best_pnl_delta.toFixed(0)})
            </span>
          )}
        </div>
      </div>

      {/* Scenario groups */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {Object.entries(groups).map(([param, scenarios]) => (
          <div key={param} className="bg-gray-900/50 rounded-lg p-2">
            <h5 className="text-[10px] font-semibold text-gray-500 uppercase mb-1.5">
              {paramLabels[param] || param}
            </h5>
            <div className="space-y-0.5">
              {scenarios.map(s => (
                <ScenarioRow key={s.scenario} scenario={s} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ScenarioRow({ scenario }: { scenario: CounterfactualResult }) {
  const isPositive = scenario.pnl_delta > 0;
  const isNeutral = Math.abs(scenario.pnl_delta) < 5;

  // Format the scenario label nicely
  const label = scenario.scenario
    .replace('sl_', 'SL ')
    .replace('tp_', 'TP ')
    .replace('hold_', 'Hold ')
    .replace('no_scale_out', 'No Scale-Out')
    .replace('x', '\u00d7');

  return (
    <div className="flex items-center justify-between text-[11px] py-0.5">
      <span className="text-gray-400 w-24 truncate" title={scenario.scenario}>
        {label}
      </span>
      <span className="text-gray-500 text-[10px]">
        {scenario.exit_reason === 'stop_loss' ? 'SL' :
         scenario.exit_reason === 'take_profit' ? 'TP' :
         scenario.exit_reason === 'time_exit' ? 'Time' : 'EOD'}
      </span>
      <span className={`font-mono w-16 text-right ${
        isNeutral ? 'text-gray-500' :
        isPositive ? 'text-emerald-400' : 'text-red-400'
      }`}>
        {scenario.pnl_delta >= 0 ? '+' : ''}${scenario.pnl_delta.toFixed(0)}
      </span>
      <div className="w-16 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${
            isNeutral ? 'bg-gray-600' :
            isPositive ? 'bg-emerald-500' : 'bg-red-500'
          }`}
          style={{
            width: `${Math.min(100, Math.abs(scenario.pnl_delta) / 2)}%`,
            marginLeft: isPositive ? '50%' : undefined,
            marginRight: !isPositive ? '50%' : undefined,
          }}
        />
      </div>
    </div>
  );
}
