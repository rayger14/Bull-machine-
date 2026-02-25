import { motion } from 'framer-motion';

interface CMIDetailPanelProps {
  title: string;
  value: number;
  weights?: Record<string, number>;
  components?: Record<string, number>;
  rawFeatures?: Record<string, number>;
}

export default function CMIDetailPanel({ title, value, weights, components, rawFeatures }: CMIDetailPanelProps) {
  return (
    <motion.div
      initial={{ height: 0, opacity: 0 }}
      animate={{ height: 'auto', opacity: 1 }}
      exit={{ height: 0, opacity: 0 }}
      transition={{ duration: 0.2 }}
      className="mt-3 bg-white/[0.02] rounded-xl border border-white/[0.05] p-3 text-xs"
    >
      <div className="text-slate-400 font-medium mb-2">{title}: {value.toFixed(3)}</div>

      {/* Component values with weights */}
      {components && weights && Object.keys(components).length > 0 && (
        <div className="mb-2">
          <div className="text-slate-600 text-[10px] uppercase mb-1">Components</div>
          <div className="space-y-1">
            {Object.entries(components).map(([k, compVal]) => {
              const w = weights[k] ?? 0;
              return (
                <div key={k} className="flex items-center gap-2">
                  <span className="text-slate-500 w-32 truncate">{k.replace(/_/g, ' ')}</span>
                  <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-cyan-500/60 rounded-full"
                      style={{ width: `${Math.min(compVal * 100, 100)}%` }}
                    />
                  </div>
                  <span className="text-slate-300 w-12 text-right font-mono">{compVal.toFixed(2)}</span>
                  <span className="text-slate-600 w-10 text-right font-mono text-[10px]">{(w * 100).toFixed(0)}%w</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Fallback: weights only (if no components) */}
      {!components && weights && Object.keys(weights).length > 0 && (
        <div className="mb-2">
          <div className="text-slate-600 text-[10px] uppercase mb-1">Weights</div>
          <div className="space-y-1">
            {Object.entries(weights).map(([k, w]) => (
              <div key={k} className="flex items-center gap-2">
                <span className="text-slate-500 w-28 truncate">{k.replace(/_/g, ' ')}</span>
                <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-cyan-500/60 rounded-full"
                    style={{ width: `${Math.min(w * 100, 100)}%` }}
                  />
                </div>
                <span className="text-slate-400 w-8 text-right font-mono">{(w * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {rawFeatures && Object.keys(rawFeatures).length > 0 && (
        <div>
          <div className="text-slate-600 text-[10px] uppercase mb-1">Raw Features</div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
            {Object.entries(rawFeatures).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-slate-500 truncate">{k.replace(/_/g, ' ')}</span>
                <span className="text-slate-300 font-mono ml-2">
                  {typeof v === 'number' ? v.toFixed(3) : String(v)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}
