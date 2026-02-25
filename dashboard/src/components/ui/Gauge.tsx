interface GaugeProps {
  value: number;
  label: string;
  description?: string;
  onClick?: () => void;
  colorStops?: string[];
}

const defaultColors = [
  'bg-emerald-500',
  'bg-emerald-400',
  'bg-amber-400',
  'bg-orange-400',
  'bg-rose-500',
];

export default function Gauge({ value, label, description, onClick, colorStops = defaultColors }: GaugeProps) {
  const pct = Math.min(Math.max(value * 100, 0), 100);

  return (
    <div
      className={`${onClick ? 'cursor-pointer' : ''}`}
      onClick={onClick}
    >
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs text-slate-400">{label}</span>
        <span className="text-sm font-mono font-bold text-slate-200">{value.toFixed(2)}</span>
      </div>
      <div className="relative h-2.5 rounded-full overflow-hidden bg-slate-800">
        {/* Color segments background */}
        <div className="absolute inset-0 flex">
          {colorStops.map((c, i) => (
            <div key={i} className={`flex-1 ${c} opacity-20`} />
          ))}
        </div>
        {/* Fill bar */}
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-500 bg-gradient-to-r from-cyan-400 to-emerald-400"
          style={{ width: `${pct}%` }}
        />
      </div>
      {description && (
        <div className="text-[10px] text-slate-600 mt-1">{description}</div>
      )}
    </div>
  );
}
