import GlassCard from './GlassCard';

interface StatCardProps {
  label: string;
  value: string | number;
  sub?: string;
  accent?: boolean;
  valueColor?: string;
  icon?: React.ReactNode;
}

export default function StatCard({ label, value, sub, accent = false, valueColor = 'text-slate-100', icon }: StatCardProps) {
  return (
    <GlassCard accent={accent} padding="p-4">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-slate-500 uppercase tracking-wider">{label}</span>
        {icon && <span className="text-slate-500">{icon}</span>}
      </div>
      <div className={`text-2xl font-bold ${valueColor} tracking-tight`}>
        {value}
      </div>
      {sub && <div className="text-xs text-slate-500 mt-1">{sub}</div>}
    </GlassCard>
  );
}
