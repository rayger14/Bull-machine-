interface StatusDotProps {
  online: boolean;
  label?: string;
}

export default function StatusDot({ online, label }: StatusDotProps) {
  return (
    <div className="flex items-center gap-2">
      <div className={`w-2 h-2 rounded-full ${online ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]' : 'bg-rose-400 shadow-[0_0_8px_rgba(248,113,113,0.5)]'} animate-pulse`} />
      {label && <span className="text-xs text-slate-500">{label}</span>}
    </div>
  );
}
