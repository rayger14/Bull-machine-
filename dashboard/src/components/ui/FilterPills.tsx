interface FilterPillsProps<T extends string> {
  options: { value: T; label: string }[];
  selected: T;
  onChange: (value: T) => void;
}

export default function FilterPills<T extends string>({ options, selected, onChange }: FilterPillsProps<T>) {
  return (
    <div className="flex gap-1.5 flex-wrap">
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={`px-3 py-1 rounded-lg text-xs font-medium transition-all duration-200 border ${
            selected === opt.value
              ? 'bg-cyan-500/15 border-cyan-500/30 text-cyan-400'
              : 'bg-white/[0.03] border-white/[0.08] text-slate-400 hover:bg-white/[0.06] hover:border-white/[0.12]'
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
