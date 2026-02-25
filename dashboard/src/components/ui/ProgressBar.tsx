interface Segment {
  label: string;
  value: number;
  color: string;
}

interface ProgressBarProps {
  segments: Segment[];
  height?: string;
  showLabels?: boolean;
}

export default function ProgressBar({ segments, height = 'h-3', showLabels = true }: ProgressBarProps) {
  const total = segments.reduce((s, seg) => s + seg.value, 0);
  if (total === 0) return null;

  return (
    <div>
      <div className={`${height} rounded-full overflow-hidden flex bg-slate-800`}>
        {segments.map((seg, i) => (
          <div
            key={i}
            className="transition-all duration-500"
            style={{
              width: `${(seg.value / total) * 100}%`,
              backgroundColor: seg.color,
            }}
          />
        ))}
      </div>
      {showLabels && (
        <div className="flex gap-3 mt-2 flex-wrap">
          {segments.map((seg, i) => (
            <div key={i} className="flex items-center gap-1.5 text-[10px] text-slate-400">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: seg.color }} />
              {seg.label}: {((seg.value / total) * 100).toFixed(0)}%
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
