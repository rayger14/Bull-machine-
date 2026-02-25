interface BadgeProps {
  children: React.ReactNode;
  variant?: 'green' | 'red' | 'yellow' | 'orange' | 'cyan' | 'violet' | 'blue' | 'neutral';
  className?: string;
}

const variantClasses: Record<string, string> = {
  green: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  red: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
  yellow: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  orange: 'text-orange-400 bg-orange-500/10 border-orange-500/20',
  cyan: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20',
  violet: 'text-violet-400 bg-violet-500/10 border-violet-500/20',
  blue: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
  neutral: 'text-slate-400 bg-slate-500/10 border-slate-500/20',
};

export default function Badge({ children, variant = 'neutral', className = '' }: BadgeProps) {
  return (
    <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium border ${variantClasses[variant]} ${className}`}>
      {children}
    </span>
  );
}
