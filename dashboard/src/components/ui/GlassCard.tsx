import type { ReactNode } from 'react';

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  accent?: boolean;
  onClick?: () => void;
  padding?: string;
}

export default function GlassCard({ children, className = '', accent = false, onClick, padding = 'p-5' }: GlassCardProps) {
  const base = accent
    ? 'bg-white/[0.03] backdrop-blur-xl border border-cyan-500/20 rounded-2xl shadow-[0_0_20px_rgba(6,182,212,0.08)]'
    : 'bg-white/[0.03] backdrop-blur-xl border border-white/[0.08] rounded-2xl';
  const hover = 'hover:bg-white/[0.06] hover:border-white/[0.15] hover:shadow-[0_0_30px_rgba(6,182,212,0.05)]';
  const transition = 'transition-all duration-300';

  return (
    <div
      className={`${base} ${hover} ${transition} ${padding} ${className}`}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      {children}
    </div>
  );
}
