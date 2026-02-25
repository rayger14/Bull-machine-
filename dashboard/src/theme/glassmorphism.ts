export const glass = {
  card: 'bg-white/[0.03] backdrop-blur-xl border border-white/[0.08] rounded-2xl transition-all duration-300',
  cardHover: 'hover:bg-white/[0.06] hover:border-white/[0.15] hover:shadow-[0_0_30px_rgba(6,182,212,0.05)]',
  cardAccent: 'bg-white/[0.03] backdrop-blur-xl border border-cyan-500/20 rounded-2xl shadow-[0_0_20px_rgba(6,182,212,0.08)]',
  sidebar: 'bg-white/[0.02] backdrop-blur-2xl border-r border-white/[0.06]',
  input: 'bg-white/[0.05] border border-white/[0.10] rounded-lg px-3 py-2 text-slate-100 placeholder-slate-500 focus:border-cyan-500/40 focus:outline-none focus:ring-1 focus:ring-cyan-500/20 transition-colors',
  badge: 'px-2.5 py-0.5 rounded-full text-xs font-medium border',
} as const;

export const glassCard = `${glass.card} ${glass.cardHover}`;
export const glassCardAccent = `${glass.cardAccent} ${glass.cardHover}`;
