export function fmt(n: number | string | null | undefined, decimals = 0): string {
  const v = Number(n ?? 0);
  if (isNaN(v)) return '--';
  return v.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function fmtUsd(n: number | string | null | undefined, decimals = 2): string {
  const v = Number(n ?? 0);
  if (isNaN(v)) return '--';
  return '$' + fmt(v, decimals);
}

export function fmtPct(n: number | string | null | undefined, decimals = 2): string {
  const v = Number(n ?? 0);
  if (isNaN(v)) return '--';
  return v.toFixed(decimals) + '%';
}

export function fmtSign(n: number | string | null | undefined, decimals = 2): string {
  const v = Number(n ?? 0);
  if (isNaN(v)) return '--';
  const prefix = v > 0 ? '+' : '';
  return prefix + v.toFixed(decimals);
}

export function timeSince(iso: string | null | undefined): string {
  if (!iso) return '--';
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 0) return 'just now';
  if (s < 60) return Math.floor(s) + 's ago';
  if (s < 3600) return Math.floor(s / 60) + 'm ago';
  return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm ago';
}

export function uptimeStr(iso: string | null | undefined): string {
  if (!iso) return '--';
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  return (d ? d + 'd ' : '') + h + 'h ' + m + 'm';
}

const TZ = 'America/Los_Angeles';

export function shortDate(iso: string | null | undefined): string {
  if (!iso) return '--';
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: TZ }) +
    ' ' + d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: TZ });
}

export function shortDatePST(iso: string | null | undefined): string {
  if (!iso) return '--';
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: TZ }) +
    ' ' + d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: TZ }) + ' PST';
}

export function chartDatePST(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: TZ });
}

export function regimeBadgeColor(regime: string | null | undefined): string {
  const r = (regime ?? '').toLowerCase();
  if (r === 'bull') return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
  if (r === 'bear' || r === 'crisis') return 'text-rose-400 bg-rose-500/10 border-rose-500/20';
  if (r === 'stagflation') return 'text-orange-400 bg-orange-500/10 border-orange-500/20';
  if (r === 'neutral') return 'text-amber-400 bg-amber-500/10 border-amber-500/20';
  return 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20';
}

export function corrColor(v: number | null | undefined): string {
  if (v == null) return '#475569';
  const a = Math.abs(v);
  if (a < 0.2) return '#6b7a8d';
  if (v < -0.5) return '#f87171';
  if (v < -0.2) return '#fb923c';
  if (v > 0.5) return '#34d399';
  if (v > 0.2) return '#4da6ff';
  return '#6b7a8d';
}

export function fearGreedLabel(v: number | null | undefined): string {
  if (v == null) return '--';
  if (v < 20) return 'Extreme Fear';
  if (v < 40) return 'Fear';
  if (v < 60) return 'Neutral';
  if (v < 80) return 'Greed';
  return 'Extreme Greed';
}
