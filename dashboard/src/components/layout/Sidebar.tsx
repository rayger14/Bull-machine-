import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  BookOpen,
  Radio,
  FlaskConical,
  Receipt,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { useAppStore } from '../../stores/useAppStore';
import { useStatus } from '../../api/hooks/useStatus';
import StatusDot from '../ui/StatusDot';
import { fmt } from '../../utils/format';

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/strategy', icon: BookOpen, label: 'Strategy' },
  { to: '/signals', icon: Radio, label: 'Signals' },
  { to: '/backtest', icon: FlaskConical, label: 'Backtest' },
  { to: '/trades', icon: Receipt, label: 'Trades' },
];

export default function Sidebar() {
  const expanded = useAppStore((s) => s.sidebarExpanded);
  const toggle = useAppStore((s) => s.toggleSidebar);
  const { data, isSuccess } = useStatus();
  const btcPrice = data?.heartbeat?.btc_price;

  return (
    <aside
      className={`fixed left-0 top-0 h-screen z-40 flex flex-col bg-white/[0.02] backdrop-blur-2xl border-r border-white/[0.06] transition-all duration-300 ${
        expanded ? 'w-60' : 'w-16'
      }`}
    >
      {/* Brand */}
      <div className={`flex items-center gap-3 px-4 h-16 border-b border-white/[0.06] ${expanded ? '' : 'justify-center'}`}>
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-emerald-500 flex items-center justify-center text-sm font-bold text-white shrink-0">
          BM
        </div>
        {expanded && (
          <span className="text-sm font-semibold text-slate-200 whitespace-nowrap">Bull Machine</span>
        )}
      </div>

      {/* Nav items */}
      <nav className="flex-1 py-4 space-y-1 px-2">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-xl transition-all duration-200 ${
                expanded ? 'px-3 py-2.5' : 'px-0 py-2.5 justify-center'
              } ${
                isActive
                  ? 'bg-cyan-500/[0.08] text-cyan-400 border-l-2 border-cyan-400 shadow-[inset_0_0_20px_rgba(6,182,212,0.05)]'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-white/[0.04] border-l-2 border-transparent'
              }`
            }
          >
            <item.icon className="w-5 h-5 shrink-0" />
            {expanded && <span className="text-sm font-medium">{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Bottom section */}
      <div className={`border-t border-white/[0.06] p-3 space-y-2 ${expanded ? '' : 'flex flex-col items-center'}`}>
        {btcPrice != null && (
          <div className={`${expanded ? '' : 'text-center'}`}>
            <div className="text-[10px] text-slate-600 uppercase">BTC</div>
            <div className="text-sm font-mono font-bold text-amber-400">
              ${fmt(btcPrice, 0)}
            </div>
          </div>
        )}
        <StatusDot online={isSuccess && !!data?.heartbeat?.timestamp} label={expanded ? (isSuccess ? 'Connected' : 'Offline') : undefined} />
        <button
          onClick={toggle}
          className="w-full flex items-center justify-center py-1.5 rounded-lg text-slate-600 hover:text-slate-400 hover:bg-white/[0.04] transition-colors"
        >
          {expanded ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </button>
      </div>
    </aside>
  );
}
