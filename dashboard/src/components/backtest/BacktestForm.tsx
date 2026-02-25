import { useState } from 'react';
import { Play } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import type { BacktestParams } from '../../api/types';

interface BacktestFormProps {
  onSubmit: (params: BacktestParams) => void;
  isRunning: boolean;
}

export default function BacktestForm({ onSubmit, isRunning }: BacktestFormProps) {
  const [form, setForm] = useState<BacktestParams>({
    capital: 10000,
    leverage: 1.5,
    commission: '0.0002',
    slippage: 3,
    start_date: '2023-01-01',
    end_date: '2024-12-31',
  });

  const inputClass = 'bg-white/[0.05] border border-white/[0.10] rounded-lg px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:border-cyan-500/40 focus:outline-none focus:ring-1 focus:ring-cyan-500/20 transition-colors w-full';

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(form);
  };

  return (
    <GlassCard>
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-4">Backtest Configuration</div>
      <form onSubmit={handleSubmit} className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div>
          <label className="text-[10px] text-slate-600 uppercase block mb-1">Capital ($)</label>
          <input
            type="number"
            className={inputClass}
            value={form.capital}
            onChange={(e) => setForm({ ...form, capital: Number(e.target.value) })}
          />
        </div>
        <div>
          <label className="text-[10px] text-slate-600 uppercase block mb-1">Leverage</label>
          <input
            type="number"
            step="0.5"
            className={inputClass}
            value={form.leverage}
            onChange={(e) => setForm({ ...form, leverage: Number(e.target.value) })}
          />
        </div>
        <div>
          <label className="text-[10px] text-slate-600 uppercase block mb-1">Commission</label>
          <input
            type="text"
            className={inputClass}
            value={form.commission}
            onChange={(e) => setForm({ ...form, commission: e.target.value })}
          />
        </div>
        <div>
          <label className="text-[10px] text-slate-600 uppercase block mb-1">Slippage (bps)</label>
          <input
            type="number"
            className={inputClass}
            value={form.slippage}
            onChange={(e) => setForm({ ...form, slippage: Number(e.target.value) })}
          />
        </div>
        <div>
          <label className="text-[10px] text-slate-600 uppercase block mb-1">Start Date</label>
          <input
            type="date"
            className={inputClass}
            value={form.start_date}
            onChange={(e) => setForm({ ...form, start_date: e.target.value })}
          />
        </div>
        <div>
          <label className="text-[10px] text-slate-600 uppercase block mb-1">End Date</label>
          <input
            type="date"
            className={inputClass}
            value={form.end_date}
            onChange={(e) => setForm({ ...form, end_date: e.target.value })}
          />
        </div>
        <div className="col-span-full">
          <button
            type="submit"
            disabled={isRunning}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
              isRunning
                ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-cyan-500 to-emerald-500 text-white hover:shadow-[0_0_20px_rgba(6,182,212,0.3)]'
            }`}
          >
            {isRunning ? (
              <>
                <div className="w-4 h-4 border-2 border-slate-400/30 border-t-slate-400 rounded-full animate-spin" />
                Running...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run Backtest
              </>
            )}
          </button>
        </div>
      </form>
    </GlassCard>
  );
}
