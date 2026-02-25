import { useBacktest } from '../api/hooks/useBacktest';
import BacktestForm from '../components/backtest/BacktestForm';
import BacktestResults from '../components/backtest/BacktestResults';
import GlassCard from '../components/ui/GlassCard';

export default function BacktestPage() {
  const { submit, isSubmitting, job, error, reset } = useBacktest();

  const isRunning = isSubmitting || (job?.status === 'running') || (job?.status === 'queued');

  return (
    <div className="space-y-4">
      <BacktestForm onSubmit={submit} isRunning={isRunning} />

      {/* Status */}
      {isRunning && (
        <GlassCard>
          <div className="flex items-center gap-3">
            <div className="w-5 h-5 border-2 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin" />
            <span className="text-sm text-slate-300">
              {job?.status === 'queued' ? 'Queued...' : 'Running backtest (may take 1-5 minutes)...'}
            </span>
          </div>
        </GlassCard>
      )}

      {/* Error */}
      {error && (
        <GlassCard>
          <div className="text-sm text-rose-400">{error}</div>
          <button onClick={reset} className="text-xs text-cyan-400 hover:underline mt-2">Reset</button>
        </GlassCard>
      )}

      {/* Results */}
      {job?.status === 'complete' && job.result && (
        <BacktestResults result={job.result} />
      )}
    </div>
  );
}
