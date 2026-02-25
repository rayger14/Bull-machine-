import { createHashRouter } from 'react-router-dom';
import { lazy, Suspense } from 'react';
import App from './App';

const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const StrategyPage = lazy(() => import('./pages/StrategyPage'));
const SignalsPage = lazy(() => import('./pages/SignalsPage'));
const BacktestPage = lazy(() => import('./pages/BacktestPage'));
const TradesPage = lazy(() => import('./pages/TradesPage'));

function LazyWrapper({ children }: { children: React.ReactNode }) {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center h-64">
          <div className="w-6 h-6 border-2 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin" />
        </div>
      }
    >
      {children}
    </Suspense>
  );
}

export const router = createHashRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <LazyWrapper><DashboardPage /></LazyWrapper> },
      { path: 'strategy', element: <LazyWrapper><StrategyPage /></LazyWrapper> },
      { path: 'signals', element: <LazyWrapper><SignalsPage /></LazyWrapper> },
      { path: 'backtest', element: <LazyWrapper><BacktestPage /></LazyWrapper> },
      { path: 'trades', element: <LazyWrapper><TradesPage /></LazyWrapper> },
    ],
  },
]);
