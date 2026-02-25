import { useEffect, useRef } from 'react';
import { createChart, ColorType, LineSeries, HistogramSeries, type IChartApi, type UTCTimestamp } from 'lightweight-charts';
import GlassCard from '../ui/GlassCard';
import type { CandleRow } from '../../api/types';

interface BTCPriceChartProps {
  data: CandleRow[] | undefined;
  height?: number;
}

export default function BTCPriceChart({ data, height = 300 }: BTCPriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || !data || data.length === 0) return;

    // Remove previous chart instance if it exists
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    // Deduplicate by timestamp — API can return duplicate timestamps
    // which causes lightweight-charts to silently fail
    const uniqueMap = new Map<string, CandleRow>();
    for (const c of data) {
      uniqueMap.set(c.timestamp, c);
    }
    const unique = Array.from(uniqueMap.values());

    if (unique.length === 0) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#6b7a8d',
        fontSize: 10,
      },
      grid: {
        vertLines: { color: 'rgba(30,39,56,0.5)' },
        horzLines: { color: 'rgba(30,39,56,0.5)' },
      },
      width: containerRef.current.clientWidth,
      height,
      rightPriceScale: {
        borderColor: 'rgba(30,39,56,0.5)',
      },
      timeScale: {
        borderColor: 'rgba(30,39,56,0.5)',
        timeVisible: true,
      },
      crosshair: {
        horzLine: { color: 'rgba(6,182,212,0.3)', labelBackgroundColor: '#0f172a' },
        vertLine: { color: 'rgba(6,182,212,0.3)', labelBackgroundColor: '#0f172a' },
      },
    });

    const lineData = unique.map((c) => ({
      time: (Math.floor(new Date(c.timestamp).getTime() / 1000)) as UTCTimestamp,
      value: parseFloat(c.close),
    }));

    const volData = unique.map((c) => ({
      time: (Math.floor(new Date(c.timestamp).getTime() / 1000)) as UTCTimestamp,
      value: parseFloat(c.volume),
      color: 'rgba(77,166,255,0.2)',
    }));

    const lineSeries = chart.addSeries(LineSeries, {
      color: '#fbbf24',
      lineWidth: 2,
      priceFormat: { type: 'price', precision: 0, minMove: 1 },
    });
    lineSeries.setData(lineData);

    const volumeSeries = chart.addSeries(HistogramSeries, {
      color: 'rgba(77,166,255,0.2)',
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });
    volumeSeries.setData(volData);

    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });

    chart.timeScale().fitContent();
    chartRef.current = chart;

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [data, height]);

  return (
    <GlassCard>
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">BTC Price</div>
      <div ref={containerRef} />
      {(!data || data.length === 0) && (
        <div className="h-64 flex items-center justify-center text-slate-600 text-sm">No candle data</div>
      )}
    </GlassCard>
  );
}
