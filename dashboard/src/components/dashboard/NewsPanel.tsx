import { Newspaper, ExternalLink, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import type { NewsData, NewsItem } from '../../api/types';

interface NewsPanelProps {
  news?: NewsData;
}

function SentimentIcon({ sentiment }: { sentiment: string }) {
  if (sentiment === 'bullish') return <TrendingUp className="w-3 h-3 text-emerald-400" />;
  if (sentiment === 'bearish') return <TrendingDown className="w-3 h-3 text-rose-400" />;
  return <Minus className="w-3 h-3 text-slate-500" />;
}

function sentimentBadge(sentiment: string) {
  if (sentiment === 'bullish') return 'green' as const;
  if (sentiment === 'bearish') return 'red' as const;
  return 'neutral' as const;
}

function timeAgo(isoStr: string): string {
  try {
    const diff = Date.now() - new Date(isoStr).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    const days = Math.floor(hrs / 24);
    return `${days}d ago`;
  } catch {
    return '';
  }
}

function HeadlineRow({ item }: { item: NewsItem }) {
  return (
    <div className="flex items-start gap-2 py-1.5 border-b border-white/[0.03] last:border-0">
      <SentimentIcon sentiment={item.sentiment} />
      <div className="flex-1 min-w-0">
        <a
          href={item.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-slate-300 hover:text-cyan-400 transition-colors line-clamp-2"
        >
          {item.headline}
          <ExternalLink className="w-2.5 h-2.5 inline ml-1 opacity-40" />
        </a>
        <div className="flex items-center gap-2 mt-0.5">
          <span className="text-[10px] text-slate-600">{item.source}</span>
          <span className="text-[10px] text-slate-600">{timeAgo(item.published_at)}</span>
          <Badge variant={sentimentBadge(item.sentiment)}>
            {item.sentiment} {item.sentiment_score > 0 ? '+' : ''}{item.sentiment_score.toFixed(1)}
          </Badge>
        </div>
      </div>
    </div>
  );
}

export default function NewsPanel({ news }: NewsPanelProps) {
  if (!news || !news.headlines?.length) {
    return (
      <GlassCard>
        <div className="flex items-center gap-2 mb-2">
          <Newspaper className="w-4 h-4 text-cyan-400" />
          <span className="text-xs text-slate-500 uppercase tracking-wider">Crypto News</span>
        </div>
        <div className="text-center text-slate-600 py-4 text-xs">No news data available</div>
      </GlassCard>
    );
  }

  const sentiment = news.sentiment;

  return (
    <GlassCard>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Newspaper className="w-4 h-4 text-cyan-400" />
          <span className="text-xs text-slate-500 uppercase tracking-wider">Crypto News</span>
        </div>
        {sentiment && (
          <div className="flex items-center gap-2">
            <span className={`text-xs font-mono font-bold ${
              sentiment.avg_score > 0.1 ? 'text-emerald-400' :
              sentiment.avg_score < -0.1 ? 'text-rose-400' : 'text-slate-400'
            }`}>
              {sentiment.summary}
            </span>
            <span className="text-[10px] text-slate-600">
              avg: {sentiment.avg_score > 0 ? '+' : ''}{sentiment.avg_score.toFixed(2)}
            </span>
          </div>
        )}
      </div>
      <div className="max-h-[300px] overflow-y-auto">
        {news.headlines.slice(0, 10).map((item, i) => (
          <HeadlineRow key={`${item.published_at}-${i}`} item={item} />
        ))}
      </div>
    </GlassCard>
  );
}
