import { useQuery } from '@tanstack/react-query';
import api from '../client';
import type { CandleRow } from '../types';

export function useCandleHistory() {
  return useQuery({
    queryKey: ['candle-history'],
    queryFn: () => api.get('api/candle-history').json<CandleRow[]>(),
    refetchInterval: 30_000,
    staleTime: 25_000,
  });
}
