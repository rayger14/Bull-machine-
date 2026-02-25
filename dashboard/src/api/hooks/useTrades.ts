import { useQuery } from '@tanstack/react-query';
import api from '../client';
import type { Trade } from '../types';

export function useTrades() {
  return useQuery({
    queryKey: ['trades'],
    queryFn: () => api.get('api/trades').json<Trade[]>(),
    refetchInterval: 30_000,
    staleTime: 25_000,
  });
}
