import { useQuery } from '@tanstack/react-query';
import api from '../client';
import type { EquityRow } from '../types';

export function useEquityHistory() {
  return useQuery({
    queryKey: ['equity-history'],
    queryFn: () => api.get('api/equity-history').json<EquityRow[]>(),
    refetchInterval: 30_000,
    staleTime: 25_000,
  });
}
