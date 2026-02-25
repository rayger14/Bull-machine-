import { useQuery } from '@tanstack/react-query';
import api from '../client';
import type { Signal } from '../types';

export function useSignalLog() {
  return useQuery({
    queryKey: ['signal-log'],
    queryFn: async () => {
      const data = await api.get('api/signal-log').json<Signal[]>();
      return data.reverse();
    },
    refetchInterval: 30_000,
    staleTime: 25_000,
  });
}
