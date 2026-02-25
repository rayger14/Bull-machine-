import { useQuery } from '@tanstack/react-query';
import api from '../client';
import type { StatusResponse } from '../types';

export function useStatus() {
  return useQuery({
    queryKey: ['status'],
    queryFn: () => api.get('api/status').json<StatusResponse>(),
    refetchInterval: 10_000,
    staleTime: 8_000,
  });
}
