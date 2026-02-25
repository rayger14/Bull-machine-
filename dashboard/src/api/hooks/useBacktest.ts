import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import api from '../client';
import type { BacktestParams, BacktestJob } from '../types';

export function useBacktest() {
  const [jobId, setJobId] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: async (params: BacktestParams) => {
      const data = await api.post('api/run-backtest', { json: params }).json<{ job_id: string; status: string }>();
      return data;
    },
    onSuccess: (data) => setJobId(data.job_id),
  });

  const status = useQuery({
    queryKey: ['backtest-status', jobId],
    queryFn: () => api.get(`api/backtest-status/${jobId}`).json<BacktestJob>(),
    refetchInterval: 3_000,
    enabled: !!jobId && mutation.isSuccess,
  });

  const reset = () => {
    setJobId(null);
    mutation.reset();
  };

  return {
    submit: mutation.mutate,
    isSubmitting: mutation.isPending,
    job: status.data,
    isPolling: status.isFetching,
    error: mutation.error?.message || status.data?.error,
    reset,
  };
}
