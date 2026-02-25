import ky from 'ky';

const api = ky.create({
  prefixUrl: import.meta.env.VITE_API_URL || '',
  timeout: 30_000,
  retry: { limit: 2, methods: ['get'], delay: () => 1000 },
});

export default api;
