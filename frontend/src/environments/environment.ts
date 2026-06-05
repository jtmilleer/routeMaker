// LAYER: Configuration — Development Environment
// PURPOSE: API URL used by route-api.service.ts in development mode.
//          The proxy.conf.json forwards /api and /auth to localhost:8000,
//          so the URL is just '' (same origin). This hides the backend port entirely.
export const environment = {
  production: false,
  apiUrl: '',  // Empty string = same origin; proxy handles the /api → :8000 forwarding
};
