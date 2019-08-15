const API_DOMAIN = 'http://0.0.0.0';
const API_PORT = '9090';

export const environment = {
  production: false,

  apiUrl: `${API_DOMAIN}:${API_PORT}`,

  // TODO: these temporary fields will be taken from API
  videoStreamUrl: `${API_DOMAIN}:9091/stream.mjpg`,
};
