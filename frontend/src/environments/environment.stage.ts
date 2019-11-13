const API_DOMAIN = 'http://0.0.0.0';
const API_PORT = '9090';

export const environment = {
  production: false,

  // apiUrl: 'http://10.101.5.110:9090',
  apiUrl: 'http://0.0.0.0:9090',

  // TODO: these temporary fields will be taken from API
  videoStreamUrl: `${API_DOMAIN}:9091/stream.mjpg`,
};
