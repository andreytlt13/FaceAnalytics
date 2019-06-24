const API_DOMAIN = 'http://0.0.0.0';
const API_PORT = '9090';

export const environment = {
  production: false,

  apiUrl: `${API_DOMAIN}:${API_PORT}`,

  // TODO: these temporary fields will be taken from API
  cameraUrl: 'rtsp://admin:0ZKaxVFi@10.101.106.4:554/live/main',
  videoStreamUrl: `${API_DOMAIN}:9091/stream.mjpg`,
};
