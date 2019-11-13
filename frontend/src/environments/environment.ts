// This file can be replaced during build by using the `fileReplacements` array.
// `ng build --prod` replaces `environment.ts` with `environment.prod.ts`.
// The list of file replacements can be found in `angular.json`.

const API_DOMAIN = 'http://10.101.1.23';
const API_PORT = '9090';

export const environment = {
  production: false,

  apiUrl: `${API_DOMAIN}:${API_PORT}`,

  // TODO: these temporary fields will be taken from API
  videoStreamUrl: `${API_DOMAIN}:9091/stream.mjpg`,
};

/*
 * For easier debugging in development mode, you can import the following file
 * to ignore zone related error stack frames such as `zone.run`, `zoneDelegate.invokeTask`.
 *
 * This import should be commented out in production mode because it will have a negative impact
 * on performance if an error is thrown.
 */
import 'zone.js/dist/zone-error'; // Included with Angular CLI.
