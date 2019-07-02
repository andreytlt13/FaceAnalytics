import {environment} from '../../../environments/environment';

export class Camera {
  constructor(
    public id: number = null,
    public camera_url: string = '',
    public name: string = '',
    public status: string = '',
    public url_stream: string = '') {
  }

  get videoStreamUrl() {
    if (this.camera_url.includes('rtsp://')) {
      // return `${environment.apiUrl}/video_stream?camera_url=${this.url}`;
      return `${environment.apiUrl}/video_stream?camera_url=0`;
    } else {
      return this.camera_url;
    }
  }

  static parse({id, camera_url, name, status, url_stream}) {
    return new Camera(id, camera_url, name, status, url_stream);
  }

  static isValid(camera: Camera) {
    return camera.camera_url && camera.name;
  }

  toJSON() {
    return {
      id: this.id,
      camera_url: this.camera_url,
      name: this.name,
      status: this.status,
      url_stream: this.url_stream
    };
  }
}
