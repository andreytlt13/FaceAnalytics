import {environment} from '../../../environments/environment';

export class Camera {
  constructor(public id: string = '', public name = '', public url = '', public type = 'image') {}

  get videoStreamUrl() {
    if (this.url.includes('rtsp://')) {
      // return `${environment.apiUrl}/video_stream?camera_url=${this.url}`;
      return `${environment.apiUrl}/video_stream?camera_url=0`;
    } else {
      return this.url;
    }
  }

  static parse({ id, name, url, type }) {
    return new Camera(id.toString(), name, url, type);
  }

  static isValid(camera: Camera) {
    return camera.name && camera.url && camera.type;
  }

  toJSON() {
    return {
      id: this.id,
      name: this.name,
      url: this.url,
      type: this.type
    };
  }
}
