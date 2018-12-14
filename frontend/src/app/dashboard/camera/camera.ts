export class Camera {
  constructor(public id: string = '', public name = '', public url = '', public type = 'image') {}

  get videoStreamUrl() {
    if (this.url.includes('rtsp://')) {
      return `http://10.101.1.18:9090/video_stream?camera_url=${this.url}`;
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
