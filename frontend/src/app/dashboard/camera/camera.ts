export class Camera {
  constructor(public id: string = '', public name = '', private _url = '', public type = 'image') {}

  get url() {
    if (this._url.includes('rtsp://')) {
      return `http://10.101.1.18:9090/video_stream?camera_url=${this._url}`;
    } else {
      return this._url;
    }
  }

  set url(url) {
    this._url = url;
  }

  static parse({ id, name, url, type }) {
    return new Camera(id, name, url, type);
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
