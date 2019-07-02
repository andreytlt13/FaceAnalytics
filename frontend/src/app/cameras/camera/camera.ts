export class Camera {
  constructor(public id: number = null, public camera_url: string = '', public name: string = '', public status = '', public url_stream = '') {}

  get videoStreamUrl() {
    return this.url_stream;
  }

  static parse({ id, camera_url, name, status, url_stream }) {
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
