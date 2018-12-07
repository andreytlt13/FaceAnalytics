export class Camera {
  constructor(public id: string = '', public name = '', public url = '', public type = 'image') {}

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
