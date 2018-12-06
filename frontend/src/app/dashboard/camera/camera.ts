export class Camera {
  constructor(public id: string, public name: string, public url: string, public type = 'video') {}

  static parse({ id, name, url, type }) {
    return new Camera(id, name, url, type);
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
