export class Camera {
  constructor(private id: string, public name: string, public url: string, public type = 'video') {}

  static parse({ id, name, url, type }) {
    return new Camera(id, name, url, type);
  }
}
