import {environment} from '../../../environments/environment';

export class Client {
  constructor(
    public id: number,
    public name?: string,
    public description?: string,
    public rate?: number,
  ) {}

  get photo() {
    if (this.name) {
      return `${environment.apiUrl}/get_face_known?name=${this.name}`;
    }

    if (this.id) {
      return `${environment.apiUrl}/get_face?object_id=${this.id}`;
    }

    throw new Error('Both id and name are not set to the client');
  }

  static parse(json: any) {
    return new Client(json.id, json.name, json.description, json.rate);
  }
}
