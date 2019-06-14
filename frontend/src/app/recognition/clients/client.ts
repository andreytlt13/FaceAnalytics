import {environment} from '../../../environments/environment';

export class Client {
  constructor(
    public name: string, // because id is not defined for known person
    public id?: number,
    public description?: string,
    public rate?: number,
  ) {}

  get photo() {
    if (this.name) {
      return `${environment.apiUrl}/camera/get_name_img?name=${this.name}`;
    }

    if (this.id) {
      return `${environment.apiUrl}/camera/get_object_img?object_id=${this.id}`;
    }

    throw new Error('Both id and name are not set to the client');
  }

  static parse(json: {
    id?: number,
    name?: string,
    description?: string,
    rate?: number,
  }) {
    return new Client(json.name || '', json.id, json.description, json.rate);
  }
}
