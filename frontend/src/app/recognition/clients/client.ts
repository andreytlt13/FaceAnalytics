import {environment} from '../../../environments/environment';
import {isNumber} from 'lodash-es';

export class Client {
  constructor(
    public name: string, // because id is not defined for known person
    public id?: number,
    public description?: string,
    public stars?: number,
  ) {}

  get photo() {
    if (isNumber(this.id) && this.id >= 0) {
      return `${environment.apiUrl}/camera/get_object_img?object_id=${this.id}`;
    }

    if (this.name) {
      return `${environment.apiUrl}/camera/get_name_img?name=${this.name}`;
    }

    throw new Error('Both id and name are not set to the client');
  }

  static parse(json: {
    id?: string | number,
    name?: string,
    description?: string,
    stars?: string | number
  }) {
    return new Client(
      json.name || '',
      isFinite(+json.id) ? +json.id : undefined,
      json.description,
      isFinite(+json.stars) ? +json.stars : undefined
    );
  }
}
