import {environment} from '../../../environments/environment';
import {isNumber} from 'lodash-es';

export class Client {
  constructor(
    public cameraName: string,
    public name: string, // because id is not defined for known person
    public id?: number,
    public description?: string,
    public stars?: number,
    public isFaceDetected?: boolean,
  ) {}

  static parse(cameraName: string, json: {
    id?: string | number,
    name?: string,
    description?: string,
    stars?: string | number,
    face_detected?: boolean,
  }) {
    return new Client(
      cameraName,
      json.name || '',
      isFinite(+json.id) ? +json.id : undefined,
      json.description,
      isFinite(+json.stars) ? +json.stars : undefined,
      json.face_detected
    );
  }

  photo(by: string) {
    if (by === 'id' && isNumber(this.id) && this.id >= 0) {
      return `${environment.apiUrl}/camera/object/photo?object_id=${this.id}&camera_name=${this.cameraName}`;
    }

    if (by === 'name' && this.name) {
      return `${environment.apiUrl}/camera/name/photo?name=${this.name}&camera_name=${this.cameraName}`;
    }

    throw new Error('Both id and name are not set to the client');
  }
}


// Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
