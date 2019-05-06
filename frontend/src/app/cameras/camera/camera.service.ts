import {Injectable} from '@angular/core';
import {Observable, of} from 'rxjs';
import {Camera} from './camera';
import {HttpClient} from '@angular/common/http';
import {map} from 'rxjs/operators';

import CAMERAS from './mock-cameras';
const CAMERA_URL = '';

@Injectable({
  providedIn: 'root'
})
export class CameraService {

  constructor(private http: HttpClient) {
  }

  load(): Observable<Camera[]> {
    const observable = CAMERA_URL ? this.http.get(CAMERA_URL) : of({rows: CAMERAS});

    return observable
      .pipe(
        map(({rows: cameras}: any) => cameras.map(camera => Camera.parse(camera)))
      );
  }

  create(camera: Camera): Observable<Camera> {
    const id = CAMERAS.reduce((memo, cmr) => cmr.id > memo ? cmr.id : memo, 0) + 1;
    const newCamera = {
      ...camera.toJSON(),
      id
    };

    CAMERAS.push(newCamera);

    return of(Camera.parse(newCamera));
  }

  // update(camera: Camera): Observable<Camera> {
  //   const id = +camera.id;
  //   let existing = CAMERAS.find(cmr => cmr.id === id);
  //
  //   if (!existing) {
  //     throw new Error('Camera hasn\'t been created yet');
  //   }
  //
  //   existing = {
  //     ...camera.toJSON(),
  //     id
  //   };
  //
  //   return of(Camera.parse(existing));
  // }

  delete(camera: Camera): Observable<{id: string}> {
    const id = +camera.id;
    const existing = CAMERAS.find(cmr => cmr.id === id);

    if (!existing) {
      throw new Error('Camera hasn\'t been created yet');
    }

    CAMERAS.splice(CAMERAS.indexOf(existing), 1);

    return of({id: camera.id});
  }
}