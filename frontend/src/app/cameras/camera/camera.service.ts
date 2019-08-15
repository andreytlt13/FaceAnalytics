import {Injectable} from '@angular/core';
import {Observable, of} from 'rxjs';
import {Camera} from './camera';
import {HttpClient, HttpParams} from '@angular/common/http';
import {map} from 'rxjs/operators';
import {environment} from '../../../environments/environment';

import CAMERAS from './mock-cameras';
const CAMERA_URL = environment.apiUrl + '/camera';

@Injectable({
  providedIn: 'root'
})
export class CameraService {
  private cameras: Camera[] = [];

  constructor(private http: HttpClient) {
  }

  load(): Observable<Camera[]> {
    const observable = CAMERA_URL ? this.http.get(CAMERA_URL) : of({rows: CAMERAS});

    return observable
      .pipe(
        map((cameras: any) => cameras
          .map((camera) => {
            camera.id = (cameras.reduce((memo, cmr) => +cmr.id > memo ? +cmr.id : memo, 0) + 1);
            const cam = Camera.parse(camera);
            this.cameras.push(cam);
            return cam;
          }))
      );
  }

  create(camera: Camera): Observable<Camera> {
    const id = this.cameras.reduce((memo, cmr) => cmr.id > memo ? cmr.id : memo, 0) + 1;
    const newCamera = new Camera(id, camera.camera_url, camera.name, camera.status, camera.url_stream);

    this.cameras.push(newCamera);

    const observable = CAMERA_URL ? this.http.post(CAMERA_URL, {
      camera_url: newCamera.camera_url,
      camera_name: newCamera.name,
    }).pipe(map(() => newCamera)) : of(Camera.parse(newCamera));

    return observable;
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

  delete(camera: Camera): Observable<{id: number}> {
    const id = +camera.id;
    const existing = this.cameras.find(cmr => cmr.id === id);

    if (!existing) {
      throw new Error('Camera hasn\'t been created yet');
    }

    this.cameras.splice(this.cameras.indexOf(existing), 1);

    const params = new HttpParams({fromObject: {
        camera_name: camera.name
      }});

    const observable = CAMERA_URL ? this.http.delete(CAMERA_URL, {
      params
    }).pipe(map(() => ({id: camera.id}))) : of({id: camera.id});

    return observable;
  }
}
