import { Injectable } from '@angular/core';
import {Observable, of} from 'rxjs';
import {Camera} from './camera';
import {HttpClient} from '@angular/common/http';
import {map} from 'rxjs/operators';


const CAMERA_URL = '';
const CAMERA_SAMPLES = [{
  id: 0,
  name: 'Camera 1 (mpeg-dash)',
  type: 'video',
  url: 'http://dash.akamaized.net/dash264/TestCasesUHD/2b/11/MultiRate.mpd'
}, {
  id: 1,
  name: 'Camera 2 (img)',
  type: 'image',
  url: 'http://88.53.197.250/axis-cgi/mjpg/video.cgi?resolution=320x240'
}];

@Injectable({
  providedIn: 'root'
})
export class CameraService {

  constructor(private http: HttpClient) { }

  load(): Observable<Camera[]> {
    const observable = CAMERA_URL ? this.http.get(CAMERA_URL) : of({rows: CAMERA_SAMPLES});

    return observable
      .pipe(
        map(({rows: cameras}: any) => cameras.map(camera => Camera.parse(camera)))
      );
  }
}
