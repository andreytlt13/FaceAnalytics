import {Injectable} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {Observable} from 'rxjs';

import {environment} from '../../../environments/environment';

const DATA_URL = `${environment.apiUrl}/camera/db_select`;

@Injectable({
  providedIn: 'root'
})
export class EventDataService {

  constructor(private http: HttpClient) {
  }

  load(cameraName, startDate, endDate): Observable<Object> {
    let url = DATA_URL;
    const queryParams = {
      start_date: startDate,
      end_date: endDate
    };
    const queryParamsString = Object.keys(queryParams).map(key => `${key}=${queryParams[key]}`).join('&');

    url += `?${queryParamsString}&camera_name=${cameraName}`;

    return this.http.get(url);
  }

  // loadHeatmap(cameraUrl): Observable<Heatmap> {
  //   return this.load(cameraUrl)
  //     .pipe(
  //       map((data: CameraEvent[]) => {
  //         return Heatmap.parse(data);
  //       })
  //     );
  // }
  //
  // loadGraph(cameraUrl): Observable<Graph> {
  //   return this.load(cameraUrl)
  //     .pipe(
  //       map((data: CameraEvent[]) => {
  //         return Graph.parse('Unique Objects', data);
  //       })
  //     );
  // }
}

export interface CameraEvent {
  id: number;
  object_id: number;
  event_time: string;
  centroid_x: number;
  centroid_y: number;
}
