import {Injectable} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {map} from 'rxjs/operators';
import {Graph} from './graph';
import {Observable} from 'rxjs';
import Heatmap from './heatmap';

const DATA_URL = 'http://10.101.1.18:9090/db_select';

@Injectable({
  providedIn: 'root'
})
export class EventDataService {

  constructor(private http: HttpClient) {
  }

  load(cameraUrl): Observable<Object> {
    let url = DATA_URL;
    const queryParams = {
      start_date: '2018-12-01 00:00:00',
      end_date: '2019-12-31 23:59:59'
    };
    const queryParamsString = Object.keys(queryParams).map(key => `${key}=${queryParams[key]}`).join('&');

    url += `?table=${cameraUrl}&${queryParamsString}`;

    return this.http.get(url);
  }

  loadHeatmap(cameraUrl): Observable<Heatmap> {
    return this.load(cameraUrl)
      .pipe(
        map((data: CameraEvent[]) => {
          return Heatmap.parse(data);
        })
      );
  }

  loadGraph(cameraUrl): Observable<Graph> {
    return this.load(cameraUrl)
      .pipe(
        map((data: CameraEvent[]) => {
          return Graph.parse('Unique Objects', data);
        })
      );
  }
}

export interface CameraEvent {
  id: number;
  object_id: number;
  event_time: string;
  enter: number;
  exit: number;
  x: number;
  y: number;
}
