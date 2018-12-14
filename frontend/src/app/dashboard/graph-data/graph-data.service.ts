import {Injectable} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {tap} from 'rxjs/operators';
import {Graph} from './graph';
import {Observable} from 'rxjs';

const DATA_URL = 'http://10.101.1.18:9090/db_select';

@Injectable({
  providedIn: 'root'
})
export class GraphDataService {

  constructor(private http: HttpClient) {
  }

  load(cameraUrl): any {
    let url = DATA_URL;
    const queryParams = {
      start_date: '2018-12-10 10:14:32',
      end_date: '2018-12-13 21:20:32'
    };
    const queryParamsString = Object.keys(queryParams).map(key => `${key}=${queryParams[key]}`).join('&');

    url += `?table=${cameraUrl}&${queryParamsString}`;

    return this.http.get(url)
      .pipe(
        tap(data => {
          console.log('=======', data);
          return data;
          // return Graph.parse(graphName, data);
        })
      );
  }
}
