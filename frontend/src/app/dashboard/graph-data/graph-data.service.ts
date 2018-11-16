import {Injectable} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {map} from 'rxjs/operators';
import {Graph} from './graph';
import {forkJoin, Observable} from 'rxjs';

const dataURLs = [];

@Injectable({
  providedIn: 'root'
})
export class GraphDataService {

  constructor(private http: HttpClient) {
  }

  loadAll(): Observable<Array<Graph>> {
    return forkJoin(
        dataURLs.map(({name}) => this.load(name))
      ).pipe(
        map(graphs => [...graphs])
      );
  }

  load(graphName): Observable<Graph> {
    const {url} = dataURLs.find(({name}) => graphName === name);

    return this.http.get(url)
      .pipe(
        map(data => {
          return Graph.parse(graphName, data);
        })
      );
  }
}
