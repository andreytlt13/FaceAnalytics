import {Injectable} from '@angular/core';
import {HttpClient, HttpParams} from '@angular/common/http';
import {forkJoin, Observable} from 'rxjs';
import {Client} from './client';
import {map, mergeMap, tap} from 'rxjs/operators';

import {environment} from '../../../environments/environment';
import {isNumber} from 'lodash-es';

const CLIENTS_PATH = '';


interface UnknownPersonResponse {
  object_id: number[];
  camera_url: string;
}

interface MatchedPersonResponse {
  camera_url: string;
  names: string[];
  object_id: number;
}

interface KnownPersonResponse {
  camera_url: string;
  description: string;
  name: string;
  stars: number;
}

type UpdatePersonResponse = KnownPersonResponse & {object_id: string, stars: string};

@Injectable({
  providedIn: 'root'
})
export class ClientsService {
  constructor(private readonly http: HttpClient) {
  }

  getUnknown(cameraUrl: string): Observable<Client[]> {
    const params = new HttpParams()
      .set('camera_url', cameraUrl);

    return this.http.get<UnknownPersonResponse>(`${environment.apiUrl}/camera/object_id`, {responseType: 'json', params})
      .pipe(
        map(response => response.object_id.map(id => Client.parse({id})))
      );
  }

  getMatched(cameraUrl: string, client: Client): Observable<Client[]> {
    const params = new HttpParams()
      .set('camera_url', cameraUrl)
      .set('object_id', client.id.toString());

    return this.http.get<MatchedPersonResponse>(`${environment.apiUrl}/camera/name`, {params})
      .pipe(
        map(response => response.names.map(name => Client.parse({name}))),
        mergeMap(clients => forkJoin(
          // tslint:disable-next-line:no-shadowed-variable
          clients.map(client => this.getKnown(cameraUrl, client.name)
            .pipe(
              tap(c => c.name = client.name)
            )
          )
        ))
      );
  }

  bind(cameraUrl: string, client1: Client, client2: Client): Observable<Client> {
    const body = new FormData();

    body.set('camera_url', cameraUrl);
    body.set('object_id', client1.id.toString());
    body.set('name', client2.name);
    body.set('stars', client2.stars.toString());
    body.set('description', client2.description);

    return this.http.put<UpdatePersonResponse>(`${environment.apiUrl}/camera/object_id`, body)
      .pipe(
        map(response => Client.parse({id: response.object_id, ...response}))
      );
  }

  getKnown(cameraUrl: string, name: string): Observable<Client> {
    const params = new HttpParams()
      .set('camera_url', cameraUrl)
      .set('name', name);

    return this.http.get<KnownPersonResponse>(`${environment.apiUrl}/camera/name/info`, {params})
      .pipe(
        map(response => Client.parse(response))
      );
  }

  update(cameraUrl: string, client: Client): Observable<Client> {
    if (!isNumber(client.id) || !client.name) {
      console.error(client);
      throw new Error('Can\'t update client without id and name set');
    }

    const body = new FormData();

    body.set('camera_url', cameraUrl);
    body.set('object_id', client.id.toString());
    body.set('name', client.name);
    body.set('stars', client.stars.toString());
    body.set('description', client.description);

    return this.http.put<UpdatePersonResponse>(`${environment.apiUrl}/camera/object_id`, body)
      .pipe(
        map(response => Client.parse({id: response.object_id, ...response}))
      );
  }
}
