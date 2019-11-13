import {Injectable} from '@angular/core';
import {HttpClient, HttpParams} from '@angular/common/http';
import {forkJoin, from, Observable, of} from 'rxjs';
import {Client} from './client';
import {map, mergeAll, mergeMap, tap, toArray} from 'rxjs/operators';

import {environment} from '../../../environments/environment';
import {isNumber} from 'lodash-es';
import * as uuidv4 from 'uuid/v4';

interface UnknownPersonResponse {
  [id: string]: {
    face_detected: boolean;
    name: string | null;
    names: string[];
    probability: number[];
    age?: number;
    gender?: string;
  };
}

interface KnownPersonResponse {
  camera_name: string;
  description: string;
  name: string;
  stars: number;
}

interface UpdatePersonResponse {
  status: string;
}

@Injectable({
  providedIn: 'root'
})
export class ClientsService {
  private readonly nameInfoCache = new Map<string, Client>();

  constructor(private readonly http: HttpClient) {
  }

  getUnknown(cameraName: string): Observable<{ person: Client, matches: Client[] }[]> {
    const params = new HttpParams()
      .set('camera_name', cameraName);

    return this.http.get<UnknownPersonResponse>(`${environment.apiUrl}/camera/objects`, {responseType: 'json', params})
      .pipe(
        map(response => Object.keys(response).map(id => ({
          client: Client.parse(cameraName, {
            id,
            name: response[id].name ? response[id].name : undefined,
            face_detected: response[id].face_detected,
            age: response[id].age,
            gender: response[id].gender,
          }),
          names: response[id].names && response[id].names.length > 0 && response[id].names[0] !== null ? response[id].names : [],
        }))),
        map(data => from(data)
          .pipe(
            mergeMap(client => {
              let observable = forkJoin(
                // tslint:disable-next-line:no-shadowed-variable
                client.names.map(name => this.getKnown(cameraName, name))
              ).pipe(
                map(matches => ({person: client.client, matches}))
              );

              if (client.client.name) {
                observable = this.getKnown(cameraName, client.client.name).pipe(map(c => {
                  c.id = client.client.id;

                  return {
                    person: c,
                    matches: [c]
                  };
                }));
              }

              return observable;
            })
          )
        ),
        mergeAll(),
        toArray()
      );
  }

  bind(cameraName: string, client1: Client, client2: Client): Observable<Client> {
    const body = new FormData();

    body.set('camera_name', cameraName);
    body.set('object_id', client1.id.toString());
    body.set('name', client2.name);
    body.set('stars', client2.stars.toString());
    body.set('description', client2.description);

    return this.http.put<UpdatePersonResponse>(`${environment.apiUrl}/camera/object`, body)
      .pipe(
        map(() => Client.parse(cameraName, {
          id: client1.id,
          name: client2.name,
          stars: client2.stars,
          description: client2.description,
          face_detected: client1.isFaceDetected
        })),
        tap(c => this.nameInfoCache.set(c.name, c)),
      );
  }

  getKnown(cameraName: string, name: string): Observable<Client> {
    if (this.nameInfoCache.has(name)) {
      return of(this.nameInfoCache.get(name));
    }

    const params = new HttpParams()
      .set('camera_name', cameraName)
      .set('name', name);

    return this.http.get<KnownPersonResponse>(`${environment.apiUrl}/camera/name/info`, {params})
      .pipe(
        map(response => Client.parse(cameraName, response)),
        tap(c => this.nameInfoCache.set(c.name, c))
      );
  }

  update(cameraName: string, client: Client): Observable<Client> {
    if (!isNumber(client.id) || !client.name) {
      console.error(client);
      throw new Error('Can\'t update client without id and name set');
    }

    const body = new FormData();

    body.set('camera_name', cameraName);
    body.set('object_id', client.id.toString());
    body.set('name', client.name);
    body.set('stars', client.stars.toString());
    body.set('description', client.description);

    return this.http.put<UpdatePersonResponse>(`${environment.apiUrl}/camera/object`, body)
      .pipe(
        map(() => client),
        tap(c => this.nameInfoCache.set(client.name, client)),
      );
  }

  create(cameraName: string, client: Client): Observable<Client> {
    const body = new FormData();

    client.name = uuidv4();

    body.set('camera_name', cameraName);
    body.set('object_id', client.id.toString());
    body.set('name', client.name);
    body.set('stars', '0');
    body.set('description', '');

    return this.http.put<UpdatePersonResponse>(`${environment.apiUrl}/camera/object`, body)
      .pipe(
        map(() => Client.parse(cameraName, {
          id: client.id,
          name: client.name,
          stars: '0',
          description: '',
          face_detected: client.isFaceDetected
        })),
        tap(c => this.nameInfoCache.set(c.name, c)),
      );
  }
}
