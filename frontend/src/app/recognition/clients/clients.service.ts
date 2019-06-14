import {Injectable} from '@angular/core';
import {HttpClient, HttpParams} from '@angular/common/http';
import {Observable} from 'rxjs';
import {Client} from './client';
import {map} from 'rxjs/operators';

import {environment} from '../../../environments/environment';

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

@Injectable({
  providedIn: 'root'
})
export class ClientsService {
  constructor(private readonly http: HttpClient) {
  }

  getUnknown(cameraUrl: string): Observable<Client[]> {
    const params = new HttpParams();

    params.append('camera_url', cameraUrl);

    return this.http.get<UnknownPersonResponse>(`${environment.apiUrl}/camera/object_id`, {params})
      .pipe(
        map(response => response.object_id.map(id => Client.parse({id})))
      );
  }

  getMatched(cameraUrl: string, client: Client): Observable<Client[]> {
    const params = new HttpParams();

    params.append('camera_url', cameraUrl);
    params.append('object_id', client.id.toString());

    return this.http.get<MatchedPersonResponse>(`${environment.apiUrl}/camera/name`, {params})
      .pipe(
        map(response => response.names.map(name => Client.parse({name})))
      );
  }

  bind(cameraUrl: string, client1: Client, client2: Client): Observable<any> {
    return this.http.put(`${environment.apiUrl}/select_name`, {
      camera_url: cameraUrl,
      object_id: client1.id,
      name: client2.name
    });
  }
}
