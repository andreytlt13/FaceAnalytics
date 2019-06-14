import {Injectable} from '@angular/core';
import {HttpClient, HttpParams} from '@angular/common/http';
import {Observable} from 'rxjs';
import {Client} from './client';
import {map} from 'rxjs/operators';

import {environment} from '../../../environments/environment';

const CLIENTS_PATH = '';


@Injectable({
  providedIn: 'root'
})
export class ClientsService {
  constructor(private readonly http: HttpClient) {}
  getAll(): Observable<any> {
    return this.http.get(`${environment.apiUrl}/get_person_id`)
      .pipe(
        map(response => response)
      );
  }

  getKnown(name: string): Observable<any> {
    const params =  new HttpParams();

    params.append('name', name);

    return this.http.get(`${environment.apiUrl}/get_face_known`, {params})
      .pipe(
        map(response => response)
      );
  }

  link(client1: Client, client2: Client): Observable<any> {
    return this.http.put(`${environment.apiUrl}/select_name`, {
      object_id: client1.id,
      name: client2.name
    });
  }

  getClientDescription(client: Client): Observable<any> {
    const params = new HttpParams();

    params.append('object_id', client.id.toString());

    return this.http.get(`${environment.apiUrl}/get_description`, {params});
  }

  update(client: Client): Observable<any> {
    return this.http.put(`${environment.apiUrl}/put_description`, {
      description: client.description,
      stars: client.rate ? client.rate : 0
    });
  }
}
