import {Injectable} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {Observable, of} from 'rxjs';
import {People} from './mock-clients';
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
    return this.http.get(`${environment.apiUrl}/get_person_id`);
  }

  get(id: number): Observable<Client> {
    return of({clients: [People.find(c => c.id === id)]})
      .pipe(
        map(({clients}) => Client.parse(clients[0]))
      );
  }

  map(client1: Client, client2: Client): Observable<any> {
    return this.http.put(`${environment.apiUrl}/select_name`, {
      object_id: 1,
      name: 'cococo'
    });
  }

  rate(client: Client) {
    return of(true);
  }

  update(client: Client) {
    return of(true);
  }
}
