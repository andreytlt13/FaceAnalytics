import {Injectable} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {Observable, of} from 'rxjs';
import {People} from './mock-clients';
import {Client} from './client';
import {map} from 'rxjs/operators';

const CLIENTS_PATH = '';

@Injectable({
  providedIn: 'root'
})
export class ClientsService {
  constructor(private readonly http: HttpClient) {}
  getAll(): Observable<Client[]> {
    return of({clients: People})
      .pipe(
        map(({clients}) => clients.map(c => Client.parse(c)))
      );
  }

  get(id: number): Observable<Client> {
    return of({clients: [People.find(c => c.id === id)]})
      .pipe(
        map(({clients}) => Client.parse(clients[0]))
      );
  }

  map(client1: Client, client2: Client) {
    return of(true);
  }

  rate(client: Client) {
    return of(true);
  }

  update(client: Client) {
    return of(true);
  }
}
