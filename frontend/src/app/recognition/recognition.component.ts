import {Component, OnDestroy, OnInit} from '@angular/core';
import {ClientsService} from './clients/clients.service';
import {Client} from './clients/client';
import {map, mergeMap, tap} from 'rxjs/operators';
import {forkJoin, Observable, Subscription} from 'rxjs';
import {MatTableDataSource} from '@angular/material';

const CAMERA_URL = 'rtsp://admin:0ZKaxVFi@10.101.106.4:554/live/main';

@Component({
  selector: 'app-recognition',
  templateUrl: './recognition.component.html',
  styleUrls: ['./recognition.component.scss']
})
export class RecognitionComponent implements OnInit, OnDestroy {
  displayedColumns: string[] = ['person', 'matches'];

  dataSource$: Observable<any> = this.clientService.getUnknown(CAMERA_URL)
    .pipe(
      mergeMap(unknownClients => forkJoin(
        unknownClients.map(client => this.clientService.getMatched(CAMERA_URL, client)
          .pipe(
            map(clients => ({
                person: client,
                matches: clients
              })
            )
          )
        )
      ))
    );

  subscription: Subscription;

  clientsDS: MatTableDataSource<{
    person: Client,
    matches: Client[],
  }>;

  constructor(private readonly clientService: ClientsService) {}

  ngOnInit() {
    this.clientsDS = new MatTableDataSource();

    this.subscription = this.dataSource$.subscribe(data => {
      this.clientsDS.data = data;
    });
  }

  bind(person: Client, match: Client) {
    this.clientService.bind(CAMERA_URL, person, match)
      .pipe(
        tap(client => {
          this.clientsDS.data.splice(this.clientsDS.data.findIndex(item => item.person === person), 1, {person: client, matches: [match]});
          this.clientsDS.data = [...this.clientsDS.data];
        })
      )
      .subscribe();
  }

  trackByFn(i, item) {
    return `${item.person.id}-${item.person.name}-${item.matches.length}`;
  }

  rate(client: Client, stars: number) {
    client.stars = stars;

    this.clientService.update(CAMERA_URL, client)
      .pipe(
        tap(() => {
          this.clientsDS.data = [...this.clientsDS.data];
        })
      )
      .subscribe();
  }

  updateDescription(client: Client, description: string) {
    client.description = description;

    this.clientService.update(CAMERA_URL, client)
      .pipe(
        tap(() => {
          this.clientsDS.data = [...this.clientsDS.data];
        })
      )
      .subscribe();
  }

  ngOnDestroy() {
    this.subscription.unsubscribe();
  }
}
