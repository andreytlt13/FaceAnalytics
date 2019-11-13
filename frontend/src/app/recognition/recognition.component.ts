import {Component, OnDestroy, OnInit} from '@angular/core';
import {ClientsService} from './clients/clients.service';
import {Client} from './clients/client';
import {switchMap, tap} from 'rxjs/operators';
import {Observable, Subscription, timer} from 'rxjs';
import {MatTableDataSource} from '@angular/material';
import {environment} from '../../environments/environment';

const CAMERA_NAME = 'andrey_vitya_mp4';
const VIDEO_STREAM_URL = environment.videoStreamUrl;

@Component({
  selector: 'app-recognition',
  templateUrl: './recognition.component.html',
  styleUrls: ['./recognition.component.scss']
})
export class RecognitionComponent implements OnInit, OnDestroy {
  displayedColumns: string[] = ['person', 'matches'];
  streamUrl: string;

  dataSource$: Observable<any> = this.clientService.getUnknown(CAMERA_NAME);

  subscription: Subscription;

  clientsDS: MatTableDataSource<{
    person: Client,
    matches: Client[],
  }>;

  constructor(private readonly clientService: ClientsService) {}

  ngOnInit() {
    this.clientsDS = new MatTableDataSource();

    this.streamUrl = VIDEO_STREAM_URL;

    this.subscription = timer(0, 5000).pipe(
      switchMap(() => this.dataSource$),
      tap(data => {
        this.clientsDS.data = data;
      })
    ).subscribe();
  }

  create(person: Client) {
    this.clientService.create(CAMERA_NAME, person)
      .pipe(
        tap(() => {
          this.clientsDS.data.splice(this.clientsDS.data.findIndex(item => item.person === person), 1, {
            person,
            matches: [person]
          });
          this.clientsDS.data = [...this.clientsDS.data];
        })
      )
      .subscribe();
  }

  bind(person: Client, match: Client) {
    this.clientService.bind(CAMERA_NAME, person, match)
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

    this.clientService.update(CAMERA_NAME, client)
      .pipe(
        tap(() => {
          this.clientsDS.data = [...this.clientsDS.data];
        })
      )
      .subscribe();
  }

  updateDescription(client: Client, description: string) {
    client.description = description;

    this.clientService.update(CAMERA_NAME, client)
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
