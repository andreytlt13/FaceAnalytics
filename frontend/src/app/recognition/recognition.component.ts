import {Component, OnInit} from '@angular/core';
import * as moment from 'moment';
import {ClientsService} from './clients/clients.service';
import {Client} from './clients/client';
import {map, mergeAll, mergeMap, tap} from 'rxjs/operators';
import {forkJoin, Observable} from 'rxjs';

const CAMERA_URL = 'rtsp://admin:0ZKaxVFi@10.101.106.4:554/live/main';

@Component({
  selector: 'app-recognition',
  templateUrl: './recognition.component.html',
  styleUrls: ['./recognition.component.scss']
})
export class RecognitionComponent implements OnInit {
  displayedColumns: string[] = ['eventTime', 'photo', 'matches'];
  dataSource$: Observable<{
    person: Client,
    matches: Client[],
  }[]> = this.clientService.getUnknown(CAMERA_URL)
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
      )),
      tap(console.log)
    );


  // static randomDate(start: Date, end: Date) {
  //   return new Date(start.getTime() + Math.random() * (end.getTime() - start.getTime()));
  // }

  constructor(private readonly clientService: ClientsService) {}

  ngOnInit() {

  }
}
