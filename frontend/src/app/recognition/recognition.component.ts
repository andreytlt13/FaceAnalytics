import {Component, OnInit} from '@angular/core';
import * as moment from 'moment';
import {ClientsService} from './clients/clients.service';

@Component({
  selector: 'app-recognition',
  templateUrl: './recognition.component.html',
  styleUrls: ['./recognition.component.scss']
})
export class RecognitionComponent implements OnInit {
  displayedColumns: string[] = ['eventTime', 'photo', 'matches'];
  dataSource = Array.from({length: 10}).map((value, index) => {
    return {
      eventTime: moment(RecognitionComponent.randomDate(new Date(2019, 0, 1), new Date())).format(),
      photo: `assets/photos/${index + 1}.jpg`,
      matches: Array.from({length: 3}).map(() => {
        return `assets/photos/${Math.floor(Math.random() * 10 + 1)}.jpg`;
      })
    };
  });


  static randomDate(start: Date, end: Date) {
    return new Date(start.getTime() + Math.random() * (end.getTime() - start.getTime()));
  }

  constructor(private readonly clientService: ClientsService) {}

  ngOnInit() {
    console.log(this.dataSource);

    this.clientService.getAll()
      .subscribe((data) => {
        console.log(data);
      });

    this.clientService.map(null, null)
      .subscribe((data) => {
        console.log(data);
      });
  }
}
