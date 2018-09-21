import { Component } from '@angular/core';
import { BreakpointObserver, Breakpoints, BreakpointState } from '@angular/cdk/layout';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

@Component({
  selector: 'app-homepage',
  templateUrl: './homepage.component.html',
  styleUrls: ['./homepage.component.scss']
})
export class HomepageComponent {
  cameras: Array<object> = [
    {name: 'Camera 1'},
    {name: 'Camera 2'},
    {name: 'Camera 3'}
  ];

  public graph = {
    data: [
      { x: [1, 2, 3], y: [2, 5, 3], type: 'bar' },
    ],
    layout: {width: 1024, height: 768, title: 'A Fancy Plot'}
  };

  isHandset$: Observable<boolean> = this.breakpointObserver.observe(Breakpoints.Handset)
    .pipe(
      map(result => result.matches)
    );

  constructor(private breakpointObserver: BreakpointObserver) {}

  }
