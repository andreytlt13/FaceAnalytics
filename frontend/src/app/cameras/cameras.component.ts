import {Component, OnInit, ViewEncapsulation} from '@angular/core';
import {Store} from '@ngxs/store';
import {LoadCameras, SelectCamera} from './cameras.actions';
import {CamerasState} from './cameras.state';
import {Observable} from 'rxjs';
import {BreakpointObserver, Breakpoints} from '@angular/cdk/layout';
import {map} from 'rxjs/operators';
import {Camera} from './camera/camera';
import {ActivatedRoute, Router} from '@angular/router';

@Component({
  selector: 'app-cameras',
  templateUrl: './cameras.component.html',
  styleUrls: ['./cameras.component.scss'],
  encapsulation: ViewEncapsulation.None
})
export class CamerasComponent implements OnInit {
  isHandset$: Observable<boolean> = this.breakpointObserver.observe(Breakpoints.Handset)
    .pipe(
      map(result => result.matches)
    );

  public cameras$: Observable<Camera[]> = this.store.select(CamerasState.cameras);

  public title = '';

  constructor(
    private breakpointObserver: BreakpointObserver,
    private store: Store,
    private router: Router,
    private route: ActivatedRoute
  ) {
  }

  ngOnInit() {
    this.store.dispatch(new LoadCameras());

    if (this.route.firstChild) {
      this.route.firstChild.data.subscribe((data: { title: string }) => {
        this.title = data.title;
      });
    }
  }

  createCamera() {
    this.title = 'Camera Create';
    this.router.navigate(['create'], {relativeTo: this.route});
  }
}
