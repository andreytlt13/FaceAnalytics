import {Component, OnInit, ViewChild, ViewEncapsulation} from '@angular/core';
import {Actions, ofActionDispatched, Store} from '@ngxs/store';
import {GraphLoadedSuccess, LoadCameras, LoadGraphData, SelectCamera} from './dashboard.actions';
import {DashboardState} from './dashboard.state';
import {Observable} from 'rxjs';
import {BreakpointObserver, Breakpoints} from '@angular/cdk/layout';
import {map} from 'rxjs/operators';
import {Graph} from './graph-data/graph';
import {Camera} from './camera/camera';
import {ActivatedRoute, Router} from '@angular/router';
import {MatDrawer} from '@angular/material';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
  encapsulation: ViewEncapsulation.None
})
export class DashboardComponent implements OnInit {
  isHandset$: Observable<boolean> = this.breakpointObserver.observe(Breakpoints.Handset)
    .pipe(
      map(result => result.matches)
    );
  public graphs: Array<Graph> = [];
  public layout = {
    autosize: true,
    barmode: 'group',
    xaxis: {
      rangeslider: {
        yaxis: {
          rangemode: 'auto'
        }
      }
    },
    yaxis: {
      autorange: true,
      fixedrange: false
    }
  };

  public cameras$: Observable<Camera[]> = this.store.select(DashboardState.cameras);

  public title = '';

  constructor(
    private breakpointObserver: BreakpointObserver,
    private store: Store,
    private router: Router,
    private route: ActivatedRoute
  ) {}

  ngOnInit() {
    this.store.dispatch(new LoadCameras());

    if (this.route.firstChild) {
      this.route.firstChild.data.subscribe((data: {title: string}) => {
        this.title = data.title;
      });
    }
    // this.store.dispatch(new LoadGraphData);

    // this.actions.pipe(ofActionDispatched(GraphLoadedSuccess)).subscribe(() => {
    //   this.graphs = this.store.selectSnapshot(DashboardState.graphData);
    // });
  }

  selectCamera(camera: Camera) {
    this.title = 'Camera View';
    this.store.dispatch(new SelectCamera({camera}));
  }

  editCamera() {
    this.title = 'Camera Edit';
    this.router.navigate(['camera/create'], {relativeTo: this.route});
  }
}
