import {AfterViewInit, Component, ElementRef, OnDestroy, OnInit, ViewChild} from '@angular/core';
import {Camera} from '../camera/camera';
import {Observable, of} from 'rxjs';
import {DashboardState} from '../dashboard.state';
import {Actions, ofActionDispatched, Store} from '@ngxs/store';
import {Graph} from '../event-data/graph';
import {DeleteCamera, LoadGraphData, LoadHeatmap, SelectCamera} from '../dashboard.actions';

import h337 from 'heatmap.js';
import {ActivatedRoute, Router} from '@angular/router';
import Heatmap from '../event-data/heatmap';
import {CameraEvent, EventDataService} from '../event-data/event-data.service';
import {catchError, delay, map, tap} from 'rxjs/operators';
import {DateAdapter, MAT_DATE_FORMATS, MAT_DATE_LOCALE} from '@angular/material';
import {MomentDateAdapter} from '@angular/material-moment-adapter';
import {Moment} from 'moment';
import * as moment from 'moment';

export const MY_FORMATS = {
  parse: {
    dateInput: 'YYYY-MM-DD',
  },
  display: {
    dateInput: 'YYYY-MM-DD',
    monthYearLabel: 'MMM YYYY',
    dateA11yLabel: 'YYYY-MM-DD',
    monthYearA11yLabel: 'MM YYYY',
  },
};

@Component({
  selector: 'app-camera-view',
  templateUrl: './camera-view.component.html',
  styleUrls: ['./camera-view.component.scss'],
  providers: [
    {provide: DateAdapter, useClass: MomentDateAdapter, deps: [MAT_DATE_LOCALE]},
    {provide: MAT_DATE_FORMATS, useValue: MY_FORMATS}
  ]
})
export class CameraViewComponent implements OnInit, OnDestroy {

  public selected$: Observable<Camera> = this.store.select(DashboardState.selectedCamera);
  // public heatmapData$: Observable<Heatmap> = this.store.select(DashboardState.heatmapData);
  public graphData$: Observable<Graph>; // = this.store.select(DashboardState.graphData);

  public startDate: Moment = moment().startOf('day');
  public endDate: Moment = moment().endOf('day');

  public graphLoading = false;

  @ViewChild('heatmap') private heatmapElement: ElementRef;

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

  public play = false;
  public streamLoading = false;

  private heatmapInstance;

  constructor(
    private store: Store,
    private router: Router,
    private route: ActivatedRoute,
    private eventDataService: EventDataService
  ) {
  }

  ngOnInit(): void {
    this.streamLoading = true;

    this.route.paramMap.subscribe((params) => {
      this.play = false;
      const cameraId = params.get('id');

      this.store.select(DashboardState.cameras).subscribe((cameras: Camera[]) => {
        const camera = cameras.find(cmr => cmr.id === cameraId);

        if (camera) {
          this.store.dispatch(new SelectCamera({camera})).subscribe(() => {
            this.streamLoading = true;
            // this.store.dispatch(new LoadGraphData({camera}));
            // this.store.dispatch(new LoadHeatmap({camera}));

            this.reloadGraph(camera);
          });
        }
      });
    });

    // this.heatmapData$.subscribe((heatmapData: Heatmap) => {
    //   if (heatmapData) {
    //     this.heatmapInstance = null;
    //     this.renderHeatmap(heatmapData);
    //   }
    // });
  }

  ngOnDestroy(): void {
    this.heatmapInstance = null;
    this.heatmapElement.nativeElement.innerHTML = '';
  }

  resetHeatmap(): void {
    const heatmapElement = this.heatmapElement.nativeElement;
    heatmapElement.innerHTML = '';

    const heatmap = this.heatmapElement.nativeElement;

    const config = {
      container: heatmap,
      radius: 10,
      maxOpacity: .5,
      minOpacity: 0,
      blur: .75
    };

    this.heatmapInstance = h337.create(config);
  }

  reloadGraph(camera: Camera): void {
    this.graphLoading = true;
    this.graphData$ = this.eventDataService
      .load(camera.url, this.startDate.format('YYYY-MM-DD HH:mm:ss'), this.endDate.endOf('day').format('YYYY-MM-DD HH:mm:ss'))
      .pipe(
        tap(() => this.graphLoading = false),
        map((events: CameraEvent[]) => Graph.parse('Unique objects per date', events))
      );
  }

  async playHeatmap(camera: Camera) {
    this.resetHeatmap();
    this.play = true;

    const start = this.startDate.clone();

    while (start < this.endDate.endOf('day')) {
      if (!this.play) {
        break;
      }
      const heatmap = await this.eventDataService
        .load(camera.url, start.format('YYYY-MM-DD HH:mm:ss'), start.add(30, 'minute').format('YYYY-MM-DD HH:mm:ss'))
        .pipe(
          // delay(100),
          catchError(() => of([])),
          map((events: CameraEvent[]) => Heatmap.parse(events)),
        )
        .toPromise();

      this.renderHeatmap(heatmap);

      start.add(30, 'minute');
    }
  }

  renderHeatmap(heatmapData: Heatmap, cumulative: boolean = true) {
    // create heatmap with configuration
    if (cumulative) {
      this.heatmapInstance.addData(heatmapData.dataPoints);
    } else {
      this.heatmapInstance.setData(heatmapData.dataPoints);
    }
  }

  editCamera(camera: Camera) {
    this.router.navigate(['/dashboard', 'camera', 'update', camera.id]);
  }

  deleteCamera(camera: Camera) {
    this.store.dispatch(new DeleteCamera({camera}));
  }

}
