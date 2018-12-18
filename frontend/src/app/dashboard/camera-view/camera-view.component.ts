import {AfterViewInit, Component, ElementRef, OnDestroy, OnInit, ViewChild} from '@angular/core';
import {Camera} from '../camera/camera';
import {Observable} from 'rxjs';
import {DashboardState} from '../dashboard.state';
import {Actions, ofActionDispatched, Store} from '@ngxs/store';
import {Graph} from '../event-data/graph';
import {DeleteCamera, LoadGraphData, LoadHeatmap, SelectCamera} from '../dashboard.actions';

import h337 from 'heatmap.js';
import {ActivatedRoute, Router} from '@angular/router';
import Heatmap from '../event-data/heatmap';

@Component({
  selector: 'app-camera-view',
  templateUrl: './camera-view.component.html',
  styleUrls: ['./camera-view.component.scss']
})
export class CameraViewComponent implements OnInit, OnDestroy {

  public selected$: Observable<Camera> = this.store.select(DashboardState.selectedCamera);
  public heatmapData$: Observable<Heatmap> = this.store.select(DashboardState.heatmapData);
  public graphData$: Observable<Graph> = this.store.select(DashboardState.graphData);
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

  public streamLoading = false;

  private heatmapInstance;

  constructor(
    private store: Store,
    private router: Router,
    private route: ActivatedRoute,
  ) {
  }

  ngOnInit(): void {
    this.streamLoading = true;

    this.route.paramMap.subscribe((params) => {
      const cameraId = params.get('id');

      this.store.select(DashboardState.cameras).subscribe((cameras: Camera[]) => {
        const camera = cameras.find(cmr => cmr.id === cameraId);

        this.store.dispatch(new SelectCamera({camera})).subscribe(() => {
          this.streamLoading = true;
          this.store.dispatch(new LoadGraphData({camera}));
          this.store.dispatch(new LoadHeatmap({camera}));
        });
      });
    });

    this.heatmapData$.subscribe((heatmapData: Heatmap) => {
      if (heatmapData) {
        this.heatmapInstance = null;
        this.renderHeatmap(heatmapData);
      }
    });
  }

  ngOnDestroy(): void {
    this.heatmapInstance = null;
  }

  renderHeatmap(heatmapData: Heatmap) {
    const heatmap = this.heatmapElement.nativeElement;

    heatmap.innerHTML = '';

    const config = {
      container: heatmap,
      radius: 10,
      maxOpacity: .5,
      minOpacity: 0,
      blur: .75
    };
    // create heatmap with configuration
    this.heatmapInstance = h337.create(config);

    this.heatmapInstance.addData(heatmapData.dataPoints);
  }

  editCamera(camera: Camera) {
    this.router.navigate(['/dashboard', 'camera', 'update', camera.id]);
  }

  deleteCamera(camera: Camera) {
    this.store.dispatch(new DeleteCamera({camera}));
  }

}
