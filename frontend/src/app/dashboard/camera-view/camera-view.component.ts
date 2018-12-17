import {AfterViewInit, Component, OnDestroy, OnInit} from '@angular/core';
import {Camera} from '../camera/camera';
import {Observable} from 'rxjs';
import {DashboardState} from '../dashboard.state';
import {Actions, ofActionDispatched, Store} from '@ngxs/store';
import {Graph} from '../event-data/graph';
import {LoadGraphData, LoadHeatmap, SelectCamera} from '../dashboard.actions';

import h337 from 'heatmap.js';
import {ActivatedRoute} from '@angular/router';
import {CameraService} from '../camera/camera.service';
import Heatmap from '../event-data/heatmap';

@Component({
  selector: 'app-camera-view',
  templateUrl: './camera-view.component.html',
  styleUrls: ['./camera-view.component.scss']
})
export class CameraViewComponent implements OnInit, OnDestroy {

  public selected$: Observable<Camera> = this.store.select(DashboardState.selectedCamera);
  public heatmapData$: Observable<Heatmap> = this.store.select(DashboardState.heatmapData);

  public graph: Graph;
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
    private route: ActivatedRoute,
    private actions: Actions,
    private cameraService: CameraService
  ) {
  }

  ngOnInit(): void {
    this.streamLoading = true;

    this.route.paramMap.subscribe((params) => {
      const cameraId = params.get('id');

      this.cameraService.load().subscribe((cameras: Camera[]) => {
        const camera = cameras.find(cmr => cmr.id === cameraId);

        this.store.dispatch(new SelectCamera({camera})).subscribe(() => {
          this.streamLoading = true;
          this.store.dispatch(new LoadGraphData);
          this.store.dispatch(new LoadHeatmap({camera}));
        });
      });
    });

    // this.store.dispatch(new LoadGraphData);

    // this.actions.pipe(ofActionDispatched(GraphLoadedSuccess)).subscribe(() => {
    //   this.graphs = this.store.selectSnapshot(DashboardState.graphData);
    // });

    this.generateRandomGraph();

    // this.renderHeatmap();

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

  generateRandomGraph() {
    const data = {
      rows: []
    };

    for (let i = 0; i < 60; i++) {
      const date = new Date();

      date.setDate(date.getDate() - i);

      const curr_date = date.getDate();
      const curr_month = date.getMonth() + 1;
      const curr_year = date.getFullYear();


      data.rows.push({
        date: curr_year + '-' + curr_month + '-' + curr_date,
        value: getRandomInt(0, 100)
      });
    }

    this.graph = Graph.parse('Objects statistics', data);

    function getRandomInt(min, max) {
      return Math.floor(Math.random() * (max - min)) + min;
    }
  }

  renderHeatmap(heatmapData: Heatmap) {
    const config = {
      container: document.getElementById('heatmap'),
      radius: 10,
      maxOpacity: .5,
      minOpacity: 0,
      blur: .75
    };
    // create heatmap with configuration
    this.heatmapInstance = h337.create(config);

    this.heatmapInstance.addData(heatmapData.dataPoints);
  }

}
