import {Component, OnInit} from '@angular/core';
import {Camera} from '../camera/camera';
import {Observable} from 'rxjs';
import {DashboardState} from '../dashboard.state';
import {Store} from '@ngxs/store';
import {Graph} from '../graph-data/graph';

@Component({
  selector: 'app-camera-view',
  templateUrl: './camera-view.component.html',
  styleUrls: ['./camera-view.component.scss']
})
export class CameraViewComponent implements OnInit {

  public selected$: Observable<Camera> = this.store.select(DashboardState.selectedCamera);

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

  constructor(private store: Store) {
  }

  ngOnInit() {
    // this.store.dispatch(new LoadGraphData);

    // this.actions.pipe(ofActionDispatched(GraphLoadedSuccess)).subscribe(() => {
    //   this.graphs = this.store.selectSnapshot(DashboardState.graphData);
    // });

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

}
