import {Component, OnInit} from '@angular/core';
import {Camera} from '../camera/camera';
import {Observable} from 'rxjs';
import {DashboardState} from '../dashboard.state';
import {Store} from '@ngxs/store';

@Component({
  selector: 'app-camera-view',
  templateUrl: './camera-view.component.html',
  styleUrls: ['./camera-view.component.scss']
})
export class CameraViewComponent implements OnInit {

  public selected$: Observable<Camera> = this.store.select(DashboardState.selectedCamera);

  constructor(private store: Store) { }

  ngOnInit() {
  }

}
