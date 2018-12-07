import { Component, OnInit } from '@angular/core';
import {Location} from '@angular/common';
import {tap} from 'rxjs/operators';
import {Camera} from '../camera/camera';
import {Observable} from 'rxjs';
import {DashboardState} from '../dashboard.state';
import {Store} from '@ngxs/store';
import {CreateCamera} from '../dashboard.actions';

@Component({
  selector: 'app-camera-edit',
  templateUrl: './camera-edit.component.html',
  styleUrls: ['./camera-edit.component.scss']
})
export class CameraEditComponent implements OnInit {
  public selected$: Observable<Camera> = this.store.select(DashboardState.selectedCamera);
  public camera = new Camera();

  constructor(
    private store: Store,
    private location: Location
  ) { }

  ngOnInit() {
    console.log(this.camera);
    this.selected$.pipe(
      tap((camera: Camera) => {
        this.camera = camera;
      })
    );
  }

  saveCamera() {
    if (Camera.isValid(this.camera)) {
      this.store.dispatch(new CreateCamera({camera: this.camera}));
    }
  }

  cancel() {
    this.location.back();
  }

}
