import {Component, OnInit} from '@angular/core';
import {Location} from '@angular/common';
import {Camera} from '../camera/camera';
import {Observable} from 'rxjs';
import {DashboardState} from '../dashboard.state';
import {Actions, ofActionDispatched, Store} from '@ngxs/store';
import {CreateCamera, SelectCamera} from '../dashboard.actions';
import {Router} from '@angular/router';

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
    private router: Router,
    private actions: Actions,
    private location: Location
  ) {
  }

  ngOnInit() {
    this.selected$.subscribe((camera: Camera) => {
      if (camera) {
        this.camera = camera;
      }
    });

    // this.actions.pipe(ofActionDispatched(SelectCamera)).subscribe(({payload: {camera}}) => {
    //   this.router.navigate(['dashboard', camera.id]);
    // });
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
