import {Component, OnInit} from '@angular/core';
import {Location} from '@angular/common';
import {Camera} from '../camera/camera';
import {from, Observable, of} from 'rxjs';
import {DashboardState} from '../dashboard.state';
import {Actions, ofActionDispatched, Store} from '@ngxs/store';
import {CreateCamera, LoadGraphData, LoadHeatmap, SelectCamera, UpdateCamera} from '../dashboard.actions';
import {ActivatedRoute, Router} from '@angular/router';
import {filter, first, map, mergeMap, tap} from 'rxjs/operators';

@Component({
  selector: 'app-camera-edit',
  templateUrl: './camera-edit.component.html',
  styleUrls: ['./camera-edit.component.scss']
})
export class CameraEditComponent implements OnInit {
  public camera$: Observable<Camera> = this.route.paramMap.pipe(
    map(params => params.get('id')),
    mergeMap(cameraId => {
      if (!cameraId) {
        return of(new Camera());
      }
      return this.store.select(DashboardState.cameras)
        .pipe(
          mergeMap((cameras: Camera[]) => from(cameras)),
          first((camera: Camera) => camera.id === cameraId, new Camera()),
          map((camera: Camera) => Camera.parse({...camera.toJSON()}))
        );
    }),
    tap(elem => console.log(elem))
  );

  constructor(
    private store: Store,
    private router: Router,
    private route: ActivatedRoute,
    private actions: Actions,
    private location: Location
  ) {
  }

  ngOnInit() {
  }


  saveCamera(camera) {
    if (Camera.isValid(camera)) {
      const observable = camera.id ? this.store.dispatch(new UpdateCamera({camera: camera})) : this.store.dispatch(new CreateCamera({camera: camera}));

      return observable
        .subscribe(() => {
          this.router.navigate(['/dashboard', camera.id]);
        });
    } else {
      console.log('=== camera is not valid');
    }
  }

  cancel() {
    this.location.back();
  }

}
