import {NgModule} from '@angular/core';
import {Routes, RouterModule} from '@angular/router';
import {CamerasComponent} from './cameras.component';
import {AuthRequiredService} from '../shared/auth-guard/auth-guard.service';
import {CameraEditComponent} from './camera-edit/camera-edit.component';
import {CameraViewComponent} from './camera-view/camera-view.component';

const routes: Routes = [
  {
    path: 'cameras',
    component: CamerasComponent,
    canActivate: [AuthRequiredService],
    canActivateChild: [AuthRequiredService],
    children: [
      {path: 'create', component: CameraEditComponent, data: {title: 'Camera Create'}},
      {path: 'update/:id', component: CameraEditComponent, data: {title: 'Camera Edit'}},
      {path: ':id', component: CameraViewComponent, data: {title: 'Camera View'}},
    ]
  }
];

@NgModule({
  imports: [
    RouterModule.forChild(routes)
  ],
  exports: [RouterModule]
})
export class CamerasRoutingModule {
}
