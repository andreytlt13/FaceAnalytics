import {NgModule} from '@angular/core';
import {Routes, RouterModule} from '@angular/router';
import {DashboardComponent} from './dashboard.component';
import {AuthRequiredService} from '../shared/auth-guard/auth-guard.service';
import {CameraEditComponent} from './camera-edit/camera-edit.component';
import {CameraViewComponent} from './camera-view/camera-view.component';

const routes: Routes = [
  {
    path: 'dashboard',
    component: DashboardComponent,
    canActivate: [AuthRequiredService],
    canActivateChild: [AuthRequiredService],
    children: [
      {path: ':id', component: CameraViewComponent, data: {title: 'Camera View'}},
      {path: 'camera/create', component: CameraEditComponent, data: {title: 'Camera Edit'}},
    ]
  },
  {path: '**', redirectTo: '/dashboard'}
  // { path: 'dashboard/camera/create', component: CameraEditComponent, canActivate: [AuthRequiredService],  },
  // { path: 'dashboard/:id', component: DashboardComponent, canActivate: [AuthRequiredService],  }
];

@NgModule({
  imports: [RouterModule.forChild(
    routes
  )],
  exports: [RouterModule]
})
export class DashboardRoutingModule {
}
