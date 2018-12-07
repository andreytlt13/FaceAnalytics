import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';
import {FormsModule} from '@angular/forms';
import {NgxsModule} from '@ngxs/store';
import {PlotlyModule} from 'angular-plotly.js';

import {SharedModule} from '../shared/shared.module';

import {DashboardComponent} from './dashboard.component';
import {DashboardState} from './dashboard.state';
import {GraphDataService} from './graph-data/graph-data.service';
import {HttpClientModule} from '@angular/common/http';
import {CameraService} from './camera/camera.service';
import {DashboardRoutingModule} from './dashboard-routing.module';
import { CameraEditComponent } from './camera-edit/camera-edit.component';
import { CameraViewComponent } from './camera-view/camera-view.component';

@NgModule({
  declarations: [DashboardComponent, CameraEditComponent, CameraViewComponent],
  imports: [
    CommonModule,
    FormsModule,
    HttpClientModule,
    SharedModule,
    PlotlyModule,

    NgxsModule.forFeature([DashboardState]),

    DashboardRoutingModule
  ],
  providers: [GraphDataService, CameraService]
})
export class DashboardModule {
}
