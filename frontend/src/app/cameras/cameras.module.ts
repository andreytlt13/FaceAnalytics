import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';
import {FormsModule} from '@angular/forms';
import {NgxsModule} from '@ngxs/store';
import {PlotlyModule} from 'angular-plotly.js';

import {SharedModule} from '../shared/shared.module';

import {CamerasComponent} from './cameras.component';
import {CamerasState} from './cameras.state';
import {EventDataService} from './event-data/event-data.service';
import {HttpClientModule} from '@angular/common/http';
import {CameraService} from './camera/camera.service';
import {CamerasRoutingModule} from './cameras-routing.module';
import {CameraEditComponent} from './camera-edit/camera-edit.component';
import {CameraViewComponent} from './camera-view/camera-view.component';

@NgModule({
  declarations: [CamerasComponent, CameraEditComponent, CameraViewComponent],
  imports: [
    CommonModule,
    FormsModule,
    HttpClientModule,
    SharedModule,
    PlotlyModule,

    NgxsModule.forFeature([CamerasState]),

    CamerasRoutingModule
  ],
  providers: [EventDataService, CameraService]
})
export class CamerasModule {
}
