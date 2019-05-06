import {BrowserModule} from '@angular/platform-browser';
import {NgModule} from '@angular/core';

import {NgxsModule} from '@ngxs/store';
import {NgxsStoragePluginModule} from '@ngxs/storage-plugin';
import {NgxsLoggerPluginModule} from '@ngxs/logger-plugin';

import {AppRoutingModule} from './app-routing.module';
import {AppComponent} from './app.component';
import {AuthState} from './login/auth.state';
import {CamerasModule} from './cameras/cameras.module';
import {LoginModule} from './login/login.module';
import {BrowserAnimationsModule} from '@angular/platform-browser/animations';
import {SharedModule} from './shared/shared.module';
import {environment} from '../environments/environment';
import {RecognitionModule} from './recognition/recognition.module';

@NgModule({
  declarations: [
    AppComponent
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,

    NgxsStoragePluginModule.forRoot({
      key: [
        'auth.username',
        'cameras.cameras'
      ]
    }),
    NgxsLoggerPluginModule.forRoot(),
    NgxsModule.forRoot([AuthState], { developmentMode: !environment.production }),

    CamerasModule,
    LoginModule,
    RecognitionModule,

    SharedModule,
    AppRoutingModule,
  ],
  bootstrap: [AppComponent]
})
export class AppModule {
}
