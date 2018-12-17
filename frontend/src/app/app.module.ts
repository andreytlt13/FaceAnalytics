import {BrowserModule} from '@angular/platform-browser';
import {NgModule} from '@angular/core';

import {NgxsModule} from '@ngxs/store';
import {NgxsStoragePluginModule} from '@ngxs/storage-plugin';
import {NgxsLoggerPluginModule} from '@ngxs/logger-plugin';

import {AppRoutingModule} from './app-routing.module';
import {AppComponent} from './app.component';
import {AuthState} from './login/auth.state';
import {DashboardModule} from './dashboard/dashboard.module';
import {LoginModule} from './login/login.module';
import {BrowserAnimationsModule} from '@angular/platform-browser/animations';
import {SharedModule} from './shared/shared.module';
import {environment} from '../environments/environment';

@NgModule({
  declarations: [
    AppComponent
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    AppRoutingModule,

    DashboardModule,
    LoginModule,

    NgxsStoragePluginModule.forRoot({
      key: [
        'auth.username',
        'dashboard.cameras'
      ]
    }),
    NgxsLoggerPluginModule.forRoot(),
    NgxsModule.forRoot([AuthState], { developmentMode: !environment.production }),

    SharedModule
  ],
  bootstrap: [AppComponent]
})
export class AppModule {
}
