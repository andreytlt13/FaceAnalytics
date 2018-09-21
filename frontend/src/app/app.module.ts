import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { CommonModule } from "@angular/common";
import { FlexLayoutModule } from "@angular/flex-layout";

import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

import { AppRouterModule } from './app-router.module';
import { CoreModule } from "./core/core.module";
import {AppHeaderComponent} from "./components/app-header/app-header.component";
import {AppFooterComponent} from "./components/app-footer/app-footer.component";
import {MaterialModule} from "./material.module";
import { HomepageComponent } from './homepage/homepage.component';
import { LayoutModule } from '@angular/cdk/layout';
import { MatToolbarModule, MatButtonModule, MatSidenavModule, MatIconModule, MatListModule, MatCardModule } from '@angular/material';
import {HomeComponent} from "./components/home/home.component";
import {PlotlyModule} from "angular-plotly.js";

@NgModule({
  declarations: [
    AppComponent,
    AppHeaderComponent,
    AppFooterComponent,
    HomepageComponent,
    HomeComponent
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    AppRouterModule,
    CommonModule,
    PlotlyModule,
    FlexLayoutModule,
    CoreModule,
    MaterialModule,
    LayoutModule,
    MatToolbarModule,
    MatButtonModule,
    MatSidenavModule,
    MatIconModule,
    MatListModule,
    MatCardModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
