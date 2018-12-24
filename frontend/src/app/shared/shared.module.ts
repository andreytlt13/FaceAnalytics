import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';
import {
  MatButtonModule, MatCardModule,
  MatDialogModule,
  MatFormFieldModule, MatGridListModule,
  MatInputModule, MatListModule, MatMenuModule,
  MatProgressSpinnerModule,
  MatSidenavModule,
  MatButtonToggleModule,
  MatDatepickerModule,
  MatNativeDateModule
} from '@angular/material';
import {MatIconModule} from '@angular/material/icon';
import {MatToolbarModule} from '@angular/material/toolbar';
import {FormsModule} from '@angular/forms';
import {FlexLayoutModule} from '@angular/flex-layout';
import {LayoutModule} from '@angular/cdk/layout';

@NgModule({
  declarations: [],
  imports: [
    CommonModule,
    FormsModule,

    LayoutModule,

    FlexLayoutModule,

    MatDialogModule,
    MatFormFieldModule,
    MatIconModule,
    MatButtonModule,
    MatToolbarModule,
    MatIconModule,
    MatInputModule,
    MatProgressSpinnerModule,

    MatListModule,
    MatGridListModule,
    MatCardModule,
    MatMenuModule,

    MatSidenavModule,
    MatButtonToggleModule,
    MatDatepickerModule,
    MatNativeDateModule
  ],
  exports: [
    CommonModule,
    FormsModule,

    LayoutModule,

    FlexLayoutModule,

    MatDialogModule,
    MatFormFieldModule,
    MatIconModule,
    MatButtonModule,
    MatToolbarModule,
    MatIconModule,
    MatButtonModule,
    MatInputModule,
    MatProgressSpinnerModule,

    MatListModule,
    MatGridListModule,
    MatCardModule,
    MatMenuModule,

    MatSidenavModule,
    MatButtonToggleModule,
    MatDatepickerModule,
    MatNativeDateModule
  ]
})
export class SharedModule {
}
