import {NgModule} from '@angular/core';

import {LoginComponent} from './login.component';
import {SharedModule} from '../shared/shared.module';
import {LoginDialogComponent} from './login-dialog/login-dialog.component';

@NgModule({
  declarations: [LoginComponent, LoginDialogComponent],
  imports: [
    SharedModule
  ],
  entryComponents: [LoginDialogComponent]
})
export class LoginModule {
}
