import {NgModule} from '@angular/core';

import {SharedModule} from '../shared/shared.module';
import {LoginDialogComponent} from './login-dialog/login-dialog.component';
import {LoginRoutingModule, routedComponents} from './login-routing.module';

@NgModule({
  declarations: [
    routedComponents,
    LoginDialogComponent
  ],
  imports: [
    SharedModule,

    LoginRoutingModule
  ],
  entryComponents: [LoginDialogComponent]
})
export class LoginModule {
}
