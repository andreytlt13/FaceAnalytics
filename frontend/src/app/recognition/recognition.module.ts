import {NgModule} from '@angular/core';
import {RecognitionRoutingModule, routedComponents} from './recognition-routing.module';
import {SharedModule} from '../shared/shared.module';
import {ClientsService} from './clients/clients.service';

@NgModule({
  imports: [
    SharedModule,

    RecognitionRoutingModule
  ],
  declarations: [routedComponents],
  providers: [ClientsService]
})
export class RecognitionModule {
}
