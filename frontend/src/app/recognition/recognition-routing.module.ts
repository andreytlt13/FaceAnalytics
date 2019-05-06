import {NgModule} from '@angular/core';
import {Routes, RouterModule} from '@angular/router';

import {RecognitionComponent} from './recognition.component';

const routes: Routes = [
  {path: 'recognition', component: RecognitionComponent},
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class RecognitionRoutingModule {
}

export const routedComponents = [RecognitionComponent];
