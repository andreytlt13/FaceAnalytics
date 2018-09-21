import { NgModule }              from '@angular/core';
import { RouterModule, Routes }  from '@angular/router';
import { PageNotFoundComponent } from './core/page-not-found/page-not-found.component';
import {HomepageComponent} from "./homepage/homepage.component";


const appRoutes: Routes = [
  { path: 'home', component: HomepageComponent },
  { path: '', redirectTo: 'home', pathMatch: 'full' },
  { path: '**', component: PageNotFoundComponent }
];
@NgModule({
  imports: [
    RouterModule.forRoot(
      appRoutes,
      // { enableTracing: true } // <-- debugging purposes only
    )
  ],
  exports: [
    RouterModule
  ]
})
export class AppRouterModule { }
