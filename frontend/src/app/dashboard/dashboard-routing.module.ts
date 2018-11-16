import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { DashboardComponent } from './dashboard.component';
import {AuthRequiredService} from '../shared/auth-guard/auth-guard.service';

const routes: Routes = [
  { path: 'dashboard', component: DashboardComponent, canActivate: [AuthRequiredService],  },
  { path: 'dashboard/:id', component: DashboardComponent, canActivate: [AuthRequiredService],  }
];

@NgModule({
  imports: [RouterModule.forChild(
    routes,
    // { enableTracing: true }
  )],
  exports: [RouterModule]
})
export class DashboardRoutingModule { }
