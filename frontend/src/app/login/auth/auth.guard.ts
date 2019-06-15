import {Injectable} from '@angular/core';
import {ActivatedRouteSnapshot, CanActivate, CanActivateChild, Router, RouterStateSnapshot} from '@angular/router';
import {Store} from '@ngxs/store';
import {Observable, of} from 'rxjs';
import {AuthState} from '../auth.state';
import {LoginModule} from '../login.module';

@Injectable({
  providedIn: LoginModule
})
export class AuthGuard implements CanActivate, CanActivateChild {

  constructor(private store: Store, private router: Router) {
  }

  canActivate(route: ActivatedRouteSnapshot, state: RouterStateSnapshot): Observable<boolean> {

    const username = this.store.selectSnapshot(AuthState.username);

    if (!this.isLoggedIn(username)) {
      this.router.navigate(['/login']);
    }

    return of(this.isLoggedIn(username));
  }

  canActivateChild(route: ActivatedRouteSnapshot, state: RouterStateSnapshot): Observable<boolean> {
    return this.canActivate(route, state);
  }

  isLoggedIn(username: string) {
    return !!username;
  }
}
