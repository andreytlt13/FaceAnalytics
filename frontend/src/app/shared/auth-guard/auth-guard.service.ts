import {Injectable} from '@angular/core';
import {ActivatedRouteSnapshot, CanActivate, Router, RouterStateSnapshot} from '@angular/router';
import {Store} from '@ngxs/store';
import {Observable, of} from 'rxjs';
import {AuthState} from '../../login/auth.state';
import {LoginModule} from '../../login/login.module';

@Injectable({
  providedIn: LoginModule
})
export class AuthRequiredService implements CanActivate {

  constructor(private store: Store, private router: Router) {
  }

  canActivate(route: ActivatedRouteSnapshot, state: RouterStateSnapshot): Observable<boolean> {

    const username = this.store.selectSnapshot(AuthState.username);

    if (!this.isLoggedIn(username)) {
      this.router.navigate(['/login']);
    }

    return of(this.isLoggedIn(username));
  }

  isLoggedIn(username) {
    return !!username;
  }
}
