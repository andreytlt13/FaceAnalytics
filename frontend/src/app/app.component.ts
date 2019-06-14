import {Component, OnInit, ViewEncapsulation} from '@angular/core';
import {Actions, ofActionDispatched, Store} from '@ngxs/store';
import {LoginSuccess, Logout} from './login/auth.actions';
import {Router} from '@angular/router';
import {AuthState} from './login/auth.state';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
  encapsulation: ViewEncapsulation.None
})
export class AppComponent implements OnInit {
  loggedIn = false;

  constructor(private store: Store, private actions: Actions, private router: Router) { }

  ngOnInit(): void {
    this.actions.pipe(ofActionDispatched(Logout)).subscribe(() => {
      this.loggedIn = false;
      this.router.navigate(['/login']);
    });

    this.actions.pipe(ofActionDispatched(LoginSuccess)).subscribe(() => {
      this.loggedIn = true;
    });

    this.loggedIn = !!this.store.selectSnapshot(AuthState.username);
  }

  onSignOut() {
    this.store.dispatch(new Logout());
  }
}
