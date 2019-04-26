import { Component, OnInit } from '@angular/core';
import {Actions, ofActionDispatched, Store} from '@ngxs/store';
import {Login, LoginFailed, LoginSuccess} from '../auth.actions';
import {Router} from '@angular/router';
import {MatDialogRef} from '@angular/material';

@Component({
  selector: 'app-login-dialog',
  templateUrl: './login-dialog.component.html',
  styleUrls: ['./login-dialog.component.scss']
})
export class LoginDialogComponent implements OnInit {

  username = '';
  password = '';

  loading = false;
  hide = true;

  constructor(
    private dialogRef: MatDialogRef<LoginDialogComponent>,
    private actions: Actions,
    private router: Router,
    private store: Store
  ) { }

  ngOnInit() {
    this.actions.pipe(ofActionDispatched(LoginSuccess)).subscribe(() => {
      this.dialogRef.close();
      this.router.navigate(['/cameras']);
    });

    this.actions.pipe(ofActionDispatched(LoginFailed)).subscribe(() => {
      this.loading = false;
    });
  }

  onLogin() {
    if (!this.username || !this.password) {
      return;
    }

    this.loading = true;

    this.store.dispatch(new Login({
      username: this.username,
      password: this.password
    }));
  }
}
