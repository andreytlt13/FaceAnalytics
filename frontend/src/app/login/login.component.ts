import { Component, OnInit } from '@angular/core';;
import {MatDialog} from '@angular/material';
import {LoginDialogComponent} from './login-dialog/login-dialog.component';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss']
})
export class LoginComponent implements OnInit {

  constructor(public dialog: MatDialog) { }

  ngOnInit() {
    this.openDialog();
  }

  openDialog() {
    Promise.resolve().then(() => {
      this.dialog.open(LoginDialogComponent, {
        disableClose: true,
        width: '30%',
        minHeight: 250
      });
    });
  }
}
