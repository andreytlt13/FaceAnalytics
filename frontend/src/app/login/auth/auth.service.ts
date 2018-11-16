import {Injectable} from '@angular/core';
import {Observable, of} from 'rxjs';
import {LoginModule} from '../login.module';

import hmacSHA512 from 'crypto-js/hmac-sha256';
import Base64 from 'crypto-js/enc-base64';

@Injectable({
  providedIn: LoginModule
})
export class AuthService {
  private key = 'super secret key 123';
  private pwd = 'admin';

  constructor() {
  }

  login(username: string, password: string): Observable<boolean> {
    return of(username === 'admin' && password === this.pwd);
  }
}
