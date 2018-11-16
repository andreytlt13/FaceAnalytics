export class Login {
  static readonly type = '[Auth] Login';
  constructor(public payload: { username: string, password: string }) { }
}

export class LoginSuccess {
  static readonly type = '[Auth] Login Success';
  constructor(public payload: { username: string }) { }
}

export class LoginFailed {
  static readonly type = '[Auth] Login Failed';
}


export class Logout {
  static readonly type = '[Auth] Logout';
}

