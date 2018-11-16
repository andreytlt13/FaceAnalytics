import {Action, Selector, State, StateContext} from '@ngxs/store';
import {Login, LoginFailed, LoginSuccess, Logout} from './auth.actions';
import {AuthService} from './auth/auth.service';
import {delay, tap} from 'rxjs/operators';
import {ResetGraphs} from '../dashboard/dashboard.actions';

export interface AuthStateModel {
  username: string;
}

@State<AuthStateModel>({
  name: 'auth',
  defaults: {
    username: ''
  }
})
export class AuthState {
  @Selector()
  static username(state: AuthStateModel) { return state.username; }

  constructor(private authService: AuthService) {}

  @Action(Login)
  login({ dispatch }: StateContext<AuthStateModel>, { payload }: Login) {
    return this.authService.login(payload.username, payload.password)
      .pipe(
        delay(2000),
        tap(success => (success ? dispatch(new LoginSuccess({username: payload.username})) : dispatch(LoginFailed)))
      );
  }

  @Action(Logout)
  logout({ patchState, dispatch }: StateContext<AuthStateModel>) {
    dispatch(new ResetGraphs());

    patchState({username: ''});
  }

  @Action(LoginSuccess)
  loginSuccess({ patchState }: StateContext<AuthStateModel>, { payload: { username }}) {
    patchState({ username });
  }

  @Action(LoginFailed)
  loginFailed() {}
}
