import {Action, State, StateContext} from '@ngxs/store';
import {Client} from './clients/client';
import {LinkClients, LoadClients, UpdateClient} from './recognition.actions';

export interface RecognitionStateModel {
  clients: Client[];
}

@State<RecognitionStateModel>({
  name: 'recognition',
  defaults: {
    clients: []
  }
})
export class RecognitionState {

  @Action(LoadClients)
  loadClients({patchState}: StateContext<RecognitionStateModel>) {

  }

  @Action(LinkClients)
  linkClients() {}

  @Action(UpdateClient)
  selectCamera({patchState}: StateContext<RecognitionStateModel>, {payload}: UpdateClient) {}
}
