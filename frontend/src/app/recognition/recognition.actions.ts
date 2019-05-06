import {Client} from './clients/client';

export class LoadClients {
  static readonly type = '[Recognition] Load clients';
}

export class LinkClients {
  static readonly type = '[Recognition] Map clients';

  constructor(public payload: { client1: Client, client2: Client }) {
  }
}

export class UpdateClient {
  static readonly type = '[Recognition] Update client';
  constructor(public payload: { client: Client }) {}
}
