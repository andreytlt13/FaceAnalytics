import {Graph} from './event-data/graph';
import {Camera} from './camera/camera';

export class LoadCameras {
  static readonly type = '[Dashboard] Load cameras';
}

export class SelectCamera {
  static readonly type = '[Dashboard] Select camera';
  constructor(public payload: { camera: Camera }) {}
}

export class CreateCamera {
  static readonly type = '[Dashboard] Create camera';
  constructor(public payload: { camera: Camera }) {}
}

export class LoadGraphData {
  static readonly type = '[Dashboard] Load graph data';
}

export class GraphLoadedSuccess {
  static readonly type = '[Dashboard] Graph data loaded';
  constructor(public payload: { graph: Graph }) {}
}

export class ResetGraphs {
  static readonly type = '[Dashboard] Removing all graphs';
}

export class DeleteCamera {
  static readonly type = '[Dashboard] Delete camera';
  constructor(public payload: { camera: Camera }) {};
}

export class LoadHeatmap {
  static readonly type = '[Dashboard] Load heatmap';

  constructor(public payload: { camera: Camera }) {}
}
