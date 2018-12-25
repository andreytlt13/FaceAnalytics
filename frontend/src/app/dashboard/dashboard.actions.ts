import {Graph} from './event-data/graph';
import {Camera} from './camera/camera';

export class LoadCameras {
  static readonly type = '[Dashboard] Load cameras';
}

export class SelectCamera {
  static readonly type = '[Dashboard] Select camera';
  constructor(public payload: { camera: Camera }) {}
}

// Camera Managing
export class CreateCamera {
  static readonly type = '[Dashboard] Create camera';
  constructor(public payload: { camera: Camera }) {}
}

export class DeleteCamera {
  static readonly type = '[Dashboard] Delete camera';
  constructor(public payload: { camera: Camera }) {};
}

export class UpdateCamera {
  static readonly type = '[Dashboard] Update camera';
  constructor(public payload: { camera: Camera }) {};
}


// Loading statistics
export class LoadHeatmap {
  static readonly type = '[Dashboard] Load heatmap';
  constructor(public payload: { camera: Camera }) {}
}

export class LoadGraphData {
  static readonly type = '[Dashboard] Load graph data';
  constructor(public payload: { camera: Camera }) {}
}
