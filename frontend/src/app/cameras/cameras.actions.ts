import {Graph} from './event-data/graph';
import {Camera} from './camera/camera';

export class LoadCameras {
  static readonly type = '[Cameras] Load cameras';
}

export class SelectCamera {
  static readonly type = '[Cameras] Select camera';
  constructor(public payload: { camera: Camera }) {}
}

// Camera Managing
export class CreateCamera {
  static readonly type = '[Cameras] Create camera';
  constructor(public payload: { camera: Camera }) {}
}

export class DeleteCamera {
  static readonly type = '[Cameras] Delete camera';
  constructor(public payload: { camera: Camera }) {}
}

export class UpdateCamera {
  static readonly type = '[Cameras] Update camera';
  constructor(public payload: { camera: Camera }) {}
}


// Loading statistics
export class LoadHeatmap {
  static readonly type = '[Cameras] Load heatmap';
  constructor(public payload: { camera: Camera }) {}
}

export class LoadGraphData {
  static readonly type = '[Cameras] Load graph data';
  constructor(public payload: { camera: Camera }) {}
}
