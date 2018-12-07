import {Action, Selector, State, StateContext} from '@ngxs/store';
import {CreateCamera, GraphLoadedSuccess, LoadCameras, LoadGraphData, ResetGraphs, SelectCamera} from './dashboard.actions';
import {GraphDataService} from './graph-data/graph-data.service';
import {Graph} from './graph-data/graph';
import {tap} from 'rxjs/operators';
import {Camera} from './camera/camera';
import {CameraService} from './camera/camera.service';
import {Router} from '@angular/router';

export interface DashboardStateModel {
  graphData: Array<Graph>;
  cameras: Array<Camera>;
  selectedCamera: Camera;
}

@State<DashboardStateModel>({
  name: 'dashboard',
  defaults: {
    graphData: [],
    cameras: [],
    selectedCamera: null
  }
})
export class DashboardState {
  @Selector()
  static graphData(state: DashboardStateModel) {
    return state.graphData;
  }

  @Selector()
  static cameras(state: DashboardStateModel) {
    return state.cameras;
  }

  @Selector()
  static selectedCamera(state: DashboardStateModel) {
    return state.selectedCamera;
  }

  constructor(
    private cameraService: CameraService,
    private graphDataService: GraphDataService
  ) {
  }

  @Action(LoadGraphData)
  loadGraphData({dispatch}: StateContext<DashboardStateModel>) {
    return this.graphDataService.loadAll()
      .pipe(
        tap(graphs => graphs.forEach(graph => dispatch(new GraphLoadedSuccess({graph}))))
      );
  }

  @Action(GraphLoadedSuccess)
  graphDataLoaded({getState, patchState}: StateContext<DashboardStateModel>, {payload}: GraphLoadedSuccess) {
    const {graphData} = getState();
    patchState({
      graphData: [
        ...graphData,
        payload.graph
      ]
    });
  }

  @Action(ResetGraphs)
  resetGraphs({patchState}: StateContext<DashboardStateModel>) {
    patchState({
      graphData: []
    });
  }

  @Action(LoadCameras)
  loadCameras({patchState}: StateContext<DashboardStateModel>) {
    return this.cameraService.load().subscribe((cameras: Camera[]) => {
      patchState({
        cameras: cameras
      });
    });
  }

  @Action(SelectCamera)
  selectCamera({patchState}: StateContext<DashboardStateModel>, {payload}: SelectCamera) {
    patchState({
      selectedCamera: payload.camera
    });

    return payload.camera;
  }

  @Action(CreateCamera)
  createCamera({getState, patchState, dispatch}: StateContext<DashboardStateModel>, {payload}: CreateCamera) {
    return this.cameraService.create(payload.camera)
      .pipe(
        tap((camera: Camera) => {
          const ctx = getState();
          patchState({
            cameras: [
              ...ctx.cameras,
              camera
            ]
          });

          dispatch(new SelectCamera({camera}));
        })
      );
  }

}
