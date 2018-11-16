import {Action, Selector, State, StateContext} from '@ngxs/store';
import {GraphLoadedSuccess, LoadCameras, LoadGraphData, ResetGraphs, SelectCamera} from './dashboard.actions';
import {GraphDataService} from './graph-data/graph-data.service';
import {Graph} from './graph-data/graph';
import {tap} from 'rxjs/operators';
import {Camera} from './camera/camera';
import {CameraService} from './camera/camera.service';

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
  static graphData(state: DashboardStateModel) { return state.graphData; }

  @Selector()
  static cameras(state: DashboardStateModel) { return state.cameras; }

  @Selector()
  static selectedCamera(state: DashboardStateModel) { return state.selectedCamera; }

  constructor(private cameraSerivce: CameraService, private graphDataService: GraphDataService) {}

  @Action(LoadGraphData)
  loadGraphData({ dispatch }) {
    return this.graphDataService.loadAll()
      .pipe(
        tap(graphs => graphs.forEach(graph => dispatch(new GraphLoadedSuccess({graph}))))
      );
  }

  @Action(GraphLoadedSuccess)
  graphDataLoaded({ getState, patchState }: StateContext<DashboardStateModel>, { payload }: GraphLoadedSuccess) {
    const {graphData} = getState();
    patchState({
      graphData: [
        ...graphData,
        payload.graph
      ]
    });
  }

  @Action(ResetGraphs)
  resetGraphs({ patchState }: StateContext<DashboardStateModel>) {
    patchState({
      graphData: []
    });
  }

  @Action(LoadCameras)
  loadCameras({ patchState }: StateContext<DashboardStateModel>) {
    return this.cameraSerivce.load().subscribe((cameras) => {
      patchState({
        cameras: cameras
      });
    });
  }

  @Action(SelectCamera)
  selectCamera({ patchState }: StateContext<DashboardStateModel>, { payload }: SelectCamera) {
    patchState({
      selectedCamera: payload.camera
    });
  }

}
