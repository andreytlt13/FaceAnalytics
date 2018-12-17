import {Action, Selector, State, StateContext} from '@ngxs/store';
import {
  CreateCamera,
  DeleteCamera,
  GraphLoadedSuccess,
  LoadCameras,
  LoadGraphData,
  LoadHeatmap,
  ResetGraphs,
  SelectCamera
} from './dashboard.actions';
import {EventDataService} from './event-data/event-data.service';
import {Graph} from './event-data/graph';
import {tap} from 'rxjs/operators';
import {Camera} from './camera/camera';
import {CameraService} from './camera/camera.service';
import Heatmap from './event-data/heatmap';
import {of} from 'rxjs';

export interface DashboardStateModel {
  heatmapData: Heatmap;
  graphData: Graph[];
  cameras: Camera[];
  selectedCamera: Camera;
}

@State<DashboardStateModel>({
  name: 'dashboard',
  defaults: {
    graphData: [],
    heatmapData: null,
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
  static heatmapData(state: DashboardStateModel) {
    return state.heatmapData;
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
    private eventDataService: EventDataService
  ) {
  }

  @Action(LoadGraphData)
  loadGraphData({getState}: StateContext<DashboardStateModel>) {
    const ctx = getState().selectedCamera;

    return this.eventDataService.load(ctx.url);
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
  loadCameras({getState, patchState}: StateContext<DashboardStateModel>) {
    let {cameras} = getState();
    cameras = cameras.map(cmr => Camera.parse({...cmr}));
    const observable = cameras.length ? of(cameras) : this.cameraService.load();

    return observable.subscribe((cmrs: Camera[]) => {
      patchState({
        cameras: cmrs
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

  @Action(DeleteCamera)
  deleteCamera({getState, patchState}: StateContext<DashboardStateModel>, {payload}: CreateCamera) {
    return this.cameraService.delete(payload.camera)
      .pipe(
        tap((camera: Camera) => {
          const ctx = getState();
          patchState({
            cameras: [
              ...ctx.cameras.filter(cmr => cmr.id !== camera.id)
            ]
          });
        })
      );
  }

  @Action(LoadHeatmap)
  loadHeatmap({patchState}: StateContext<DashboardStateModel>, {payload}: LoadHeatmap) {
    return this.eventDataService.load(payload.camera.url)
      .pipe(
        tap((heatmapData: Heatmap) => {
          patchState({
            heatmapData
          });
        })
      );
  }

}
