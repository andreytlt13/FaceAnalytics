import {Action, Selector, State, StateContext} from '@ngxs/store';
import {
  CreateCamera,
  DeleteCamera,
  LoadCameras,
  LoadGraphData,
  LoadHeatmap,
  SelectCamera,
  UpdateCamera
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
  graphData: Graph;
  cameras: Camera[];
  selectedCamera: Camera;
}

@State<DashboardStateModel>({
  name: 'dashboard',
  defaults: {
    graphData: null,
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

  // @Action(LoadGraphData)
  // loadGraphData({patchState}: StateContext<DashboardStateModel>, {payload}: LoadGraphData) {
  //   return this.eventDataService.loadGraph(payload.camera.url)
  //     .pipe(
  //       tap((graphData: Graph) => {
  //         patchState({
  //           graphData
  //         });
  //       })
  //     );
  // }

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
    // TODO: uncomment when backend supports camera API
    // return this.cameraService.create(payload.camera)
    //   .pipe(
    //     tap((camera: Camera) => {
    //       const {cameras} = getState();
    //       patchState({
    //         cameras: [
    //           ...cameras,
    //           camera
    //         ]
    //       });
    //
    //       dispatch(new SelectCamera({camera}));
    //     })
    //   );

    const {cameras} = getState();

    payload.camera.id = (cameras.reduce((memo, cmr) => +cmr.id > memo ? +cmr.id : memo, 0) + 1).toString();

    patchState({
      cameras: [
        ...cameras,
        payload.camera
      ]
    });

    return payload.camera;
  }

  @Action(DeleteCamera)
  deleteCamera({getState, patchState}: StateContext<DashboardStateModel>, {payload}: CreateCamera) {
    // TODO: uncomment when backend supports camera API
    // return this.cameraService.delete(payload.camera)
    //   .pipe(
    //     tap((camera: Camera) => {
    //       const ctx = getState();
    //       patchState({
    //         cameras: [
    //           ...ctx.cameras.filter(cmr => cmr.id !== camera.id)
    //         ]
    //       });
    //     })
    //   );

    const {cameras} = getState();

    patchState({
      cameras: [
        ...cameras.filter(cmr => cmr.id !== payload.camera.id)
      ]
    });
  }

  @Action(UpdateCamera)
  updateCamera({getState, patchState}: StateContext<DashboardStateModel>, {payload}: UpdateCamera) {
    const {cameras} = getState();
    const index = cameras.findIndex(cmr => cmr.id === payload.camera.id);

    if (index >= 0) {
      patchState({
        cameras: [
          ...cameras.map((v, i) => i === index ? payload.camera : v )
        ]
      });
    }
  }

  // @Action(LoadHeatmap)
  // loadHeatmap({patchState}: StateContext<DashboardStateModel>, {payload}: LoadHeatmap) {
  //   return this.eventDataService.loadHeatmap(payload.camera.url)
  //     .pipe(
  //       tap((heatmapData: Heatmap) => {
  //         patchState({
  //           heatmapData
  //         });
  //       })
  //     );
  // }

}
