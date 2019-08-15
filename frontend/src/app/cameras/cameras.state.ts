import {Action, Selector, State, StateContext} from '@ngxs/store';
import {
  CreateCamera,
  DeleteCamera,
  LoadCameras,
  LoadGraphData,
  LoadHeatmap,
  SelectCamera,
  UpdateCamera
} from './cameras.actions';
import {EventDataService} from './event-data/event-data.service';
import {Graph} from './event-data/graph';
import {tap} from 'rxjs/operators';
import {Camera} from './camera/camera';
import {CameraService} from './camera/camera.service';
import Heatmap from './event-data/heatmap';
import {of} from 'rxjs';

export interface CamerasStateModel {
  heatmapData: Heatmap;
  graphData: Graph;
  cameras: Camera[];
  selectedCamera: Camera;
}

@State<CamerasStateModel>({
  name: 'cameras',
  defaults: {
    graphData: null,
    heatmapData: null,
    cameras: [],
    selectedCamera: null
  }
})
export class CamerasState {
  @Selector()
  static graphData(state: CamerasStateModel) {
    return state.graphData;
  }

  @Selector()
  static heatmapData(state: CamerasStateModel) {
    return state.heatmapData;
  }

  @Selector()
  static cameras(state: CamerasStateModel) {
    return state.cameras;
  }

  @Selector()
  static selectedCamera(state: CamerasStateModel) {
    return state.selectedCamera;
  }

  constructor(
    private cameraService: CameraService,
    private eventDataService: EventDataService
  ) {
  }

  // @Action(LoadGraphData)
  // loadGraphData({patchState}: StateContext<CamerasStateModel>, {payload}: LoadGraphData) {
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
  loadCameras({getState, patchState}: StateContext<CamerasStateModel>) {
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
  selectCamera({patchState}: StateContext<CamerasStateModel>, {payload}: SelectCamera) {
    patchState({
      selectedCamera: payload.camera
    });

    return payload.camera;
  }

  @Action(CreateCamera)
  createCamera({getState, patchState, dispatch}: StateContext<CamerasStateModel>, {payload}: CreateCamera) {
    return this.cameraService.create(payload.camera)
      .pipe(
        tap((camera: Camera) => {
          const {cameras} = getState();
          patchState({
            cameras: [
              ...cameras,
              camera
            ]
          });

          dispatch(new SelectCamera({camera}));
        })
      );

  }

  @Action(DeleteCamera)
  deleteCamera({getState, patchState}: StateContext<CamerasStateModel>, {payload}: CreateCamera) {
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

  @Action(UpdateCamera)
  updateCamera({getState, patchState}: StateContext<CamerasStateModel>, {payload}: UpdateCamera) {
    const {cameras} = getState();
    const index = cameras.findIndex(cmr => cmr.id === payload.camera.id);

    if (index >= 0) {
      patchState({
        cameras: [
          ...cameras.map((v, i) => i === index ? payload.camera : v )
        ]
      });

      return payload.camera;
    }
  }

  // @Action(LoadHeatmap)
  // loadHeatmap({patchState}: StateContext<CamerasStateModel>, {payload}: LoadHeatmap) {
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
