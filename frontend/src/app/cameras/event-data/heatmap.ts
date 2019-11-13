import {CameraEvent} from './event-data.service';

export default class Heatmap {
  constructor(public dataPoints: HeatmapPoint[] = []) {
  }

  static parse(events: CameraEvent[]) {
    return new Heatmap(events.map((event: CameraEvent) => {
      return {
        x: event.centroid_x,
        y: event.centroid_y,
        value: 1
      };
    }));
  }
}

export interface HeatmapPoint {
  x: number;
  y: number;
  value: number;
}
