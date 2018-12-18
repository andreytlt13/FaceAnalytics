import {formatDate} from '@angular/common';
import {CameraEvent} from './event-data.service';

const today = new Date();

export class Graph {
  public type = 'histogram';

  constructor(public name: string, private traces: Trace[]) {
  }

  get data() {
    return this.traces;
  }

  static parse(name: string, events: CameraEvent[]): Graph {
    const traces = [];

    events.forEach(({event_time, object_id}) => {
      let trace = traces[0];

      if (!trace) {
        trace = new Trace('default');
        traces.push(trace);
      }

      trace.x.push(event_time);
      trace.y.push(object_id);
    });

    // filling the graph with zeros up to now
    traces.forEach(({x, y}) => {
      const lastValue = x.pop();

      x.push(lastValue);

      if (formatDate(today, 'yyyy-MM-dd', 'en-US') !== lastValue) {
        x.push(formatDate(today, 'yyyy-MM-dd', 'en-US'));
        y.push(0);
      }
    });

    return new Graph(name, traces);
  }
}

class Trace {
  public type = 'histogram';
  constructor(public name: string|undefined, public x = [], public y = []) {}
}
