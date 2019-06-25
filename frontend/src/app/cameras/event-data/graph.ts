import {formatDate} from '@angular/common';
import {CameraEvent} from './event-data.service';

const today = new Date();

class Trace {
  public readonly type = 'histogram';
  public readonly histnorm = 'count';
  public readonly histfunc = 'count';

  // public autobinx = false;
  public nbinsx = 0;
  public xbins: {
    start: number;
    end: number;
    size: number;
  };

  constructor(
    public name: string | undefined,
    public x = [],
    public y = [],
    binOptions: {
      start: number;
      end: number;
      size: number;
      count: number;
    }
  ) {
    this.xbins = {
      start: binOptions.start,
      end: binOptions.end,
      size: binOptions.size
    };

    this.nbinsx = binOptions.count;
  }
}

export class Graph {
  public type = 'histogram';
  // public autobinx = false;

  constructor(public name: string, private traces: Trace[]) {
  }

  get data() {
    return this.traces;
  }

  static parse(name: string, events: CameraEvent[], binOptions): Graph {
    const traces = [];
    let object: number;

    events.forEach(({event_time, object_id}) => {
      let trace = new Trace('default', [], [], binOptions);

      if (!traces[0]) {
        traces.push(trace);
      } else {
        trace = traces[0];
      }

      if (object_id !== object) {
        trace.x.push(event_time);
        trace.y.push(object_id);
        object = object_id;
      }
    });

    // filling the graph with zeros up to now
    // traces.forEach(({x, y}) => {
    //   const lastValue = x.pop();
    //
    //   x.push(lastValue);
    //
    //   if (formatDate(today, 'yyyy-MM-dd', 'en-US') !== lastValue) {
    //     x.push(formatDate(today, 'yyyy-MM-dd', 'en-US'));
    //     y.push(0);
    //   }
    // });

    return new Graph(name, traces);
  }
}
