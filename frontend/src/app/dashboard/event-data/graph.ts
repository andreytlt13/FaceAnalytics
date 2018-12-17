import {formatDate} from '@angular/common';
import {CameraEvent} from './event-data.service';

const today = new Date();

export class Graph {
  public type = 'bar';

  constructor(public name: string, private traces: Trace[]) {
  }

  get data() {
    return this.traces;
  }

  // static parse(name: string, events: CameraEvent[]): Graph {
  static parse(name: string, {rows}: any): Graph {
    const traces = [];

    // events.forEach(({}) => {
    //   let trace = traces[0];
    //
    //   if (!trace) {
    //     trace = new Trace('default');
    //     traces.push(trace);
    //   }
    //
    //   trace.x.push(date);
    //   trace.y.push(value);
    // });
    rows.forEach(({date, value, org}) => {
      let trace = org ? traces.find(t => org === t.name) : traces[0];

      if (!trace) {
        trace = new Trace(org);
        traces.push(trace);
      }

      trace.x.push(date);
      trace.y.push(value);
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
  public type = 'bar';
  constructor(public name: string|undefined, public x = [], public y = []) {}
}
