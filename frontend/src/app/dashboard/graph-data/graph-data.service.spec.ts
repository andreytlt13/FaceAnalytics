import { TestBed } from '@angular/core/testing';

import { GraphDataService } from './graph-data.service';

describe('GraphDataService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: GraphDataService = TestBed.get(GraphDataService);
    expect(service).toBeTruthy();
  });
});
