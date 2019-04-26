import { TestBed } from '@angular/core/testing';

import { EventDataService } from './event-data.service';

describe('EventDataService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: EventDataService = TestBed.get(EventDataService);
    expect(service).toBeTruthy();
  });
});
