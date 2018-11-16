import { TestBed } from '@angular/core/testing';

import { AuthRequiredService } from './auth-guard.service';

describe('AuthGuardService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: AuthRequiredService = TestBed.get(AuthRequiredService);
    expect(service).toBeTruthy();
  });
});
