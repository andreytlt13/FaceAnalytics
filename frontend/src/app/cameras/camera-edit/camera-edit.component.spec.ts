import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { CameraEditComponent } from './camera-edit.component';

describe('CameraEditComponent', () => {
  let component: CameraEditComponent;
  let fixture: ComponentFixture<CameraEditComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ CameraEditComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(CameraEditComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
