import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EquityCurveComponent } from './equity-curve.component';

describe('EquityCurveComponent', () => {
  let component: EquityCurveComponent;
  let fixture: ComponentFixture<EquityCurveComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EquityCurveComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(EquityCurveComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
