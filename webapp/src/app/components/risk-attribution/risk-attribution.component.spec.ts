import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RiskAttributionComponent } from './risk-attribution.component';

describe('RiskAttributionComponent', () => {
  let component: RiskAttributionComponent;
  let fixture: ComponentFixture<RiskAttributionComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RiskAttributionComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(RiskAttributionComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
