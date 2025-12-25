import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DrawdownHeatmapComponent } from './drawdown-heatmap.component';

describe('DrawdownHeatmapComponent', () => {
  let component: DrawdownHeatmapComponent;
  let fixture: ComponentFixture<DrawdownHeatmapComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DrawdownHeatmapComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DrawdownHeatmapComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
