import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DataService } from '../../services/data.service';
import { DataPackage } from '../../models/data-models';
import { Observable } from 'rxjs';
import { Router } from '@angular/router';

import { EquityCurveComponent } from '../equity-curve/equity-curve.component';
import { TdaExplorerComponent } from '../tda-explorer/tda-explorer.component';
import { DrawdownHeatmapComponent } from '../drawdown-heatmap/drawdown-heatmap.component';
import { RiskAttributionComponent } from '../risk-attribution/risk-attribution.component';
import { MonthlyHeatmapComponent } from '../monthly-heatmap/monthly-heatmap.component';
import { PerformanceMetricChartComponent } from '../performance-metric-chart/performance-metric-chart.component';
import { ReturnDistributionComponent } from '../return-distribution/return-distribution.component';
import { TdaTrendsComponent } from '../tda-trends/tda-trends.component';

import { CorrelationMatrixComponent } from '../correlation-matrix/correlation-matrix.component';
import { ReturnAttributionComponent } from '../return-attribution/return-attribution.component';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [
    CommonModule,
    EquityCurveComponent,
    TdaExplorerComponent,
    DrawdownHeatmapComponent,
    RiskAttributionComponent,
    MonthlyHeatmapComponent,
    PerformanceMetricChartComponent,
    ReturnDistributionComponent,
    TdaTrendsComponent,
    CorrelationMatrixComponent,
    ReturnAttributionComponent
  ],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.css'
})
export class DashboardComponent implements OnInit {
  data$: Observable<DataPackage | null>;
  selectedStrategy: string | null = null;
  activeTab: 'overview' | 'performance' | 'risk' | 'tda' = 'overview';

  constructor(private dataService: DataService, private router: Router) {
    this.data$ = this.dataService.data$;
  }

  setTab(tab: 'overview' | 'performance' | 'risk' | 'tda') {
    this.activeTab = tab;
  }

  ngOnInit(): void {
    // Load default test export for verification
    this.dataService.loadLatestData('latest_backtest.json').subscribe(data => {
      if (data && data.backtest.strategies.length > 0) {
        this.selectedStrategy = data.backtest.strategies[0];
      }
    });
  }

  selectStrategy(strat: string) {
    this.selectedStrategy = strat;
  }

  goToConfig() {
    this.router.navigate(['/config']);
  }

  getRollingMetricOverlay(data: any, metric: string): Record<string, Record<string, number>> {
    const overlay: Record<string, Record<string, number>> = {};
    if (!data || !data.backtest || !data.backtest.rolling_stats) return overlay;

    Object.entries(data.backtest.rolling_stats).forEach(([strat, stats]: [string, any]) => {
      if (stats[metric]) {
        overlay[strat] = stats[metric];
      }
    });
    return overlay;
  }

  getStatsKeys(stats: any): string[] {
    return Object.keys(stats);
  }
}
