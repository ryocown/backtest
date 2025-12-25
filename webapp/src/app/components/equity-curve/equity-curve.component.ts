import { Component, Input, OnChanges, SimpleChanges, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import * as Highcharts from 'highcharts/highstock';
import { BacktestData } from '../../models/data-models';

@Component({
  selector: 'app-equity-curve',
  standalone: true,
  imports: [CommonModule],
  template: '<div #chartContainer class="chart-container"></div>',
  styles: ['.chart-container { height: 500px; width: 100%; }']
})
export class EquityCurveComponent implements OnChanges, AfterViewInit {
  @ViewChild('chartContainer') chartContainer!: ElementRef;
  @Input() data: BacktestData | null = null;
  @Input() selectedStrategy: string | null = null;

  private chart: Highcharts.Chart | null = null;

  ngAfterViewInit(): void {
    if (this.data) {
      this.initChart();
    }
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['data'] && this.data && this.chartContainer) {
      this.initChart();
    }
  }

  private initChart() {
    if (!this.data) return;

    const series: Highcharts.SeriesOptionsType[] = this.data.strategies.map(strat => {
      const isSelected = this.selectedStrategy === strat;
      return {
        type: 'area',
        name: strat,
        data: this.data!.prices.map(p => [new Date(p['index']).getTime(), p[strat] as number]),
        tooltip: { valueDecimals: 2 },
        opacity: (!this.selectedStrategy || isSelected) ? 1.0 : 0.1,
        zIndex: isSelected ? 10 : 1,
        lineWidth: isSelected ? 3 : 1
      };
    });

    const options: Highcharts.Options = {
      chart: {
        backgroundColor: 'transparent',
        style: { fontFamily: 'Inter, sans-serif' }
      },
      title: { text: 'Cumulative Returns', style: { color: '#f8fafc' } },
      xAxis: {
        type: 'datetime',
        labels: { style: { color: '#94a3b8' } },
        gridLineColor: '#334155'
      },
      yAxis: {
        title: { text: 'Indexed Value', style: { color: '#94a3b8' } },
        labels: { style: { color: '#94a3b8' } },
        gridLineColor: '#334155'
      },
      legend: {
        itemStyle: { color: '#e2e8f0' },
        itemHoverStyle: { color: '#38bdf8' }
      },
      plotOptions: {
        area: {
          fillColor: {
            linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
            stops: [
              [0, 'rgba(56, 189, 248, 0.2)'],
              [1, 'rgba(56, 189, 248, 0)']
            ]
          },
          marker: { enabled: false },
          lineWidth: 2,
          states: { hover: { lineWidth: 3 } },
          threshold: null
        }
      },
      series: series,
      credits: { enabled: false }
    };

    if (this.chart) {
      this.chart.destroy();
    }
    this.chart = Highcharts.chart(this.chartContainer.nativeElement, options);
  }
}
