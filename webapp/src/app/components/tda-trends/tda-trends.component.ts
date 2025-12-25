import { Component, Input, OnChanges, SimpleChanges, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import * as Highcharts from 'highcharts';

@Component({
  selector: 'app-tda-trends',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './tda-trends.component.html',
  styleUrl: './tda-trends.component.css'
})
export class TdaTrendsComponent implements OnChanges, AfterViewInit {
  @Input() trends: any = null;

  @ViewChild('chartContainer') chartContainer!: ElementRef;

  private chart?: Highcharts.Chart;

  ngAfterViewInit(): void {
    if (this.trends) {
      this.renderChart();
    }
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (this.trends && changes['trends']) {
      this.renderChart();
    }
  }

  private renderChart() {
    if (!this.trends) return;

    const categories = this.trends.dates;

    // Process Betti 0 and Betti 1
    const betti0 = this.trends.betti.map((b: number[]) => b[0]);
    const betti1 = this.trends.betti.map((b: number[]) => b[1] || 0);

    const chartOptions: Highcharts.Options = {
      chart: {
        backgroundColor: 'transparent',
        type: 'line',
        height: 400
      },
      title: { text: '' },
      xAxis: {
        categories: categories,
        gridLineColor: '#334155',
        labels: { style: { color: '#94a3b8' } }
      },
      yAxis: [{ // Primary Y: Betti/Euler
        title: { text: 'Count / χ', style: { color: '#94a3b8' } },
        gridLineColor: '#334155',
        labels: { style: { color: '#94a3b8' } }
      }, { // Secondary Y: Correlation
        title: { text: 'Correlation', style: { color: '#94a3b8' } },
        opposite: true,
        gridLineColor: 'transparent',
        labels: { style: { color: '#94a3b8' } }
      }],
      legend: {
        itemStyle: { color: '#94a3b8' },
        itemHoverStyle: { color: '#f8fafc' }
      },
      credits: { enabled: false },
      tooltip: { shared: true, backgroundColor: '#0f172a', style: { color: '#f8fafc' } },
      series: [
        {
          name: 'Betti 0 (Components)',
          data: betti0,
          color: '#38bdf8',
          yAxis: 0
        },
        {
          name: 'Betti 1 (Cycles)',
          data: betti1,
          color: '#f87171',
          yAxis: 0
        },
        {
          name: 'Euler (χ)',
          data: this.trends.euler,
          color: '#a855f7',
          yAxis: 0,
          dashStyle: 'Dash'
        },
        {
          name: 'Avg Correlation',
          data: this.trends.avg_corr,
          color: '#22c55e',
          yAxis: 1
        }
      ] as any
    };

    if (!this.chartContainer) return;

    Highcharts.chart(this.chartContainer.nativeElement, chartOptions);
  }
}
