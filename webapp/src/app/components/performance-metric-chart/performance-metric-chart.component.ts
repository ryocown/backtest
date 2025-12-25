import { Component, Input, OnChanges, SimpleChanges, LOCALE_ID, Inject, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule, formatDate } from '@angular/common';
import * as Highcharts from 'highcharts';

@Component({
  selector: 'app-performance-metric-chart',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './performance-metric-chart.component.html',
  styleUrl: './performance-metric-chart.component.css'
})
export class PerformanceMetricChartComponent implements OnChanges, AfterViewInit {
  @Input() data: Record<string, number> = {}; // Legacy support for single series
  @Input() multiData: Record<string, Record<string, number>> = {}; // For overlays
  @Input() title: string = 'Metric';
  @Input() seriesName: string = 'Value';
  @Input() color: string = '#38bdf8';
  @Input() yAxisTitle: string = 'Ratio';

  @ViewChild('chartContainer') chartContainer!: ElementRef;

  private chart?: Highcharts.Chart;

  // Modern institutional colors for multiple series
  private colors = ['#38bdf8', '#f87171', '#22c55e', '#fbbf24', '#818cf8', '#a78bfa', '#f472b6'];

  constructor(@Inject(LOCALE_ID) private locale: string) { }

  ngAfterViewInit(): void {
    this.refresh();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['data'] || changes['multiData'] || changes['title']) {
      this.refresh();
    }
  }

  private refresh() {
    if ((this.data && Object.keys(this.data).length > 0) || (this.multiData && Object.keys(this.multiData).length > 0)) {
      this.updateChart();
    }
  }

  private updateChart() {
    if (!this.chartContainer) return;

    const series: any[] = [];

    if (this.multiData && Object.keys(this.multiData).length > 0) {
      Object.entries(this.multiData).forEach(([name, data], idx) => {
        const seriesData = Object.entries(data)
          .map(([date, val]) => [new Date(date).getTime(), val])
          .sort((a, b) => a[0] - b[0]);

        series.push({
          name,
          data: seriesData,
          color: this.colors[idx % this.colors.length],
          lineWidth: 2,
          type: 'line'
        });
      });
    } else if (this.data && Object.keys(this.data).length > 0) {
      const seriesData = Object.entries(this.data)
        .map(([date, val]) => [new Date(date).getTime(), val])
        .sort((a, b) => a[0] - b[0]);

      series.push({
        name: this.seriesName,
        data: seriesData,
        color: this.color,
        lineWidth: 2,
        type: 'line'
      });
    }

    if (!series.length) return;

    if (!this.chart) {
      this.chart = Highcharts.chart(this.chartContainer.nativeElement, {
        chart: {
          type: 'line',
          backgroundColor: 'transparent',
          height: 300
        },
        title: { text: '' },
        xAxis: {
          type: 'datetime',
          gridLineColor: '#334155',
          labels: { style: { color: '#94a3b8' } }
        },
        yAxis: {
          title: { text: this.yAxisTitle, style: { color: '#94a3b8' } },
          gridLineColor: '#334155',
          labels: { style: { color: '#94a3b8' } }
        },
        legend: {
          enabled: series.length > 1,
          itemStyle: { color: '#94a3b8' },
          itemHoverStyle: { color: '#f8fafc' }
        },
        credits: { enabled: false },
        tooltip: {
          shared: true,
          backgroundColor: '#0f172a',
          style: { color: '#f8fafc' },
          borderWidth: 1,
          borderColor: '#334155',
          valueDecimals: 2
        },
        series: series as any
      });
    } else {
      // Clear existing series and add new ones
      while (this.chart.series.length > 0) {
        this.chart.series[0].remove(false);
      }
      series.forEach(s => this.chart?.addSeries(s, false));
      this.chart.redraw();
    }
  }
}
