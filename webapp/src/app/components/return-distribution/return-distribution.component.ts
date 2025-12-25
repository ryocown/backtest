import { Component, Input, OnChanges, SimpleChanges, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import * as Highcharts from 'highcharts';

@Component({
  selector: 'app-return-distribution',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './return-distribution.component.html',
  styleUrl: './return-distribution.component.css'
})
export class ReturnDistributionComponent implements OnChanges, AfterViewInit {
  @Input() distData: { values: number[], bins: number[] } | null = null;
  @Input() title: string = 'Return Distribution';

  @ViewChild('chartContainer') chartContainer!: ElementRef;

  ngAfterViewInit(): void {
    if (this.distData) {
      this.renderChart();
    }
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (this.distData && changes['distData']) {
      this.renderChart();
    }
  }

  private renderChart() {
    const dist = this.distData;
    if (!this.chartContainer || !dist) return;

    // Convert bins to centers for Highcharts column chart
    const data = [];
    for (let i = 0; i < dist.values.length; i++) {
      const center = (dist.bins[i] + dist.bins[i + 1]) / 2;
      data.push([center * 100, dist.values[i]]);
    }

    Highcharts.chart(this.chartContainer.nativeElement, {
      chart: {
        type: 'column',
        backgroundColor: 'transparent',
        height: 350
      },
      title: { text: '' },
      xAxis: {
        title: { text: 'Daily Return (%)', style: { color: '#94a3b8' } },
        gridLineColor: '#334155',
        labels: { style: { color: '#94a3b8' } },
        plotLines: [{
          value: 0,
          color: '#f8fafc',
          width: 2,
          zIndex: 5
        }]
      },
      yAxis: {
        title: { text: 'Frequency', style: { color: '#94a3b8' } },
        gridLineColor: '#334155',
        labels: { style: { color: '#94a3b8' } }
      },
      legend: { enabled: false },
      credits: { enabled: false },
      tooltip: {
        headerFormat: 'Return: <b>{point.key:.2f}%</b><br>',
        pointFormat: 'Count: <b>{point.y}</b>'
      },
      plotOptions: {
        column: {
          borderWidth: 0,
          pointPadding: 0,
          groupPadding: 0,
          shadow: false,
          color: '#38bdf8'
        }
      },
      series: [{
        name: 'Returns',
        data: data
      } as any]
    });
  }
}
