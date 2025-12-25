import { Component, Input, OnChanges, SimpleChanges, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import * as Highcharts from 'highcharts';

@Component({
  selector: 'app-return-attribution',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './return-attribution.component.html',
  styleUrl: './return-attribution.component.css'
})
export class ReturnAttributionComponent implements OnChanges, AfterViewInit {
  @Input() attributionData: any[] = [];
  @Input() sectorData: any[] = [];
  @Input() title: string = "Cumulative Return Attribution";

  @ViewChild('chartContainer') chartContainer!: ElementRef;

  private chart?: Highcharts.Chart;
  showSectors: boolean = false;

  ngAfterViewInit(): void {
    this.refresh();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (this.attributionData && (changes['attributionData'] || changes['sectorData'] || changes['title'])) {
      this.refresh();
    }
  }

  toggleView() {
    this.showSectors = !this.showSectors;
    this.refresh();
  }

  private refresh() {
    if ((this.showSectors && this.sectorData?.length > 0) || (this.attributionData?.length > 0)) {
      this.renderChart();
    }
  }

  private renderChart() {
    const activeData = this.showSectors ? this.sectorData : this.attributionData;
    if (!activeData || activeData.length === 0) return;

    const items = Object.keys(activeData[0]).filter(k => k !== 'index');
    const series: any[] = items.map(item => ({
      name: item,
      data: activeData.map(d => [new Date(d.index).getTime(), d[item] * 100]),
      type: 'area'
    }));

    if (!this.chartContainer) return;

    Highcharts.chart(this.chartContainer.nativeElement, {
      chart: {
        type: 'area',
        backgroundColor: 'transparent',
        height: 400
      },
      title: {
        text: this.title + (this.showSectors ? ' (By Sector)' : ' (By Asset)'),
        style: { color: '#e2e8f0', fontSize: '14px' }
      },
      xAxis: {
        type: 'datetime',
        gridLineColor: '#334155',
        labels: { style: { color: '#94a3b8' } }
      },
      yAxis: {
        title: { text: 'Cumulative Contribution (%)', style: { color: '#94a3b8' } },
        gridLineColor: '#334155',
        labels: { style: { color: '#94a3b8' } }
      },
      legend: {
        itemStyle: { color: '#94a3b8' },
        itemHoverStyle: { color: '#f8fafc' }
      },
      credits: { enabled: false },
      plotOptions: {
        area: {
          stacking: 'normal',
          lineWidth: 1,
          marker: { enabled: false }
        }
      },
      tooltip: {
        shared: true,
        backgroundColor: '#0f172a',
        style: { color: '#f8fafc' },
        valueDecimals: 2,
        valueSuffix: '%'
      },
      series: series
    });
  }
}
