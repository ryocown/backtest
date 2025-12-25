import { Component, Input, OnChanges, SimpleChanges, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import * as Highcharts from 'highcharts';
import * as TreemapModule from 'highcharts/modules/treemap';

// Initialize the treemap module robustly for Vite/Angular 18
const initTreemap = (TreemapModule as any).default || TreemapModule;
if (typeof initTreemap === 'function') {
  initTreemap(Highcharts);
}

@Component({
  selector: 'app-risk-attribution',
  standalone: true,
  imports: [CommonModule],
  template: '<div #chartContainer class="chart-container"></div>',
  styles: ['.chart-container { height: 400px; width: 100%; background: #1e293b; padding: 1.5rem; border-radius: 12px; border: 1px solid #334155; }']
})
export class RiskAttributionComponent implements OnChanges, AfterViewInit {
  @ViewChild('chartContainer') chartContainer!: ElementRef;
  @Input() riskData: { [ticker: string]: number } = {};
  @Input() sectorMap: { [ticker: string]: string } = {};
  @Input() title: string = 'Risk Attribution (MCTR)';

  private chart: Highcharts.Chart | null = null;

  ngAfterViewInit(): void {
    this.refresh();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if ((changes['riskData'] || changes['sectorMap']) && this.chartContainer) {
      this.refresh();
    }
  }

  private refresh() {
    if (this.riskData && Object.keys(this.riskData).length > 0) {
      this.initChart();
    }
  }

  private initChart() {
    if (!this.riskData || Object.keys(this.riskData).length === 0) return;

    const chartData: any[] = [];
    const sectors = new Set<string>();

    // 1. identify sectors and add them as parent nodes
    if (this.sectorMap && Object.keys(this.sectorMap).length > 0) {
      Object.entries(this.riskData).forEach(([ticker, risk]) => {
        const sector = this.sectorMap[ticker] || 'Other';
        sectors.add(sector);
      });

      Array.from(sectors).forEach(sector => {
        chartData.push({
          id: sector,
          name: sector,
          color: 'transparent' // Sectors are just containers
        });
      });

      // 2. Add individual tickers with parent links
      Object.entries(this.riskData).forEach(([ticker, risk]) => {
        const sector = this.sectorMap[ticker] || 'Other';
        chartData.push({
          name: ticker,
          parent: sector,
          value: Math.max(0, risk),
          colorValue: risk
        });
      });
    } else {
      // Flat list fallback
      Object.entries(this.riskData).forEach(([name, value]) => {
        chartData.push({
          name,
          value: Math.max(0, value),
          colorValue: value
        });
      });
    }

    const options: Highcharts.Options = {
      chart: {
        type: 'treemap',
        backgroundColor: 'transparent',
      },
      title: {
        text: this.title,
        style: { color: '#38bdf8', fontSize: '18px', fontWeight: 'bold' }
      },
      colorAxis: {
        minColor: '#1e293b',
        maxColor: '#ef4444'
      },
      series: [{
        type: 'treemap',
        layoutAlgorithm: 'squarified',
        allowTraversingTree: true,
        levels: [{
          level: 1,
          layoutAlgorithm: 'squarified',
          dataLabels: {
            enabled: true,
            align: 'left',
            verticalAlign: 'top',
            style: { fontSize: '15px', fontWeight: 'bold', color: '#f8fafc' }
          }
        }],
        data: chartData
      }],
      tooltip: {
        pointFormat: '<b>{point.name}</b>: {(point.value * 100).toFixed(2)}% of total risk'
      },
      credits: { enabled: false }
    };

    if (this.chart) {
      this.chart.destroy();
    }
    this.chart = Highcharts.chart(this.chartContainer.nativeElement, options);
  }
}
