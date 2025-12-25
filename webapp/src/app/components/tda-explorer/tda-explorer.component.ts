import { Component, Input, OnChanges, SimpleChanges, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import * as Plotly from 'plotly.js-dist-min';
import { TdaData, TdaWindow } from '../../models/data-models';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-tda-explorer',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './tda-explorer.component.html',
  styleUrl: './tda-explorer.component.css'
})
export class TdaExplorerComponent implements OnChanges, AfterViewInit {
  @ViewChild('projectionPlot') projectionPlot!: ElementRef;
  @ViewChild('diagramPlot') diagramPlot!: ElementRef;
  @ViewChild('landscapePlot') landscapePlot!: ElementRef;
  @ViewChild('barcodePlot') barcodePlot!: ElementRef;
  @Input() data: TdaData | null = null;

  currentIndex: number = 0;
  currentWindow: TdaWindow | null = null;

  ngAfterViewInit(): void {
    if (this.data) {
      this.updateWindow();
    }
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['data'] && this.data) {
      // Find the first non-null window
      const firstValid = this.data.windows.findIndex(w => w !== null);
      this.currentIndex = firstValid >= 0 ? firstValid : 0;
      this.updateWindow();
    }
  }

  onSliderChange(event: any) {
    this.currentIndex = parseInt(event.target.value);
    this.updateWindow();
  }

  private updateWindow() {
    if (!this.data || !this.data.windows[this.currentIndex]) {
      this.currentWindow = null;
      return;
    }

    this.currentWindow = this.data.windows[this.currentIndex] as TdaWindow;
    this.renderPlots();
  }

  private renderPlots() {
    if (!this.currentWindow) return;

    // 1. 3D Projection Plot
    const coords = this.currentWindow.coords;
    const tickers = this.currentWindow.tickers;

    const trace3d: any = {
      x: coords.map(c => c[0]),
      y: coords.map(c => c[1]),
      z: coords.map(c => c[2]),
      mode: 'markers+text',
      type: 'scatter3d',
      text: tickers,
      marker: {
        size: 5,
        color: this.currentWindow.avg_corr,
        colorscale: 'Viridis',
        opacity: 0.8
      }
    };

    const traces: any[] = [trace3d];

    // Add Edges (Simplicial Complex connections)
    if (this.currentWindow.edges) {
      this.currentWindow.edges.forEach(edge => {
        const i = edge[0];
        const j = edge[1];
        traces.push({
          x: [coords[i][0], coords[j][0]],
          y: [coords[i][1], coords[j][1]],
          z: [coords[i][2], coords[j][2]],
          mode: 'lines',
          type: 'scatter3d',
          line: { color: 'rgba(56, 189, 248, 0.2)', width: 1 },
          showlegend: false
        });
      });
    }

    const layout3d: any = {
      title: `Market Manifold - ${this.currentWindow.date}`,
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#f8fafc' },
      margin: { l: 0, r: 0, b: 0, t: 30 },
      scene: {
        xaxis: { title: 'D1', gridcolor: '#334155' },
        yaxis: { title: 'D2', gridcolor: '#334155' },
        zaxis: { title: 'D3', gridcolor: '#334155' }
      }
    };

    Plotly.newPlot(this.projectionPlot.nativeElement, traces, layout3d);

    // 2. Persistence Diagram
    const dgms = this.currentWindow.dgms;
    const diagramTraces: any[] = [];

    // Add H0, H1 traces
    dgms.forEach((dgm, dim) => {
      diagramTraces.push({
        x: dgm.map(p => p[0]),
        y: dgm.map(p => p[1]),
        mode: 'markers',
        name: `H${dim}`,
        marker: { size: 8 }
      });
    });

    // Add diagonal
    const maxVal = Math.max(...dgms.flat().flat()) * 1.1;
    diagramTraces.push({
      x: [0, maxVal],
      y: [0, maxVal],
      mode: 'lines',
      name: 'Diagonal',
      line: { dash: 'dash', color: '#64748b' },
      showlegend: false
    });

    const layoutDiagram: any = {
      title: 'Persistence Diagram',
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#f8fafc' },
      xaxis: { title: 'Birth', gridcolor: '#334155' },
      yaxis: { title: 'Death', gridcolor: '#334155' },
      margin: { l: 50, r: 20, b: 50, t: 50 }
    };

    Plotly.newPlot(this.diagramPlot.nativeElement, diagramTraces, layoutDiagram);

    // 3. Persistence Landscapes
    this.renderLandscapes();

    // 4. Persistence Barcodes
    this.renderBarcodes();
  }

  private renderLandscapes() {
    if (!this.currentWindow || !this.currentWindow.landscapes) return;

    const traces: any[] = [];
    const colors = ['#38bdf8', '#f87171', '#fbbf24']; // H0, H1 colors

    this.currentWindow.landscapes.forEach((ls, i) => {
      const x = Array.from({ length: ls.values[0].length }, (_, idx) =>
        ls.start + (idx * (ls.stop - ls.start) / (ls.values[0].length - 1)));

      ls.values.forEach((depthVals, depth) => {
        traces.push({
          x: x,
          y: depthVals,
          mode: 'lines',
          name: `H${ls.hom_deg} (d=${depth})`,
          line: {
            color: colors[ls.hom_deg % colors.length],
            width: 2 - (depth * 0.5),
            shape: 'hv'
          },
          legendgroup: `H${ls.hom_deg}`
        });
      });
    });

    const layout: any = {
      title: 'Persistence Landscapes',
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#f8fafc' },
      xaxis: { title: 'Parameter (ε)', gridcolor: '#334155' },
      yaxis: { title: 'Amplitude', gridcolor: '#334155' },
      margin: { l: 50, r: 20, b: 50, t: 50 }
    };

    Plotly.newPlot(this.landscapePlot.nativeElement, traces, layout);
  }

  private renderBarcodes() {
    if (!this.currentWindow || !this.currentWindow.dgms) return;

    const traces: any[] = [];
    const colors = ['#38bdf8', '#f87171', '#fbbf24'];
    let barIndex = 0;

    this.currentWindow.dgms.forEach((dgm, dim) => {
      // Sort intervals by length for better visualization
      const sortedDgm = [...dgm].sort((a, b) => (b[1] - b[0]) - (a[1] - a[0]));

      sortedDgm.forEach((p, i) => {
        const death = p[1] === Infinity ? 1.5 * Math.max(...this.currentWindow!.dgms.flat().map(pair => pair[1] === Infinity ? 0 : pair[1])) : p[1];

        traces.push({
          x: [p[0], death],
          y: [barIndex, barIndex],
          mode: 'lines',
          name: `H${dim}`,
          line: { color: colors[dim % colors.length], width: 4 },
          showlegend: i === 0,
          legendgroup: `H${dim}`
        });
        barIndex++;
      });
    });

    const layout: any = {
      title: 'Persistence Barcodes',
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#f8fafc' },
      xaxis: { title: 'Parameter (ε)', gridcolor: '#334155' },
      yaxis: { title: 'Index', showticklabels: false, gridcolor: '#334155' },
      margin: { l: 50, r: 20, b: 50, t: 50 }
    };

    Plotly.newPlot(this.barcodePlot.nativeElement, traces, layout);
  }
}
