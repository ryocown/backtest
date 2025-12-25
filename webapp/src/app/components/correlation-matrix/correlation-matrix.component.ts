import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-correlation-matrix',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './correlation-matrix.component.html',
  styleUrl: './correlation-matrix.component.css'
})
export class CorrelationMatrixComponent implements OnChanges {
  @Input() correlationData: Record<string, Record<string, number>> = {};
  @Input() title: string = "Strategy Correlation Matrix";

  strategies: string[] = [];

  ngOnChanges(): void {
    if (this.correlationData) {
      this.strategies = Object.keys(this.correlationData);
    }
  }

  getCorr(s1: string, s2: string): number {
    return this.correlationData[s1]?.[s2] ?? 1;
  }

  getColor(val: number): string {
    // scale from -1 (blue) to 1 (red)
    const normalized = (val + 1) / 2;
    if (val > 0) {
      return `rgba(239, 68, 68, ${val * 0.8})`; // Red for positive
    } else {
      return `rgba(59, 130, 246, ${Math.abs(val) * 0.8})`; // Blue for negative
    }
  }
}
