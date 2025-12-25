import { Component, Input, OnChanges } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-monthly-heatmap',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './monthly-heatmap.component.html',
  styleUrl: './monthly-heatmap.component.css'
})
export class MonthlyHeatmapComponent implements OnChanges {
  @Input() monthlyData: Record<string, Record<string, number>> = {};
  @Input() title: string = "Monthly Returns Heatmap";

  years: string[] = [];
  months: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
  monthNames: string[] = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  ngOnChanges(): void {
    if (this.monthlyData) {
      this.years = Object.keys(this.monthlyData).sort((a, b) => b.localeCompare(a));
    }
  }

  getReturn(year: string, month: number): number | null {
    const val = this.monthlyData[year]?.[month];
    return val !== undefined ? val : null;
  }

  getAnnualReturn(year: string): number {
    const monthVals = Object.values(this.monthlyData[year] || {});
    if (monthVals.length === 0) return 0;
    return monthVals.reduce((acc, val) => (acc * (1 + val)), 1) - 1;
  }

  getColor(val: number | null): string {
    if (val === null) return 'transparent';

    // RdYlGn color scale logic
    const pct = val * 100;
    if (pct > 0) {
      const alpha = Math.min(Math.abs(pct) / 5, 0.8);
      return `rgba(34, 197, 94, ${alpha})`; // Green
    } else if (pct < 0) {
      const alpha = Math.min(Math.abs(pct) / 5, 0.8);
      return `rgba(239, 68, 68, ${alpha})`; // Red
    }
    return '#1e293b';
  }
}
