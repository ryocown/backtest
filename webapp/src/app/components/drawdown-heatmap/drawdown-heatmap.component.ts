import { Component, Input, OnChanges, SimpleChanges, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import * as d3 from 'd3';
import { DrawdownEvent } from '../../models/data-models';

@Component({
  selector: 'app-drawdown-heatmap',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './drawdown-heatmap.component.html',
  styleUrl: './drawdown-heatmap.component.css'
})
export class DrawdownHeatmapComponent implements OnChanges, AfterViewInit {
  @ViewChild('heatmapContainer') container!: ElementRef;
  @Input() events: DrawdownEvent[] = [];
  @Input() title: string = 'Drawdown Recovery Matrix';

  private svg: any;
  private margin = { top: 40, right: 40, bottom: 60, left: 60 };
  private width = 600;
  private height = 400;

  ngAfterViewInit(): void {
    if (this.events.length > 0) {
      this.renderHeatmap();
    }
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['events'] && this.events) {
      this.renderHeatmap();
    }
  }

  private renderHeatmap() {
    if (!this.container || !this.events || this.events.length === 0) return;

    const element = this.container.nativeElement;
    d3.select(element).selectAll('svg').remove();

    const parentWidth = element.getBoundingClientRect().width || this.width;
    const width = parentWidth - this.margin.left - this.margin.right;
    const height = this.height - this.margin.top - this.margin.bottom;

    this.svg = d3.select(element)
      .append('svg')
      .attr('width', width + this.margin.left + this.margin.right)
      .attr('height', height + this.margin.top + this.margin.bottom)
      .append('g')
      .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

    // Scales
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(this.events, d => d.magnitude * 100) || 50])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(this.events, d => d.duration) || 12])
      .range([height, 0]);

    const colorScale = d3.scaleSequential(d3.interpolateOrRd)
      .domain([0, d3.max(this.events, d => d.recovery * 100) || 100]);

    // Axes
    this.svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).ticks(10))
      .append('text')
      .attr('x', width / 2)
      .attr('y', 40)
      .attr('fill', '#94a3b8')
      .style('text-anchor', 'middle')
      .text('Magnitude of Drawdown (%)');

    this.svg.append('g')
      .call(d3.axisLeft(yScale).ticks(10))
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -45)
      .attr('x', -height / 2)
      .attr('fill', '#94a3b8')
      .style('text-anchor', 'middle')
      .text('Duration (Months)');

    // Tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'd3-tooltip')
      .style('position', 'absolute')
      .style('z-index', '10')
      .style('visibility', 'hidden')
      .style('background', '#1e293b')
      .style('color', '#f8fafc')
      .style('padding', '8px')
      .style('border-radius', '4px')
      .style('border', '1px solid #334155')
      .style('font-size', '12px');

    // Points
    this.svg.selectAll('.dot')
      .data(this.events)
      .enter()
      .append('circle')
      .attr('class', 'dot')
      .attr('cx', (d: any) => xScale(d.magnitude * 100))
      .attr('cy', (d: any) => yScale(d.duration))
      .attr('r', 8)
      .attr('fill', (d: any) => colorScale(d.recovery * 100))
      .attr('stroke', '#fff')
      .attr('stroke-width', 1)
      .style('opacity', 0.8)
      .on('mouseover', (event: any, d: any) => {
        d3.select(event.currentTarget).attr('r', 12).style('opacity', 1);
        tooltip.html(`
          <strong>Date:</strong> ${d.date}<br/>
          <strong>Magnitude:</strong> ${(d.magnitude * 100).toFixed(2)}%<br/>
          <strong>Duration:</strong> ${d.duration.toFixed(1)} months<br/>
          <strong>Recovery Required:</strong> ${(d.recovery * 100).toFixed(2)}%
        `).style('visibility', 'visible');
      })
      .on('mousemove', (event: any) => {
        tooltip.style('top', (event.pageY - 10) + 'px').style('left', (event.pageX + 10) + 'px');
      })
      .on('mouseout', (event: any) => {
        d3.select(event.currentTarget).attr('r', 8).style('opacity', 0.8);
        tooltip.style('visibility', 'hidden');
      });

    // Grid lines
    this.svg.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale).ticks(10).tickSize(-height).tickFormat(() => ''))
      .style('stroke-opacity', 0.1);

    this.svg.append('g')
      .attr('class', 'grid')
      .call(d3.axisLeft(yScale).ticks(10).tickSize(-width).tickFormat(() => ''))
      .style('stroke-opacity', 0.1);
  }
}
