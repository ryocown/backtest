import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

@Component({
  selector: 'app-configuration',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './configuration.component.html',
  styleUrl: './configuration.component.css'
})
export class ConfigurationComponent {
  config = {
    portfolio: 'portfolios/30_quarterly_rebalance.yaml',
    startDate: '2020-01-01',
    endDate: '2020-12-31',
    benchmarks: 'SPY,QQQ,VTI',
    enableTda: true,
    tdaStart: '2020-02-18',
    tdaEnd: '2020-04-06',
    tdaWindow: 6
  };

  constructor(private router: Router) { }

  get generatedCommand(): string {
    let cmd = `./venv/bin/python3 main.py ${this.config.portfolio} `;
    cmd += `--start_date "${this.config.startDate}" --end_date "${this.config.endDate}" `;
    cmd += `--benchmarks "${this.config.benchmarks}" --web-export `;

    if (this.config.enableTda) {
      cmd += `--tda --tda-start "${this.config.tdaStart}" --tda-end "${this.config.tdaEnd}" --tda-window ${this.config.tdaWindow} `;
    }

    cmd += `--no-graph`;
    return cmd;
  }

  copyCommand() {
    navigator.clipboard.writeText(this.generatedCommand).then(() => {
      alert('Command copied to clipboard!');
    });
  }

  goToDashboard() {
    this.router.navigate(['/']);
  }
}
