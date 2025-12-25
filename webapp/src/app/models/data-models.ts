export interface DataPackage {
  version: string;
  exported_at: string;
  backtest: BacktestData;
  tda: TdaData | null;
  metadata: any;
}

export interface BacktestData {
  strategies: string[];
  benchmark: string;
  prices: PriceRecord[];
  stats: Record<string, Record<string, number>>;
  weights: Record<string, WeightRecord[]>;
  drawdown_events: Record<string, DrawdownEvent[]>;
  risk_attribution: Record<string, Record<string, number>>;
  return_attribution: Record<string, any[]>;
  sector_returns: Record<string, any[]>;
  sector_risk: Record<string, Record<string, number>>;
  sector_maps: Record<string, Record<string, string>>;
  correlation_matrix: Record<string, Record<string, number>>;
  monthly_returns: Record<string, Record<string, Record<string, number>>>;
  rolling_stats: Record<string, Record<string, Record<string, number>>>;
  return_distribution: Record<string, { values: number[], bins: number[] }>;
  capture_ratios: Record<string, { upside_capture: number, downside_capture: number }>;
}

export interface DrawdownEvent {
  date: string;
  magnitude: number;
  duration: number;
  recovery: number;
}

export interface PriceRecord {
  index: string;
  [strategy: string]: number | string;
}

export interface WeightRecord {
  index: string;
  [ticker: string]: number | string;
}

export interface TdaData {
  windows: (TdaWindow | null)[];
  trends: {
    dates: string[];
    betti: number[][];
    euler: number[];
    avg_corr: number[];
  };
}

export interface Landscape {
  hom_deg: number;
  start: number;
  stop: number;
  values: number[][]; // (depth, num_steps)
}

export interface TdaWindow {
  date: string;
  avg_corr: number;
  chi: number;
  betti: number[];
  coords: number[][]; // (N, 3)
  dgms: number[][][]; // [dim][hole_idx][birth, death]
  landscapes: Landscape[];
  edges: number[][];
  tickers: string[];
}
