#!/usr/bin/env python3
"""
Risk Contribution Analyzer
Shows which stocks contribute most to portfolio volatility (MCTR analysis)
"""
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from src.data import DataEngine
from src.analytics import AnalyticsEngine


def load_portfolio_weights(config_path: str) -> pd.Series:
    """Load portfolio weights from YAML config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    weights = {}
    portfolio = config.get('portfolio', {})
    
    # Fixed weights
    fixed = portfolio.get('fixed_weights', {})
    if fixed:
        weights.update(fixed)
    
    return pd.Series(weights)


def analyze_risk_contribution(
    weights: pd.Series,
    start_date: str = "2024-01-01",
    end_date: str = None
) -> pd.DataFrame:
    """
    Calculate risk contribution for each holding.
    
    Returns DataFrame with:
    - weight: Portfolio weight
    - vol: Individual asset volatility (annualized)
    - mctr: Marginal Contribution to Total Risk
    - pct_ctr: Percentage Contribution to Total Risk
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Fetch price data for holdings
    tickers = list(weights.index)
    
    # Use DataEngine to fetch data
    data_engine = DataEngine()
    prices = data_engine.get_historical_data(tickers, start_date, end_date)
    
    # Align weights with available data
    valid_tickers = [t for t in weights.index if t in prices.columns]
    weights = weights[valid_tickers]
    weights = weights / weights.sum()  # Renormalize
    prices = prices[valid_tickers]
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Individual volatilities (annualized)
    individual_vol = returns.std() * np.sqrt(252)
    
    # Portfolio volatility
    cov_matrix = returns.cov() * 252  # Annualized covariance
    port_var = weights @ cov_matrix @ weights
    port_vol = np.sqrt(port_var)
    
    # Marginal Contribution to Risk (MCTR)
    # MCTR_i = (Cov * w)_i / sigma_p
    mctr = (cov_matrix @ weights) / port_vol
    
    # Percentage Contribution to Risk
    # PCR_i = w_i * MCTR_i / sigma_p
    pct_ctr = (weights * mctr) / port_vol
    
    # Build result DataFrame
    result = pd.DataFrame({
        'weight': weights,
        'vol': individual_vol,
        'mctr': mctr,
        'pct_ctr': pct_ctr
    })
    
    result = result.sort_values('pct_ctr', ascending=False)
    
    return result, port_vol


def plot_risk_contribution(result: pd.DataFrame, port_vol: float, title: str = ""):
    """Create visualization of risk contribution breakdown."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Top left: Percentage Contribution to Risk (bar chart)
    ax1 = axes[0, 0]
    colors = ['#e74c3c' if x > result['pct_ctr'].mean() else '#3498db' 
              for x in result['pct_ctr']]
    result['pct_ctr'].plot(kind='barh', ax=ax1, color=colors)
    ax1.set_xlabel('% Contribution to Total Risk')
    ax1.set_title('Risk Contribution by Holding\n(Red = Above Average)')
    ax1.axvline(result['pct_ctr'].mean(), color='black', linestyle='--', alpha=0.5)
    
    # 2. Top right: Weight vs Risk Contribution scatter
    ax2 = axes[0, 1]
    ax2.scatter(result['weight'] * 100, result['pct_ctr'] * 100, 
                s=result['vol'] * 500, alpha=0.6, c='#2ecc71')
    for ticker in result.index:
        ax2.annotate(ticker, 
                    (result.loc[ticker, 'weight'] * 100, 
                     result.loc[ticker, 'pct_ctr'] * 100),
                    fontsize=8)
    ax2.plot([0, result['weight'].max() * 100], 
             [0, result['weight'].max() * 100], 
             'k--', alpha=0.3, label='1:1 line')
    ax2.set_xlabel('Portfolio Weight (%)')
    ax2.set_ylabel('Risk Contribution (%)')
    ax2.set_title('Weight vs Risk Contribution\n(Size = Volatility)')
    ax2.legend()
    
    # 3. Bottom left: Individual volatility bar
    ax3 = axes[1, 0]
    result_sorted = result.sort_values('vol', ascending=True)
    result_sorted['vol'].plot(kind='barh', ax=ax3, color='#9b59b6')
    ax3.set_xlabel('Annualized Volatility')
    ax3.set_title('Individual Asset Volatility')
    ax3.axvline(port_vol, color='red', linestyle='-', linewidth=2, 
                label=f'Portfolio Vol: {port_vol:.1%}')
    ax3.legend()
    
    # 4. Bottom right: Top 10 risk contributors pie chart
    ax4 = axes[1, 1]
    top10 = result.head(10)['pct_ctr']
    if len(result) > 10:
        other = result.iloc[10:]['pct_ctr'].sum()
        top10 = pd.concat([top10, pd.Series({'Other': other})])
    top10.plot(kind='pie', ax=ax4, autopct='%.1f%%', startangle=90)
    ax4.set_ylabel('')
    ax4.set_title('Top 10 Risk Contributors')
    
    plt.suptitle(f'{title}\nPortfolio Volatility: {port_vol:.1%}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze portfolio risk contribution')
    parser.add_argument('config', help='Path to portfolio YAML config')
    parser.add_argument('--start', default='2024-01-01', help='Start date for analysis')
    parser.add_argument('--end', default=None, help='End date for analysis')
    args = parser.parse_args()
    
    print(f"Loading portfolio from {args.config}...")
    weights = load_portfolio_weights(args.config)
    
    print(f"Analyzing risk contribution ({args.start} to {args.end or 'now'})...")
    result, port_vol = analyze_risk_contribution(weights, args.start, args.end)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RISK CONTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    print(f"Portfolio Volatility: {port_vol:.2%}")
    print(f"\nTop 10 Risk Contributors:")
    print("-" * 40)
    for i, (ticker, row) in enumerate(result.head(10).iterrows()):
        print(f"{i+1:2}. {ticker:6} | Weight: {row['weight']:6.1%} | "
              f"Vol: {row['vol']:6.1%} | Risk Ctr: {row['pct_ctr']:6.1%}")
    
    print(f"\n{'='*60}")
    
    # Plot
    plot_risk_contribution(result, port_vol, title=args.config)


if __name__ == '__main__':
    main()
