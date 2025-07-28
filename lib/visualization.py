"""
Visualization module for the backtesting system.
Handles plotting, heatmaps, and visual reporting.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import os
from pathlib import Path
import logging
from .config import OUTPUT_CONFIG
from .backtester import BacktestResult

logger = logging.getLogger(__name__)

class Visualizer:
    """Main visualization class"""
    
    def __init__(self, config=None):
        self.config = config or OUTPUT_CONFIG
        
    def plot_equity_curves(self, results: List[Tuple[str, BacktestResult]], 
                          output_path: str, title: str = "Equity Curves",
                          split_point: pd.Timestamp = None):
        """
        Plot equity curves for multiple strategies
        
        Args:
            results: List of (name, BacktestResult) tuples
            output_path: Output file path
            title: Plot title
            split_point: Optional split point to show backtest/forward division
        """
        plt.figure(figsize=self.config.figure_size)
        
        for name, result in results:
            plt.plot(result.equity.index, result.equity.values, 
                    label=name, linewidth=1.5)
        
        if split_point:
            plt.axvline(split_point, color='gray', linestyle='--', 
                       alpha=0.7, label='Train/Test Split')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved equity curves plot to {output_path}")
    
    def plot_strategy_performance(self, result: BacktestResult, price: pd.Series,
                                 output_path: str, strategy_name: str = "Strategy"):
        """
        Plot detailed strategy performance with price, positions, and equity
        
        Args:
            result: BacktestResult object
            price: Price series
            output_path: Output file path
            strategy_name: Strategy name for title
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # Price and positions
        axes[0].plot(price.index, price.values, 'b-', linewidth=1, label='Price')
        
        # Mark long and short positions
        long_positions = result.positions[result.positions > 0]
        short_positions = result.positions[result.positions < 0]
        
        if not long_positions.empty:
            axes[0].scatter(long_positions.index, price.reindex(long_positions.index),
                          color='green', marker='^', s=30, alpha=0.7, label='Long Entry')
        
        if not short_positions.empty:
            axes[0].scatter(short_positions.index, price.reindex(short_positions.index),
                          color='red', marker='v', s=30, alpha=0.7, label='Short Entry')
        
        axes[0].set_title(f'{strategy_name} - Price and Positions')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Positions over time
        axes[1].plot(result.positions.index, result.positions.values, 'g-', linewidth=1)
        axes[1].fill_between(result.positions.index, 0, result.positions.values, 
                           alpha=0.3, color='green')
        axes[1].set_title('Position Size Over Time')
        axes[1].set_ylabel('Position')
        axes[1].grid(True, alpha=0.3)
        
        # Equity curve
        axes[2].plot(result.equity.index, result.equity.values, 'b-', linewidth=2)
        axes[2].fill_between(result.equity.index, 0, result.equity.values, 
                           alpha=0.3, color='blue')
        axes[2].set_title('Equity Curve')
        axes[2].set_ylabel('Cumulative Return')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved strategy performance plot to {output_path}")
    
    def plot_parameter_heatmaps(self, results_df: pd.DataFrame, model: str, 
                               strategy: str, output_dir: str, 
                               factor_name: str = "factor"):
        """
        Generate parameter heatmaps for model/strategy combinations
        
        Args:
            results_df: DataFrame with backtest results
            model: Model name
            strategy: Strategy name
            output_dir: Output directory
            factor_name: Factor name for filename
        """
        # Filter for specific model/strategy combination
        subset = results_df[
            (results_df['model'] == model) & 
            (results_df['strategy'] == strategy)
        ].copy()
        
        if subset.empty:
            logger.warning(f"No results found for {model}/{strategy}")
            return
        
        # Process windows parameter
        subset['win_tup'] = subset['windows'].apply(
            lambda x: x if isinstance(x, (tuple, list)) else (x,)
        )
        
        # Determine number of window dimensions
        win_dims = len(subset['win_tup'].iloc[0])
        
        # Create separate columns for each window dimension
        for i in range(win_dims):
            subset[f'w{i+1}'] = subset['win_tup'].apply(lambda t: t[i])
        
        # Ensure threshold column
        subset['thr'] = subset['threshold'].astype(float)
        
        # Collect dimension names
        dims = [f'w{i+1}' for i in range(win_dims)] + ['thr']
        total_dims = len(dims)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        def _plot_heatmap(dim_x: str, dim_y: str, title_suffix: str, fname_suffix: str):
            """Helper function to plot individual heatmaps"""
            pivot = subset.pivot_table(
                index=dim_y, columns=dim_x, values='sharpe_ratio', aggfunc='mean'
            )
            
            plt.figure(figsize=self.config.figure_size)
            sns.heatmap(
                pivot.T,  # Transpose so x-axis is dim_x
                annot=True, fmt=".2f",
                cmap="RdYlGn", center=0,
                linewidths=0.5, linecolor='white',
                cbar_kws={"label": "Sharpe Ratio"}
            )
            
            plt.title(f"{model} | {strategy} — {title_suffix}")
            plt.xlabel(dim_x)
            plt.ylabel(dim_y)
            plt.tight_layout()
            
            filename = f"{factor_name}_{model}_{strategy}_{fname_suffix}.png"
            filepath = output_path / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved heatmap to {filepath}")
        
        # Generate heatmaps based on number of dimensions
        if total_dims == 1:
            # Only threshold - line plot
            dim = dims[0]
            plt.figure(figsize=self.config.figure_size)
            sns.lineplot(data=subset, x=dim, y='sharpe_ratio', marker='o')
            plt.axhline(0, color='gray', linestyle='--')
            plt.title(f"{model} | {strategy} — Sharpe vs {dim}")
            plt.xlabel(dim)
            plt.ylabel("Sharpe Ratio")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = f"{factor_name}_{model}_{strategy}_sharpe_vs_{dim}.png"
            filepath = output_path / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
        elif total_dims == 2:
            # Two dimensions
            dim_x, dim_y = dims
            _plot_heatmap(dim_x, dim_y, f"{dim_x} vs {dim_y}", f"heat_{dim_x}_vs_{dim_y}")
            
        elif total_dims == 3:
            # Three dimensions - plot all pairs
            w1, w2, thr = dims
            _plot_heatmap(w1, w2, f"{w1} vs {w2}", f"heat_{w1}_vs_{w2}")
            _plot_heatmap(w1, thr, f"{w1} vs {thr}", f"heat_{w1}_vs_{thr}")
            _plot_heatmap(w2, thr, f"{w2} vs {thr}", f"heat_{w2}_vs_{thr}")
    
    def plot_drawdown_analysis(self, result: BacktestResult, output_path: str):
        """
        Plot drawdown analysis
        
        Args:
            result: BacktestResult object
            output_path: Output file path
        """
        # Calculate drawdown
        running_max = result.equity.cummax()
        drawdown = result.equity - running_max
        drawdown_pct = (drawdown / running_max) * 100
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Equity curve with drawdown
        axes[0].plot(result.equity.index, result.equity.values, 'b-', linewidth=2, label='Equity')
        axes[0].plot(running_max.index, running_max.values, 'g--', alpha=0.7, label='Peak')
        axes[0].fill_between(result.equity.index, result.equity.values, running_max.values,
                           alpha=0.3, color='red', label='Drawdown')
        axes[0].set_title('Equity Curve and Drawdown')
        axes[0].set_ylabel('Cumulative Return')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown percentage
        axes[1].fill_between(drawdown_pct.index, 0, drawdown_pct.values, 
                           alpha=0.7, color='red')
        axes[1].plot(drawdown_pct.index, drawdown_pct.values, 'r-', linewidth=1)
        axes[1].set_title('Drawdown Percentage')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved drawdown analysis to {output_path}")
    
    def create_performance_dashboard(self, results: List[Tuple[str, BacktestResult]], 
                                   output_path: str):
        """
        Create an interactive performance dashboard using Plotly
        
        Args:
            results: List of (name, BacktestResult) tuples
            output_path: Output HTML file path
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Equity Curves', 'Sharpe Ratios', 'Max Drawdown', 'Trade Frequency'),
            specs=[[{"secondary_y": False}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Equity curves
        for name, result in results:
            fig.add_trace(
                go.Scatter(
                    x=result.equity.index,
                    y=result.equity.values,
                    mode='lines',
                    name=name,
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Extract metrics for bar charts
        names = [name for name, _ in results]
        sharpe_ratios = [result.sharpe_ratio for _, result in results]
        max_drawdowns = [result.max_drawdown_pct * 100 for _, result in results]
        trade_frequencies = [result.trade_per_interval * 100 for _, result in results]
        
        # Sharpe ratios
        fig.add_trace(
            go.Bar(x=names, y=sharpe_ratios, name='Sharpe Ratio', showlegend=False),
            row=1, col=2
        )
        
        # Max drawdown
        fig.add_trace(
            go.Bar(x=names, y=max_drawdowns, name='Max Drawdown (%)', showlegend=False),
            row=2, col=1
        )
        
        # Trade frequency vs Sharpe
        fig.add_trace(
            go.Scatter(
                x=trade_frequencies,
                y=sharpe_ratios,
                mode='markers+text',
                text=names,
                textposition="top center",
                name='Strategies',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Strategy Performance Dashboard",
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Max Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title_text="Trade Frequency (%)", row=2, col=2)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)
        
        # Save as HTML
        fig.write_html(output_path)
        logger.info(f"Saved interactive dashboard to {output_path}")
    
    def plot_correlation_matrix(self, results: List[Tuple[str, BacktestResult]], 
                               output_path: str):
        """
        Plot correlation matrix of strategy returns
        
        Args:
            results: List of (name, BacktestResult) tuples
            output_path: Output file path
        """
        # Create returns matrix
        returns_data = {}
        for name, result in results:
            returns_data[name] = result.pnl
        
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title('Strategy Returns Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved correlation matrix to {output_path}")
    
    def plot_risk_return_scatter(self, results: List[Tuple[str, BacktestResult]], 
                                output_path: str):
        """
        Plot risk-return scatter plot
        
        Args:
            results: List of (name, BacktestResult) tuples
            output_path: Output file path
        """
        names = []
        returns = []
        risks = []
        
        for name, result in results:
            names.append(name)
            returns.append(result.annualized_return * 100)
            risks.append(result.volatility * 100)
        
        plt.figure(figsize=self.config.figure_size)
        scatter = plt.scatter(risks, returns, s=100, alpha=0.7)
        
        for i, name in enumerate(names):
            plt.annotate(name, (risks[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Volatility (%)')
        plt.ylabel('Annualized Return (%)')
        plt.title('Risk-Return Profile')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved risk-return scatter to {output_path}")

def create_summary_table(results: List[Tuple[str, BacktestResult]]) -> pd.DataFrame:
    """
    Create a summary table of strategy performance metrics
    
    Args:
        results: List of (name, BacktestResult) tuples
        
    Returns:
        DataFrame with summary statistics
    """
    summary_data = []
    
    for name, result in results:
        summary_data.append({
            'Strategy': name,
            'Total Return (%)': result.total_return * 100,
            'Annualized Return (%)': result.annualized_return * 100,
            'Sharpe Ratio': result.sharpe_ratio,
            'Sortino Ratio': result.sortino_ratio,
            'Calmar Ratio': result.calmar_ratio,
            'Max Drawdown (%)': result.max_drawdown_pct * 100,
            'Volatility (%)': result.volatility * 100,
            'Win Rate (%)': result.win_rate * 100 if result.win_rate else 0,
            'Num Trades': result.num_trades,
            'Trade Frequency (%)': result.trade_per_interval * 100
        })
    
    return pd.DataFrame(summary_data)