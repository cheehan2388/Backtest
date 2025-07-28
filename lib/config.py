"""
Configuration module for the backtesting system.
Contains all system-wide settings, parameter ranges, and default values.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    # Data settings
    date_column: str = 'datetime'
    price_column: str = 'close'
    
    # Time splits
    backtest_ratio: float = 0.6
    validation_ratio: float = 0.2
    forward_ratio: float = 0.2
    
    # Trading settings
    fee: float = 0.0006
    sr_multiplier: float = 24  # For hourly data
    
    # Filter settings
    min_sharpe: float = 2.0
    max_drawdown: float = -0.5
    min_trade_ratio: float = 0.03
    
    # Forward test consistency
    forward_consistency_range: Tuple[float, float] = (0.9, 1.1)
    
    # Statistical significance
    permutation_tests: int = 500
    significance_level: float = 0.05
    permutation_seed: int = 42

@dataclass 
class ModelParameterRanges:
    """Parameter ranges for different models"""
    
    # Z-score parameters
    zscore_windows: np.ndarray = field(default_factory=lambda: np.arange(20, 500, 10))
    zscore_thresholds: np.ndarray = field(default_factory=lambda: np.arange(0.0, 3.0, 0.15))
    
    # Min-max scaling parameters
    minmax_windows: np.ndarray = field(default_factory=lambda: np.arange(10, 490, 20))
    minmax_thresholds: np.ndarray = field(default_factory=lambda: np.arange(0.1, 1.0, 0.1))
    
    # Moving average parameters
    ma_short_windows: np.ndarray = field(default_factory=lambda: np.arange(10, 100, 10))
    ma_long_windows: np.ndarray = field(default_factory=lambda: np.arange(50, 500, 50))
    ma_thresholds: np.ndarray = field(default_factory=lambda: np.arange(0.00, 0.05, 0.01))
    
    # EWMA parameters
    ewma_fast_spans: np.ndarray = field(default_factory=lambda: np.arange(10, 200, 10))
    ewma_slow_spans: np.ndarray = field(default_factory=lambda: np.arange(50, 500, 50))
    ewma_alpha_pool: np.ndarray = field(default_factory=lambda: np.arange(0.0, 1.0, 0.05))
    ewma_thresholds: np.ndarray = field(default_factory=lambda: np.arange(0.00, 0.05, 0.01))
    
    # RSI parameters
    rsi_windows: np.ndarray = field(default_factory=lambda: np.arange(10, 50, 5))
    rsi_thresholds: List[Tuple[float, float]] = field(
        default_factory=lambda: [(20, 80), (25, 75), (30, 70)]
    )

@dataclass
class OutputConfig:
    """Configuration for output settings"""
    output_dir: str = "results"
    generate_heatmaps: bool = True
    generate_plots: bool = True
    save_detailed_trades: bool = True
    plot_format: str = "png"
    figure_size: Tuple[int, int] = (12, 8)
    
    # Report settings
    include_statistical_tests: bool = True
    detailed_json_output: bool = True

# Global configuration instances
BACKTEST_CONFIG = BacktestConfig()
MODEL_PARAMS = ModelParameterRanges()
OUTPUT_CONFIG = OutputConfig()

# Strategy mappings
STRATEGY_NAMES = {
    'trend': 'Trend Following',
    'trend_close': 'Trend Following (Close on Exit)',
    'trend_close_zero': 'Trend Following (Zero on Exit)',
    'mr': 'Mean Reversion',
    'mr_close': 'Mean Reversion (Close on Exit)', 
    'mean_reversion_zero': 'Mean Reversion (Zero on Exit)'
}

MODEL_NAMES = {
    'zscore': 'Z-Score Normalization',
    'min_max_scaling': 'Min-Max Scaling',
    'ma_diff': 'Moving Average Difference',
    'ma_double': 'Dual Moving Average',
    'ewma_diff': 'EWMA Difference',
    'rsi': 'Relative Strength Index'
}