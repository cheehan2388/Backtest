"""
Comprehensive Backtesting System

A flexible and extensible backtesting framework for quantitative trading strategies.
Supports multiple models, strategies, walk-forward analysis, and statistical significance testing.
"""

from .config import BACKTEST_CONFIG, MODEL_PARAMS, OUTPUT_CONFIG
from .data_handler import DataHandler
from .models import MODEL_REGISTRY
from .strategies import STRATEGY_REGISTRY
from .backtester import BacktestEngine, BacktestResult
from .visualization import Visualizer, create_summary_table
from .statistical_tests import PermutationTester, StatisticalAnalyzer
from .orchestrator import BacktestOrchestrator

# Backward compatibility imports
from .models import (
    compute_zscore, compute_minmax, compute_ma_diff, 
    compute_ma_double, compute_ewma_diff, compute_RSI, compute_percentile
)
from .strategies import (
    positions_trend, positions_trend_close, positions_trend_zero,
    positions_mr, positions_mr_close, positions_mr_mean_zero
)

__version__ = "1.0.0"
__author__ = "Backtesting System"

__all__ = [
    # Main components
    'BacktestOrchestrator',
    'DataHandler',
    'BacktestEngine',
    'BacktestResult',
    'Visualizer',
    'PermutationTester',
    'StatisticalAnalyzer',
    
    # Registries
    'MODEL_REGISTRY',
    'STRATEGY_REGISTRY',
    
    # Configuration
    'BACKTEST_CONFIG',
    'MODEL_PARAMS',
    'OUTPUT_CONFIG',
    
    # Utilities
    'create_summary_table',
    
    # Backward compatibility
    'compute_zscore',
    'compute_minmax', 
    'compute_ma_diff',
    'compute_ma_double',
    'compute_ewma_diff',
    'compute_RSI',
    'compute_percentile',
    'positions_trend',
    'positions_trend_close',
    'positions_trend_zero',
    'positions_mr',
    'positions_mr_close',
    'positions_mr_mean_zero'
]