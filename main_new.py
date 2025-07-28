"""
New Main Script for the Comprehensive Backtesting System

This script demonstrates how to use the refactored backtesting framework
with improved architecture, flexibility, and comprehensive reporting.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Any

from lib import BacktestOrchestrator, BACKTEST_CONFIG, MODEL_PARAMS, OUTPUT_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_comprehensive_backtest():
    """Run a comprehensive backtest using the new system"""
    
    # Configuration
    DATA_FILE = '../Data/factor/Binance_BTCUSDT_perpetual_1H_takerbuysellvolume_facto.csv'
    FACTOR_COLUMN = 'change_open_taker_ratio'
    OUTPUT_DIR = 'results_new'
    INTERVAL = '1h'
    
    # Initialize the orchestrator
    orchestrator = BacktestOrchestrator()
    
    # Run the full pipeline
    logger.info("Starting comprehensive backtesting pipeline...")
    
    try:
        results = orchestrator.run_full_pipeline(
            data_file=DATA_FILE,
            factor_column=FACTOR_COLUMN,
            output_dir=OUTPUT_DIR,
            interval=INTERVAL,
            selected_models=['zscore', 'min_max_scaling', 'ma_double'],  # Subset for demo
            selected_strategies=['trend', 'mr', 'trend_close_zero'],  # Subset for demo
            run_permutation_tests=True,
            generate_reports=True
        )
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results summary: {results}")
        
        # Get strategy selector for manual selection
        strategy_selector = orchestrator.get_strategy_selector()
        
        # Display top strategies
        print("\n" + "="*80)
        print("TOP PERFORMING STRATEGIES (sorted by Sharpe ratio)")
        print("="*80)
        
        # Sort strategies by Sharpe ratio
        sorted_strategies = sorted(
            strategy_selector.items(),
            key=lambda x: x[1]['sharpe_ratio'],
            reverse=True
        )
        
        print(f"{'Rank':<5} {'Model':<15} {'Strategy':<20} {'Sharpe':<8} {'Return':<10} {'MDD':<8} {'Trades':<8}")
        print("-" * 80)
        
        for i, (key, data) in enumerate(sorted_strategies[:20], 1):
            print(f"{i:<5} {data['model']:<15} {data['strategy']:<20} "
                  f"{data['sharpe_ratio']:<8.2f} {data['total_return']:<10.2%} "
                  f"{data['max_drawdown']:<8.2%} {data['num_trades']:<8}")
        
        # Manual strategy selection example
        print("\n" + "="*80)
        print("MANUAL STRATEGY SELECTION EXAMPLE")
        print("="*80)
        
        # Select strategies meeting specific criteria
        selected_strategies = select_strategies_by_criteria(
            strategy_selector,
            min_sharpe=1.5,
            max_drawdown=-0.3,
            min_trades=10
        )
        
        print(f"Found {len(selected_strategies)} strategies meeting criteria:")
        print("- Minimum Sharpe ratio: 1.5")
        print("- Maximum drawdown: -30%")
        print("- Minimum trades: 10")
        
        for key in selected_strategies[:5]:  # Show top 5
            data = strategy_selector[key]
            print(f"  {data['model']} + {data['strategy']}: "
                  f"Sharpe={data['sharpe_ratio']:.2f}, "
                  f"Return={data['total_return']:.2%}, "
                  f"MDD={data['max_drawdown']:.2%}")
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {DATA_FILE}")
        logger.info("Please update the DATA_FILE path to point to your actual data file")
        return
    
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise

def select_strategies_by_criteria(strategy_selector: Dict[str, Any],
                                 min_sharpe: float = 2.0,
                                 max_drawdown: float = -0.5,
                                 min_trades: int = 5) -> List[str]:
    """
    Select strategies based on specific criteria
    
    Args:
        strategy_selector: Dictionary of strategies from orchestrator
        min_sharpe: Minimum Sharpe ratio
        max_drawdown: Maximum drawdown (negative value)
        min_trades: Minimum number of trades
        
    Returns:
        List of strategy keys meeting criteria
    """
    selected = []
    
    for key, data in strategy_selector.items():
        if (data['sharpe_ratio'] >= min_sharpe and
            data['max_drawdown'] >= max_drawdown and
            data['num_trades'] >= min_trades):
            selected.append(key)
    
    # Sort by Sharpe ratio
    selected.sort(key=lambda x: strategy_selector[x]['sharpe_ratio'], reverse=True)
    
    return selected

def run_walk_forward_analysis():
    """Example of walk-forward analysis"""
    
    logger.info("Running walk-forward analysis example...")
    
    # This would be implemented using the WalkForwardAnalyzer
    # from lib.backtester import WalkForwardAnalyzer
    # 
    # analyzer = WalkForwardAnalyzer()
    # results = analyzer.run_walk_forward(...)
    
    print("Walk-forward analysis would be implemented here")
    print("This provides out-of-sample testing with rolling windows")

def demonstrate_custom_model_strategy():
    """Demonstrate how to add custom models and strategies"""
    
    from lib.models import BaseModel, MODEL_REGISTRY
    from lib.strategies import BaseStrategy, STRATEGY_REGISTRY
    import numpy as np
    import pandas as pd
    
    # Custom model example
    class CustomMomentumModel(BaseModel):
        def __init__(self):
            super().__init__("custom_momentum")
        
        def generate_signal(self, data: pd.Series, lookback: int) -> pd.Series:
            return data.pct_change(lookback)
        
        def get_param_ranges(self) -> Dict[str, Any]:
            return {
                'lookback': np.arange(5, 50, 5),
                'thresholds': np.arange(0.01, 0.1, 0.01)
            }
    
    # Custom strategy example
    class CustomBandStrategy(BaseStrategy):
        def __init__(self):
            super().__init__("custom_bands")
        
        def generate_positions(self, signal: pd.Series, upper_threshold: float, 
                             lower_threshold: float) -> pd.Series:
            positions = pd.Series(0, index=signal.index)
            positions[signal > upper_threshold] = 1
            positions[signal < lower_threshold] = -1
            return positions.replace(0, np.nan).ffill().fillna(0)
    
    # Register custom components
    custom_model = CustomMomentumModel()
    custom_strategy = CustomBandStrategy()
    
    MODEL_REGISTRY.register_model(custom_model)
    STRATEGY_REGISTRY.register_strategy(custom_strategy)
    
    print("Custom model and strategy registered successfully!")
    print(f"Available models: {MODEL_REGISTRY.get_model_names()}")
    print(f"Available strategies: {STRATEGY_REGISTRY.get_strategy_names()}")

def main():
    """Main function"""
    
    print("="*80)
    print("COMPREHENSIVE BACKTESTING SYSTEM")
    print("="*80)
    print()
    
    # Create output directory
    os.makedirs('results_new', exist_ok=True)
    
    try:
        # Run main backtest
        run_comprehensive_backtest()
        
        # Demonstrate custom components
        print("\n" + "="*80)
        print("CUSTOM MODEL/STRATEGY DEMONSTRATION")
        print("="*80)
        demonstrate_custom_model_strategy()
        
        # Show walk-forward example
        print("\n" + "="*80)
        print("WALK-FORWARD ANALYSIS")
        print("="*80)
        run_walk_forward_analysis()
        
        print("\n" + "="*80)
        print("SYSTEM FEATURES SUMMARY")
        print("="*80)
        print("✓ Modular architecture with pluggable models and strategies")
        print("✓ Comprehensive performance metrics and risk analysis")
        print("✓ Statistical significance testing (permutation tests)")
        print("✓ Interactive visualizations and heatmaps")
        print("✓ Walk-forward analysis capability")
        print("✓ Flexible data handling and validation")
        print("✓ Automated report generation")
        print("✓ Manual strategy selection interface")
        print("✓ Extensible design for custom models/strategies")
        print()
        print("Check the 'results_new' directory for detailed outputs!")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    main()