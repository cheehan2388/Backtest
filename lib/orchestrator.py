"""
Main orchestrator for the backtesting system.
Coordinates all components and provides the primary interface.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

from .config import BACKTEST_CONFIG, MODEL_PARAMS, OUTPUT_CONFIG
from .data_handler import DataHandler
from .models import MODEL_REGISTRY
from .strategies import STRATEGY_REGISTRY
from .backtester import BacktestEngine, BacktestResult
from .visualization import Visualizer, create_summary_table
from .statistical_tests import PermutationTester, StatisticalAnalyzer

logger = logging.getLogger(__name__)

class BacktestOrchestrator:
    """Main orchestrator for the backtesting system"""
    
    def __init__(self, config=None, model_params=None, output_config=None):
        self.config = config or BACKTEST_CONFIG
        self.model_params = model_params or MODEL_PARAMS
        self.output_config = output_config or OUTPUT_CONFIG
        
        # Initialize components
        self.data_handler = DataHandler(self.config)
        self.backtest_engine = BacktestEngine(self.config)
        self.visualizer = Visualizer(self.output_config)
        self.permutation_tester = PermutationTester(self.config)
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Results storage
        self.all_results = []
        self.filtered_results = []
        self.significant_results = []
        
    def run_full_pipeline(self, 
                         data_file: str,
                         factor_column: str,
                         output_dir: str,
                         interval: str = "1h",
                         selected_models: Optional[List[str]] = None,
                         selected_strategies: Optional[List[str]] = None,
                         run_permutation_tests: bool = True,
                         generate_reports: bool = True) -> Dict[str, Any]:
        """
        Run the complete backtesting pipeline
        
        Args:
            data_file: Path to data file
            factor_column: Name of the factor column
            output_dir: Output directory
            interval: Data interval
            selected_models: List of models to test (None for all)
            selected_strategies: List of strategies to test (None for all)
            run_permutation_tests: Whether to run permutation tests
            generate_reports: Whether to generate reports and plots
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting backtesting pipeline...")
        
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and split data
        logger.info("Loading and splitting data...")
        df = self.data_handler.load_data(data_file, parse_dates=[self.config.date_column])
        splits = self.data_handler.split_data()
        
        # Get model and strategy selections
        models_to_test = selected_models or MODEL_REGISTRY.get_model_names()
        strategies_to_test = selected_strategies or STRATEGY_REGISTRY.get_strategy_names()
        
        logger.info(f"Testing {len(models_to_test)} models and {len(strategies_to_test)} strategies")
        
        # Run grid search
        self.all_results = self._run_grid_search(
            splits, factor_column, models_to_test, strategies_to_test
        )
        
        logger.info(f"Generated {len(self.all_results)} total results")
        
        # Filter results
        self.filtered_results = self.backtest_engine.filter_results(
            [r['backtest_result'] for r in self.all_results]
        )
        
        logger.info(f"Filtered to {len(self.filtered_results)} promising strategies")
        
        # Run forward tests on filtered results
        forward_results = self._run_forward_tests()
        
        # Run permutation tests if requested
        if run_permutation_tests:
            logger.info("Running permutation tests...")
            self.significant_results = self._run_permutation_tests(splits, factor_column)
        
        # Generate reports and visualizations
        if generate_reports:
            self._generate_reports(output_dir, factor_column, interval)
        
        # Prepare summary
        summary = {
            'total_combinations_tested': len(self.all_results),
            'filtered_strategies': len(self.filtered_results),
            'significant_strategies': len(self.significant_results) if run_permutation_tests else 0,
            'data_info': self.data_handler.get_data_info(),
            'output_directory': str(output_path.absolute())
        }
        
        logger.info("Pipeline completed successfully!")
        return summary
    
    def _run_grid_search(self, 
                        splits: Dict[str, pd.DataFrame],
                        factor_column: str,
                        models_to_test: List[str],
                        strategies_to_test: List[str]) -> List[Dict[str, Any]]:
        """Run grid search across all model/strategy combinations"""
        
        results = []
        total_combinations = 0
        
        # Count total combinations for progress bar
        for model_name in models_to_test:
            model = MODEL_REGISTRY.get_model(model_name)
            param_ranges = model.get_param_ranges()
            
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(param_ranges)
            total_combinations += len(param_combinations) * len(strategies_to_test)
        
        logger.info(f"Running grid search over {total_combinations} combinations...")
        
        with tqdm(total=total_combinations, desc="Grid Search") as pbar:
            for model_name in models_to_test:
                model = MODEL_REGISTRY.get_model(model_name)
                param_ranges = model.get_param_ranges()
                
                # Generate parameter combinations
                param_combinations = self._generate_parameter_combinations(param_ranges)
                
                for params in param_combinations:
                    # Generate signal on backtest data
                    factor_data = splits['backtest'][factor_column]
                    
                    try:
                        if isinstance(params, dict):
                            signal = model(factor_data, **params)
                        else:
                            signal = model(factor_data, params)
                    except Exception as e:
                        logger.warning(f"Failed to generate signal for {model_name} with params {params}: {e}")
                        pbar.update(len(strategies_to_test))
                        continue
                    
                    for strategy_name in strategies_to_test:
                        strategy = STRATEGY_REGISTRY.get_strategy(strategy_name)
                        
                        # Test different thresholds
                        thresholds = param_ranges.get('thresholds', [0.5])
                        if not isinstance(thresholds, (list, np.ndarray)):
                            thresholds = [thresholds]
                        
                        for threshold in thresholds:
                            try:
                                # Generate positions
                                positions = strategy(signal, threshold=threshold)
                                
                                # Run backtest
                                price_data = splits['backtest'][self.config.price_column]
                                backtest_result = self.backtest_engine.run_backtest(price_data, positions)
                                
                                # Store result
                                result = {
                                    'model': model_name,
                                    'strategy': strategy_name,
                                    'parameters': params,
                                    'threshold': threshold,
                                    'backtest_result': backtest_result,
                                    'factor_column': factor_column
                                }
                                results.append(result)
                                
                            except Exception as e:
                                logger.warning(f"Failed backtest for {model_name}/{strategy_name}: {e}")
                        
                        pbar.update(1)
        
        return results
    
    def _generate_parameter_combinations(self, param_ranges: Dict[str, Any]) -> List[Any]:
        """Generate all parameter combinations from ranges"""
        combinations = []
        
        # Handle different parameter structures
        if 'window' in param_ranges:
            # Single window parameter
            for window in param_ranges['window']:
                combinations.append(window)
                
        elif 'short_window' in param_ranges and 'long_window' in param_ranges:
            # Dual window parameters
            for short_w in param_ranges['short_window']:
                for long_w in param_ranges['long_window']:
                    if short_w < long_w:
                        combinations.append((short_w, long_w))
                        
        elif 'fast_span' in param_ranges and 'slow_span' in param_ranges:
            # EWMA parameters
            for fast_span in param_ranges['fast_span']:
                for slow_span in param_ranges['slow_span']:
                    if fast_span < slow_span:
                        alpha = param_ranges.get('alpha', [0.1])[0]  # Use first alpha value
                        combinations.append((fast_span, slow_span, alpha))
        
        # If no combinations generated, return a default
        if not combinations:
            combinations = [{}]
        
        return combinations
    
    def _run_forward_tests(self) -> List[Dict[str, Any]]:
        """Run forward tests on filtered results"""
        forward_results = []
        
        logger.info("Running forward tests on filtered strategies...")
        
        splits = self.data_handler.splits
        
        for result in tqdm(self.filtered_results, desc="Forward Tests"):
            # Find corresponding grid search result
            grid_result = None
            for gr in self.all_results:
                if (gr['backtest_result'] == result and 
                    gr['model'] == result.model if hasattr(result, 'model') else True):
                    grid_result = gr
                    break
            
            if not grid_result:
                continue
                
            try:
                # Generate signal on forward data
                model = MODEL_REGISTRY.get_model(grid_result['model'])
                factor_data = splits['forward'][grid_result['factor_column']]
                
                if isinstance(grid_result['parameters'], dict):
                    signal = model(factor_data, **grid_result['parameters'])
                else:
                    signal = model(factor_data, grid_result['parameters'])
                
                # Generate positions
                strategy = STRATEGY_REGISTRY.get_strategy(grid_result['strategy'])
                positions = strategy(signal, threshold=grid_result['threshold'])
                
                # Run forward test
                price_data = splits['forward'][self.config.price_column]
                forward_result = self.backtest_engine.run_backtest(price_data, positions)
                
                # Check consistency
                is_consistent = self.backtest_engine.check_forward_consistency(
                    grid_result['backtest_result'], forward_result
                )
                
                forward_results.append({
                    **grid_result,
                    'forward_result': forward_result,
                    'is_consistent': is_consistent
                })
                
            except Exception as e:
                logger.warning(f"Forward test failed: {e}")
        
        return forward_results
    
    def _run_permutation_tests(self, 
                              splits: Dict[str, pd.DataFrame],
                              factor_column: str) -> List[Dict[str, Any]]:
        """Run permutation tests on forward-consistent strategies"""
        significant_results = []
        
        # Get forward-consistent results
        consistent_results = [r for r in self._run_forward_tests() if r.get('is_consistent', False)]
        
        logger.info(f"Running permutation tests on {len(consistent_results)} consistent strategies...")
        
        for result in tqdm(consistent_results, desc="Permutation Tests"):
            try:
                # Get model and strategy functions
                model = MODEL_REGISTRY.get_model(result['model'])
                strategy = STRATEGY_REGISTRY.get_strategy(result['strategy'])
                
                # Prepare parameters
                model_params = result['parameters'] if isinstance(result['parameters'], dict) else {}
                strategy_params = {'threshold': result['threshold']}
                
                # Run permutation test on full data
                full_data = splits['full']
                perm_result = self.permutation_tester.run_permutation_test(
                    data=full_data[factor_column],
                    model_func=model,
                    strategy_func=strategy,
                    model_params=model_params,
                    strategy_params=strategy_params,
                    price=full_data[self.config.price_column],
                    original_sharpe=result['backtest_result'].sharpe_ratio
                )
                
                if perm_result['significant']:
                    significant_results.append({
                        **result,
                        'permutation_test': perm_result
                    })
                    
            except Exception as e:
                logger.warning(f"Permutation test failed: {e}")
        
        return significant_results
    
    def _generate_reports(self, output_dir: str, factor_column: str, interval: str):
        """Generate comprehensive reports and visualizations"""
        output_path = Path(output_dir)
        
        # Save all results to JSON
        self._save_results_json(output_path, factor_column, interval)
        
        # Generate heatmaps if configured
        if self.output_config.generate_heatmaps:
            self._generate_heatmaps(output_path, factor_column, interval)
        
        # Generate performance plots
        if self.output_config.generate_plots:
            self._generate_performance_plots(output_path)
        
        # Generate summary reports
        self._generate_summary_reports(output_path)
    
    def _save_results_json(self, output_path: Path, factor_column: str, interval: str):
        """Save results to JSON files"""
        
        # Save backtest results
        backtest_data = []
        for result in self.all_results:
            bt_result = result['backtest_result']
            backtest_data.append({
                'factor_name': factor_column,
                'model': result['model'],
                'strategy': result['strategy'],
                'parameters': str(result['parameters']),
                'threshold': result['threshold'],
                'interval': interval,
                'sharpe_ratio': bt_result.sharpe_ratio,
                'total_return': bt_result.total_return,
                'max_drawdown': bt_result.max_drawdown_pct,
                'num_trades': bt_result.num_trades,
                'trade_frequency': bt_result.trade_per_interval,
                'start_date': bt_result.start_date.isoformat(),
                'end_date': bt_result.end_date.isoformat()
            })
        
        with open(output_path / f'{factor_column}_backtest_results_{interval}.json', 'w') as f:
            json.dump({'backtest_results': backtest_data}, f, indent=2)
        
        # Save significant results if available
        if self.significant_results:
            significant_data = []
            for result in self.significant_results:
                bt_result = result['backtest_result']
                perm_test = result['permutation_test']
                
                significant_data.append({
                    'factor_name': factor_column,
                    'model': result['model'],
                    'strategy': result['strategy'],
                    'parameters': str(result['parameters']),
                    'threshold': result['threshold'],
                    'sharpe_ratio': bt_result.sharpe_ratio,
                    'p_value': perm_test['p_value'],
                    'significant': perm_test['significant']
                })
            
            with open(output_path / f'{factor_column}_significant_strategies_{interval}.json', 'w') as f:
                json.dump({'significant_strategies': significant_data}, f, indent=2)
    
    def _generate_heatmaps(self, output_path: Path, factor_column: str, interval: str):
        """Generate parameter heatmaps"""
        
        # Convert results to DataFrame for heatmap generation
        heatmap_data = []
        for result in self.all_results:
            heatmap_data.append({
                'model': result['model'],
                'strategy': result['strategy'],
                'windows': result['parameters'],
                'threshold': result['threshold'],
                'sharpe_ratio': result['backtest_result'].sharpe_ratio
            })
        
        df_heatmap = pd.DataFrame(heatmap_data)
        
        # Generate heatmaps for each model/strategy combination
        for model in df_heatmap['model'].unique():
            for strategy in df_heatmap['strategy'].unique():
                self.visualizer.plot_parameter_heatmaps(
                    df_heatmap, model, strategy, str(output_path), factor_column
                )
    
    def _generate_performance_plots(self, output_path: Path):
        """Generate performance plots"""
        
        if not self.significant_results:
            logger.warning("No significant results to plot")
            return
        
        # Prepare data for plotting
        plot_results = []
        for result in self.significant_results[:10]:  # Plot top 10
            name = f"{result['model']}_{result['strategy']}"
            plot_results.append((name, result['backtest_result']))
        
        # Generate equity curves
        self.visualizer.plot_equity_curves(
            plot_results,
            str(output_path / "equity_curves.png"),
            "Top Performing Strategies"
        )
        
        # Generate performance dashboard
        self.visualizer.create_performance_dashboard(
            plot_results,
            str(output_path / "performance_dashboard.html")
        )
        
        # Generate individual strategy plots
        for name, result in plot_results[:5]:  # Top 5 detailed plots
            splits = self.data_handler.splits
            price_data = splits['full'][self.config.price_column]
            
            self.visualizer.plot_strategy_performance(
                result, price_data,
                str(output_path / f"strategy_detail_{name}.png"),
                name
            )
    
    def _generate_summary_reports(self, output_path: Path):
        """Generate summary reports"""
        
        if not self.significant_results:
            return
        
        # Create summary table
        plot_results = [(f"{r['model']}_{r['strategy']}", r['backtest_result']) 
                       for r in self.significant_results]
        
        summary_df = create_summary_table(plot_results)
        summary_df.to_csv(output_path / "strategy_summary.csv", index=False)
        
        # Statistical analysis
        if self.significant_results:
            significance_df = self.statistical_analyzer.analyze_strategy_significance(
                self.significant_results
            )
            significance_df.to_csv(output_path / "significance_analysis.csv", index=False)
    
    def get_strategy_selector(self) -> Dict[str, Any]:
        """
        Get a dictionary of strategies that can be manually selected
        
        Returns:
            Dictionary with strategy selection interface
        """
        if not self.all_results:
            logger.warning("No results available. Run the pipeline first.")
            return {}
        
        selection_data = {}
        
        for i, result in enumerate(self.all_results):
            bt_result = result['backtest_result']
            
            key = f"{result['model']}_{result['strategy']}_{i}"
            selection_data[key] = {
                'model': result['model'],
                'strategy': result['strategy'],
                'parameters': result['parameters'],
                'threshold': result['threshold'],
                'sharpe_ratio': bt_result.sharpe_ratio,
                'total_return': bt_result.total_return,
                'max_drawdown': bt_result.max_drawdown_pct,
                'num_trades': bt_result.num_trades,
                'meets_criteria': bt_result in self.filtered_results
            }
        
        return selection_data