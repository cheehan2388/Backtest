"""
Statistical testing module for the backtesting system.
Includes permutation tests and other statistical significance tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Callable
import logging
from scipy import stats
from .config import BACKTEST_CONFIG

logger = logging.getLogger(__name__)

class PermutationTester:
    """Permutation test implementation for strategy significance testing"""
    
    def __init__(self, config=None):
        self.config = config or BACKTEST_CONFIG
    
    def run_permutation_test(self, 
                           data: pd.Series,
                           model_func: Callable,
                           strategy_func: Callable,
                           model_params: Dict[str, Any],
                           strategy_params: Dict[str, Any],
                           price: pd.Series,
                           original_sharpe: float,
                           num_permutations: int = None,
                           seed: int = None) -> Dict[str, Any]:
        """
        Run permutation test for statistical significance
        
        Args:
            data: Factor data series
            model_func: Model function
            strategy_func: Strategy function
            model_params: Model parameters
            strategy_params: Strategy parameters
            price: Price series
            original_sharpe: Original strategy Sharpe ratio
            num_permutations: Number of permutations (default from config)
            seed: Random seed (default from config)
            
        Returns:
            Dictionary with test results
        """
        num_permutations = num_permutations or self.config.permutation_tests
        seed = seed or self.config.permutation_seed
        
        rng = np.random.default_rng(seed)
        
        # Calculate returns for permutation
        returns = price.pct_change().fillna(0)
        n = len(returns)
        
        # Generate signal and positions for original strategy
        signal = model_func(data, **model_params)
        positions = strategy_func(signal, **strategy_params)
        
        # Run permutations
        permuted_sharpes = []
        significant_count = 0
        
        logger.info(f"Running {num_permutations} permutations...")
        
        for i in range(num_permutations):
            # Create block-shuffled returns
            shuffled_returns = self._block_shuffle(returns, rng)
            
            # Calculate PnL with shuffled returns
            pos_shifted = positions.shift(1).fillna(0)
            pnl_shuffled = pos_shifted * shuffled_returns
            
            # Calculate Sharpe ratio
            if pnl_shuffled.std() > 0:
                sharpe_shuffled = (pnl_shuffled.mean() / pnl_shuffled.std()) * np.sqrt(365 * self.config.sr_multiplier)
            else:
                sharpe_shuffled = 0
            
            permuted_sharpes.append(sharpe_shuffled)
            
            # Count significant results
            if sharpe_shuffled >= original_sharpe:
                significant_count += 1
        
        # Calculate p-value
        p_value = significant_count / num_permutations
        
        # Additional statistics
        permuted_sharpes = np.array(permuted_sharpes)
        
        results = {
            'p_value': p_value,
            'original_sharpe': original_sharpe,
            'permuted_sharpes': permuted_sharpes,
            'mean_permuted_sharpe': permuted_sharpes.mean(),
            'std_permuted_sharpe': permuted_sharpes.std(),
            'percentile_95': np.percentile(permuted_sharpes, 95),
            'percentile_99': np.percentile(permuted_sharpes, 99),
            'num_permutations': num_permutations,
            'significant': p_value < self.config.significance_level
        }
        
        logger.info(f"Permutation test completed. P-value: {p_value:.4f}")
        
        return results
    
    def _block_shuffle(self, returns: pd.Series, rng: np.random.Generator) -> pd.Series:
        """
        Block shuffle returns to preserve temporal structure
        
        Args:
            returns: Original returns series
            rng: Random number generator
            
        Returns:
            Block-shuffled returns series
        """
        n = len(returns)
        blocks = []
        i = 0
        
        while i < n:
            # Random block length between 10 and 500
            block_length = rng.integers(10, min(501, n - i + 1))
            blocks.append(returns.iloc[i:i + block_length])
            i += block_length
        
        # Shuffle blocks
        rng.shuffle(blocks)
        
        # Concatenate shuffled blocks
        shuffled = pd.concat(blocks)[:n]
        shuffled.index = returns.index
        
        return shuffled

class StatisticalAnalyzer:
    """Comprehensive statistical analysis for backtesting results"""
    
    def __init__(self):
        pass
    
    def analyze_strategy_significance(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Analyze statistical significance of multiple strategies
        
        Args:
            results: List of strategy results with permutation test data
            
        Returns:
            DataFrame with significance analysis
        """
        analysis_data = []
        
        for result in results:
            if 'permutation_test' in result:
                perm_test = result['permutation_test']
                
                analysis_data.append({
                    'strategy_name': result.get('name', 'Unknown'),
                    'model': result.get('model', 'Unknown'),
                    'strategy': result.get('strategy', 'Unknown'),
                    'original_sharpe': perm_test['original_sharpe'],
                    'p_value': perm_test['p_value'],
                    'significant': perm_test['significant'],
                    'mean_permuted_sharpe': perm_test['mean_permuted_sharpe'],
                    'sharpe_percentile_95': perm_test['percentile_95'],
                    'sharpe_percentile_99': perm_test['percentile_99'],
                    'z_score': self._calculate_z_score(
                        perm_test['original_sharpe'],
                        perm_test['mean_permuted_sharpe'],
                        perm_test['std_permuted_sharpe']
                    )
                })
        
        return pd.DataFrame(analysis_data)
    
    def _calculate_z_score(self, original: float, mean_perm: float, std_perm: float) -> float:
        """Calculate z-score for original result vs permuted distribution"""
        if std_perm == 0:
            return np.inf if original > mean_perm else -np.inf
        return (original - mean_perm) / std_perm
    
    def multiple_testing_correction(self, p_values: List[float], 
                                   method: str = 'bonferroni') -> List[float]:
        """
        Apply multiple testing correction
        
        Args:
            p_values: List of p-values
            method: Correction method ('bonferroni', 'holm', 'bh')
            
        Returns:
            List of corrected p-values
        """
        p_values = np.array(p_values)
        n = len(p_values)
        
        if method == 'bonferroni':
            return np.minimum(p_values * n, 1.0)
        
        elif method == 'holm':
            # Holm-Bonferroni correction
            sorted_indices = np.argsort(p_values)
            corrected = np.zeros_like(p_values)
            
            for i, idx in enumerate(sorted_indices):
                corrected[idx] = min(p_values[idx] * (n - i), 1.0)
            
            return corrected
        
        elif method == 'bh':
            # Benjamini-Hochberg (FDR) correction
            sorted_indices = np.argsort(p_values)
            corrected = np.zeros_like(p_values)
            
            for i, idx in enumerate(sorted_indices):
                corrected[idx] = min(p_values[idx] * n / (i + 1), 1.0)
            
            return corrected
        
        else:
            raise ValueError(f"Unknown correction method: {method}")
    
    def bootstrap_confidence_interval(self, data: pd.Series, 
                                    statistic_func: Callable = np.mean,
                                    confidence: float = 0.95,
                                    n_bootstrap: int = 1000,
                                    seed: int = 42) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for a statistic
        
        Args:
            data: Data series
            statistic_func: Function to calculate statistic
            confidence: Confidence level
            n_bootstrap: Number of bootstrap samples
            seed: Random seed
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        rng = np.random.default_rng(seed)
        n = len(data)
        
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            bootstrap_sample = rng.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_func(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        alpha = 1 - confidence
        
        lower_bound = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper_bound = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return lower_bound, upper_bound
    
    def sharpe_ratio_test(self, returns1: pd.Series, returns2: pd.Series) -> Dict[str, Any]:
        """
        Test for significant difference between two Sharpe ratios
        
        Args:
            returns1: First strategy returns
            returns2: Second strategy returns
            
        Returns:
            Dictionary with test results
        """
        # Calculate Sharpe ratios
        sharpe1 = returns1.mean() / returns1.std() if returns1.std() > 0 else 0
        sharpe2 = returns2.mean() / returns2.std() if returns2.std() > 0 else 0
        
        # Jobson-Korkie test for Sharpe ratio difference
        n = min(len(returns1), len(returns2))
        
        # Align series
        common_index = returns1.index.intersection(returns2.index)
        r1_aligned = returns1.reindex(common_index)
        r2_aligned = returns2.reindex(common_index)
        
        # Calculate covariance
        cov_matrix = np.cov(r1_aligned, r2_aligned)
        var1, var2 = cov_matrix[0, 0], cov_matrix[1, 1]
        cov12 = cov_matrix[0, 1]
        
        # Test statistic
        if var1 > 0 and var2 > 0:
            theta = (sharpe1**2 * var2 + sharpe2**2 * var1 - 2 * sharpe1 * sharpe2 * cov12) / n
            if theta > 0:
                test_stat = (sharpe1 - sharpe2) / np.sqrt(theta)
                p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
            else:
                test_stat = np.inf
                p_value = 0.0
        else:
            test_stat = np.inf
            p_value = 0.0
        
        return {
            'sharpe1': sharpe1,
            'sharpe2': sharpe2,
            'difference': sharpe1 - sharpe2,
            'test_statistic': test_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def drawdown_analysis(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive drawdown analysis
        
        Args:
            equity_curve: Equity curve series
            
        Returns:
            Dictionary with drawdown statistics
        """
        running_max = equity_curve.cummax()
        drawdown = equity_curve - running_max
        drawdown_pct = drawdown / running_max
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []
        
        if in_drawdown.any():
            # Find start and end of drawdown periods
            drawdown_starts = in_drawdown & ~in_drawdown.shift(1).fillna(False)
            drawdown_ends = ~in_drawdown & in_drawdown.shift(1).fillna(False)
            
            starts = drawdown_starts[drawdown_starts].index
            ends = drawdown_ends[drawdown_ends].index
            
            # Handle case where drawdown continues to end
            if len(starts) > len(ends):
                ends = ends.append(pd.Index([equity_curve.index[-1]]))
            
            for start, end in zip(starts, ends):
                period_drawdown = drawdown.loc[start:end]
                max_dd = period_drawdown.min()
                duration = (end - start).days if hasattr((end - start), 'days') else len(period_drawdown)
                
                drawdown_periods.append({
                    'start': start,
                    'end': end,
                    'duration_days': duration,
                    'max_drawdown': max_dd,
                    'max_drawdown_pct': max_dd / running_max.loc[start] if running_max.loc[start] != 0 else 0
                })
        
        # Overall statistics
        max_drawdown = drawdown.min()
        max_drawdown_pct = drawdown_pct.min()
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        
        # Recovery analysis
        recovery_times = []
        for period in drawdown_periods:
            recovery_start = period['end']
            recovery_target = running_max.loc[period['start']]
            
            # Find when equity recovers to previous high
            future_equity = equity_curve.loc[recovery_start:]
            recovery_points = future_equity >= recovery_target
            
            if recovery_points.any():
                recovery_end = recovery_points[recovery_points].index[0]
                recovery_time = (recovery_end - recovery_start).days if hasattr((recovery_end - recovery_start), 'days') else 0
                recovery_times.append(recovery_time)
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'avg_drawdown': avg_drawdown,
            'num_drawdown_periods': len(drawdown_periods),
            'drawdown_periods': drawdown_periods,
            'avg_recovery_time': np.mean(recovery_times) if recovery_times else 0,
            'max_recovery_time': np.max(recovery_times) if recovery_times else 0
        }