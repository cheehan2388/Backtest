"""
Comprehensive backtesting engine for the trading system.
Handles performance calculation, metrics computation, and strategy filtering.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from dataclasses import dataclass
from .config import BACKTEST_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Container for backtest results"""
    # Core metrics
    positions: pd.Series
    pnl: pd.Series
    equity: pd.Series
    trade_signals: pd.Series
    
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    
    # Trade statistics
    num_trades: int
    trade_per_interval: float
    win_rate: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    profit_factor: Optional[float] = None
    
    # Additional metrics
    volatility: float = None
    skewness: float = None
    kurtosis: float = None
    var_95: float = None
    cvar_95: float = None
    
    # Metadata
    start_date: pd.Timestamp = None
    end_date: pd.Timestamp = None
    duration_days: int = None

class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, config=None):
        self.config = config or BACKTEST_CONFIG
        
    def run_backtest(self, price: pd.Series, positions: pd.Series, 
                    fee: float = None, sr_multiplier: float = None) -> BacktestResult:
        """
        Run a complete backtest
        
        Args:
            price: Price series
            positions: Position series (-1, 0, 1)
            fee: Trading fee (default from config)
            sr_multiplier: Sharpe ratio multiplier (default from config)
            
        Returns:
            BacktestResult object with all metrics
        """
        fee = fee or self.config.fee
        sr_multiplier = sr_multiplier or self.config.sr_multiplier
        
        # Align series
        price, positions = self._align_series(price, positions)
        
        # Calculate basic metrics
        pos_shifted = positions.shift(1).fillna(0)
        trade_signals = (positions != pos_shifted).astype(int)
        
        # Calculate returns and PnL
        returns = price.pct_change().fillna(0)
        pnl = pos_shifted * returns - trade_signals * fee
        equity = pnl.cumsum()
        
        # Calculate drawdown
        running_max = equity.cummax()
        drawdown = equity - running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = max_drawdown
        
        # Calculate performance metrics
        total_return = equity.iloc[-1]
        annualized_return = pnl.mean() * 365
        
        # Risk metrics
        if pnl.std() > 0:
            sharpe_ratio = (pnl.mean() / pnl.std()) * np.sqrt(365 * sr_multiplier)
            sortino_ratio = self._calculate_sortino_ratio(pnl, sr_multiplier)
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        num_trades = int(trade_signals.sum())
        trade_per_interval = num_trades / len(returns) if len(returns) > 0 else 0
        
        # Advanced trade statistics
        trade_stats = self._calculate_trade_statistics(pnl, trade_signals)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(pnl)
        
        # Create result object
        result = BacktestResult(
            positions=positions,
            pnl=pnl,
            equity=equity,
            trade_signals=trade_signals,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            num_trades=num_trades,
            trade_per_interval=trade_per_interval,
            start_date=price.index[0],
            end_date=price.index[-1],
            duration_days=(price.index[-1] - price.index[0]).days,
            **trade_stats,
            **risk_metrics
        )
        
        return result
    
    def _align_series(self, price: pd.Series, positions: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Align price and position series"""
        common_index = price.index.intersection(positions.index)
        if len(common_index) == 0:
            raise ValueError("Price and positions series have no common index")
        
        price_aligned = price.reindex(common_index)
        positions_aligned = positions.reindex(common_index).fillna(0)
        
        return price_aligned, positions_aligned
    
    def _calculate_sortino_ratio(self, returns: pd.Series, sr_multiplier: float, 
                                rf: float = 0.0) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - rf
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.inf if excess_returns.mean() > 0 else -np.inf
            
        downside_std = downside_returns.std()
        return (excess_returns.mean() / downside_std) * np.sqrt(365 * sr_multiplier)
    
    def _calculate_trade_statistics(self, pnl: pd.Series, trade_signals: pd.Series) -> Dict[str, float]:
        """Calculate detailed trade statistics"""
        if trade_signals.sum() == 0:
            return {
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        # Identify individual trades
        trade_indices = trade_signals[trade_signals == 1].index
        if len(trade_indices) <= 1:
            return {
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        # Calculate trade returns (simplified approach)
        trade_returns = []
        for i in range(len(trade_indices) - 1):
            start_idx = trade_indices[i]
            end_idx = trade_indices[i + 1]
            trade_pnl = pnl.loc[start_idx:end_idx].sum()
            trade_returns.append(trade_pnl)
        
        if not trade_returns:
            return {
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        trade_returns = np.array(trade_returns)
        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns < 0]
        
        win_rate = len(winning_trades) / len(trade_returns) if len(trade_returns) > 0 else 0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        
        profit_factor = (winning_trades.sum() / abs(losing_trades.sum()) 
                        if len(losing_trades) > 0 and losing_trades.sum() != 0 else np.inf)
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate additional risk metrics"""
        if len(returns) == 0:
            return {
                'volatility': 0,
                'skewness': 0,
                'kurtosis': 0,
                'var_95': 0,
                'cvar_95': 0
            }
        
        volatility = returns.std() * np.sqrt(365)
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Value at Risk (95%)
        var_95 = returns.quantile(0.05)
        
        # Conditional Value at Risk (95%)
        cvar_95 = returns[returns <= var_95].mean()
        
        return {
            'volatility': volatility,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def filter_results(self, results: List[BacktestResult]) -> List[BacktestResult]:
        """
        Filter backtest results based on criteria
        
        Args:
            results: List of BacktestResult objects
            
        Returns:
            Filtered list of results
        """
        filtered = []
        
        for result in results:
            if self._passes_filters(result):
                filtered.append(result)
        
        logger.info(f"Filtered {len(results)} results down to {len(filtered)}")
        return filtered
    
    def _passes_filters(self, result: BacktestResult) -> bool:
        """Check if a result passes all filters"""
        # Sharpe ratio filter
        if result.sharpe_ratio < self.config.min_sharpe:
            return False
        
        # Maximum drawdown filter
        if result.max_drawdown_pct < self.config.max_drawdown:
            return False
        
        # Minimum trade frequency filter
        if result.trade_per_interval < self.config.min_trade_ratio:
            return False
        
        return True
    
    def check_forward_consistency(self, backtest_result: BacktestResult, 
                                 forward_result: BacktestResult) -> bool:
        """
        Check if forward test results are consistent with backtest results
        
        Args:
            backtest_result: Backtest result
            forward_result: Forward test result
            
        Returns:
            True if consistent, False otherwise
        """
        if backtest_result.sharpe_ratio == 0:
            return False
        
        ratio = forward_result.sharpe_ratio / backtest_result.sharpe_ratio
        min_ratio, max_ratio = self.config.forward_consistency_range
        
        return min_ratio <= ratio <= max_ratio
    
    def calculate_fitness_score(self, result: BacktestResult) -> float:
        """
        Calculate a fitness score for ranking strategies
        
        Args:
            result: BacktestResult object
            
        Returns:
            Fitness score
        """
        # Base fitness on Sharpe ratio adjusted for trade frequency
        base_score = result.sharpe_ratio
        
        # Adjust for trade frequency (penalize over-trading)
        trade_penalty = max(0.03, result.trade_per_interval)
        adjusted_score = base_score * np.sqrt(abs(result.total_return) / trade_penalty)
        
        return adjusted_score

class WalkForwardAnalyzer:
    """Walk-forward analysis implementation"""
    
    def __init__(self, config=None):
        self.config = config or BACKTEST_CONFIG
        self.engine = BacktestEngine(config)
    
    def run_walk_forward(self, data: pd.DataFrame, model_func, strategy_func,
                        model_params: Dict, strategy_params: Dict,
                        factor_column: str, price_column: str = None,
                        window_size: int = 252, step_size: int = 63) -> List[BacktestResult]:
        """
        Run walk-forward analysis
        
        Args:
            data: Full dataset
            model_func: Model function
            strategy_func: Strategy function
            model_params: Model parameters
            strategy_params: Strategy parameters
            factor_column: Factor column name
            price_column: Price column name
            window_size: Analysis window size
            step_size: Step size for walk-forward
            
        Returns:
            List of BacktestResult objects for each window
        """
        price_column = price_column or self.config.price_column
        results = []
        
        start_idx = window_size
        while start_idx < len(data):
            end_idx = min(start_idx + step_size, len(data))
            
            # Get data window
            window_data = data.iloc[start_idx-window_size:end_idx]
            
            # Generate signal
            signal = model_func(window_data[factor_column], **model_params)
            
            # Generate positions
            positions = strategy_func(signal, **strategy_params)
            
            # Run backtest on out-of-sample period
            oos_data = window_data.iloc[-step_size:]
            oos_price = oos_data[price_column]
            oos_positions = positions.iloc[-step_size:]
            
            result = self.engine.run_backtest(oos_price, oos_positions)
            results.append(result)
            
            start_idx += step_size
        
        return results