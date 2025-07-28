"""
Enhanced strategy module for the backtesting system.
Contains various trading strategies with proper class-based architecture.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.last_params = None
        
    @abstractmethod
    def generate_positions(self, signal: pd.Series, **params) -> pd.Series:
        """Generate trading positions from signal"""
        pass
    
    def validate_params(self, **params) -> bool:
        """Validate strategy parameters"""
        return True
    
    def __call__(self, signal: pd.Series, **params) -> pd.Series:
        """Make the strategy callable"""
        if not self.validate_params(**params):
            raise ValueError(f"Invalid parameters for {self.name}: {params}")
        self.last_params = params
        return self.generate_positions(signal, **params)

class TrendFollowingStrategy(BaseStrategy):
    """Basic trend following strategy"""
    
    def __init__(self):
        super().__init__("trend")
    
    def generate_positions(self, signal: pd.Series, threshold: float) -> pd.Series:
        """
        Generate trend following positions
        Long when signal > threshold, short when signal < -threshold
        """
        raw_positions = np.where(signal > threshold, 1,
                                np.where(signal < -threshold, -1, 0))
        positions = pd.Series(raw_positions, index=signal.index)
        # Forward fill positions (hold until signal changes)
        return positions.replace(0, np.nan).ffill().fillna(0)

class TrendFollowingCloseStrategy(BaseStrategy):
    """Trend following strategy that closes positions when signal exits threshold"""
    
    def __init__(self):
        super().__init__("trend_close")
    
    def generate_positions(self, signal: pd.Series, threshold: float) -> pd.Series:
        """
        Generate trend following positions that close when signal exits threshold
        """
        raw_positions = np.where(signal > threshold, 1,
                                np.where(signal < -threshold, -1, 0))
        return pd.Series(raw_positions, index=signal.index)

class TrendFollowingZeroStrategy(BaseStrategy):
    """Trend following strategy that closes positions when signal returns to zero"""
    
    def __init__(self):
        super().__init__("trend_close_zero")
    
    def generate_positions(self, signal: pd.Series, threshold: float) -> pd.Series:
        """
        Generate trend following positions that close when signal returns to zero
        """
        # Generate raw signals
        raw = np.where(signal > threshold, 1,
                      np.where(signal < -threshold, -1, 0))
        
        positions = np.zeros_like(raw)
        
        # Find indices where we have signals
        signal_indices = np.flatnonzero(raw)
        if signal_indices.size == 0:
            return pd.Series(positions, index=signal.index)
        
        # Split consecutive indices into blocks (trades)
        splits = np.where(np.diff(signal_indices) > 1)[0] + 1
        blocks = np.split(signal_indices, splits)
        
        # Process each trade block
        for block in blocks:
            start = block[0]
            direction = raw[start]
            
            # Find where signal returns to zero
            if direction == 1:
                close_indices = np.where(signal[start+1:] <= 0)[0]
            else:
                close_indices = np.where(signal[start+1:] >= 0)[0]
            
            close_idx = close_indices[0] + start + 1 if close_indices.size else len(signal)
            positions[start:close_idx] = direction
        
        return pd.Series(positions, index=signal.index)

class MeanReversionStrategy(BaseStrategy):
    """Basic mean reversion strategy"""
    
    def __init__(self):
        super().__init__("mr")
    
    def generate_positions(self, signal: pd.Series, threshold: float) -> pd.Series:
        """
        Generate mean reversion positions
        Short when signal > threshold, long when signal < -threshold
        """
        raw_positions = np.where(signal > threshold, -1,
                                np.where(signal < -threshold, 1, 0))
        positions = pd.Series(raw_positions, index=signal.index)
        # Forward fill positions (hold until signal changes)
        return positions.replace(0, np.nan).ffill().fillna(0)

class MeanReversionCloseStrategy(BaseStrategy):
    """Mean reversion strategy that closes positions when signal exits threshold"""
    
    def __init__(self):
        super().__init__("mr_close")
    
    def generate_positions(self, signal: pd.Series, threshold: float) -> pd.Series:
        """
        Generate mean reversion positions that close when signal exits threshold
        """
        raw_positions = np.where(signal > threshold, -1,
                                np.where(signal < -threshold, 1, 0))
        return pd.Series(raw_positions, index=signal.index)

class MeanReversionZeroStrategy(BaseStrategy):
    """Mean reversion strategy that closes positions when signal returns to zero"""
    
    def __init__(self):
        super().__init__("mean_reversion_zero")
    
    def generate_positions(self, signal: pd.Series, threshold: float) -> pd.Series:
        """
        Generate mean reversion positions that close when signal returns to zero
        """
        # Generate raw signals (opposite of trend following)
        raw = np.where(signal > threshold, -1,
                      np.where(signal < -threshold, 1, 0))
        
        positions = np.zeros_like(raw)
        
        # Find indices where we have signals
        signal_indices = np.flatnonzero(raw)
        if signal_indices.size == 0:
            return pd.Series(positions, index=signal.index)
        
        # Split consecutive indices into blocks (trades)
        splits = np.where(np.diff(signal_indices) > 1)[0] + 1
        blocks = np.split(signal_indices, splits)
        
        # Process each trade block
        for block in blocks:
            start = block[0]
            direction = raw[start]
            
            # Find where signal returns to zero
            if direction == 1:
                close_indices = np.where(signal[start+1:] >= 0)[0]
            else:
                close_indices = np.where(signal[start+1:] <= 0)[0]
            
            close_idx = close_indices[0] + start + 1 if close_indices.size else len(signal)
            positions[start:close_idx] = direction
        
        return pd.Series(positions, index=signal.index)

class RSIStrategy(BaseStrategy):
    """RSI-based trading strategy"""
    
    def __init__(self):
        super().__init__("rsi")
    
    def generate_positions(self, signal: pd.Series, low_threshold: float = 30, 
                          high_threshold: float = 70) -> pd.Series:
        """
        Generate RSI-based positions
        Long when RSI < low_threshold, short when RSI > high_threshold
        """
        positions = pd.Series(0, index=signal.index)
        positions[signal < low_threshold] = 1   # Oversold -> Long
        positions[signal > high_threshold] = -1  # Overbought -> Short
        return positions

class PercentileStrategy(BaseStrategy):
    """Percentile-based trading strategy"""
    
    def __init__(self):
        super().__init__("percentile")
    
    def generate_positions(self, signal: pd.Series, low_threshold: float = 0.2, 
                          high_threshold: float = 0.8) -> pd.Series:
        """
        Generate percentile-based positions
        """
        positions = pd.Series(0, index=signal.index)
        positions[signal > high_threshold] = 1   # High percentile -> Long
        positions[signal < low_threshold] = -1   # Low percentile -> Short
        return positions

class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands strategy"""
    
    def __init__(self):
        super().__init__("bollinger")
    
    def generate_positions(self, signal: pd.Series, threshold: float = 2.0) -> pd.Series:
        """
        Generate Bollinger Bands positions
        Assumes signal is already normalized (z-score)
        """
        positions = pd.Series(0, index=signal.index)
        positions[signal > threshold] = -1   # Above upper band -> Short (mean reversion)
        positions[signal < -threshold] = 1   # Below lower band -> Long (mean reversion)
        return positions.replace(0, np.nan).ffill().fillna(0)

class StrategyRegistry:
    """Registry for all available strategies"""
    
    def __init__(self):
        self._strategies = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register all default strategies"""
        strategies = [
            TrendFollowingStrategy(),
            TrendFollowingCloseStrategy(),
            TrendFollowingZeroStrategy(),
            MeanReversionStrategy(),
            MeanReversionCloseStrategy(),
            MeanReversionZeroStrategy(),
            RSIStrategy(),
            PercentileStrategy(),
            BollingerBandsStrategy()
        ]
        
        for strategy in strategies:
            self.register_strategy(strategy)
    
    def register_strategy(self, strategy: BaseStrategy):
        """Register a new strategy"""
        self._strategies[strategy.name] = strategy
        logger.info(f"Registered strategy: {strategy.name}")
    
    def get_strategy(self, name: str) -> BaseStrategy:
        """Get a strategy by name"""
        if name not in self._strategies:
            raise ValueError(f"Strategy '{name}' not found. Available strategies: {list(self._strategies.keys())}")
        return self._strategies[name]
    
    def get_all_strategies(self) -> Dict[str, BaseStrategy]:
        """Get all registered strategies"""
        return self._strategies.copy()
    
    def get_strategy_names(self) -> List[str]:
        """Get list of all strategy names"""
        return list(self._strategies.keys())

# Global strategy registry
STRATEGY_REGISTRY = StrategyRegistry()

# Backward compatibility functions
def positions_trend(signal: pd.Series, threshold: float) -> pd.Series:
    """Backward compatibility function for trend following"""
    return STRATEGY_REGISTRY.get_strategy('trend')(signal, threshold=threshold)

def positions_trend_close(signal: pd.Series, threshold: float) -> pd.Series:
    """Backward compatibility function for trend following close"""
    return STRATEGY_REGISTRY.get_strategy('trend_close')(signal, threshold=threshold)

def positions_trend_zero(signal: pd.Series, threshold: float) -> pd.Series:
    """Backward compatibility function for trend following zero"""
    return STRATEGY_REGISTRY.get_strategy('trend_close_zero')(signal, threshold=threshold)

def positions_mr(signal: pd.Series, threshold: float) -> pd.Series:
    """Backward compatibility function for mean reversion"""
    return STRATEGY_REGISTRY.get_strategy('mr')(signal, threshold=threshold)

def positions_mr_close(signal: pd.Series, threshold: float) -> pd.Series:
    """Backward compatibility function for mean reversion close"""
    return STRATEGY_REGISTRY.get_strategy('mr_close')(signal, threshold=threshold)

def positions_mr_mean_zero(signal: pd.Series, threshold: float) -> pd.Series:
    """Backward compatibility function for mean reversion zero"""
    return STRATEGY_REGISTRY.get_strategy('mean_reversion_zero')(signal, threshold=threshold)