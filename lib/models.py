"""
Enhanced model module for the backtesting system.
Contains various technical indicators and signal generation models.
"""

import numpy as np
import pandas as pd
import talib as tb
from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Base class for all signal generation models"""
    
    def __init__(self, name: str):
        self.name = name
        self.last_params = None
        
    @abstractmethod
    def generate_signal(self, data: pd.Series, **params) -> pd.Series:
        """Generate trading signal from input data"""
        pass
    
    @abstractmethod
    def get_param_ranges(self) -> Dict[str, Any]:
        """Get parameter ranges for grid search"""
        pass
    
    def validate_params(self, **params) -> bool:
        """Validate model parameters"""
        return True
    
    def __call__(self, data: pd.Series, **params) -> pd.Series:
        """Make the model callable"""
        if not self.validate_params(**params):
            raise ValueError(f"Invalid parameters for {self.name}: {params}")
        self.last_params = params
        return self.generate_signal(data, **params)

class ZScoreModel(BaseModel):
    """Z-Score normalization model"""
    
    def __init__(self):
        super().__init__("zscore")
    
    def generate_signal(self, data: pd.Series, window: int) -> pd.Series:
        """Generate Z-score signal"""
        rolling_mean = data.rolling(window).mean()
        rolling_std = data.rolling(window).std()
        return (data - rolling_mean) / rolling_std
    
    def get_param_ranges(self) -> Dict[str, Any]:
        return {
            'window': np.arange(20, 500, 10),
            'thresholds': np.arange(0.0, 3.0, 0.15)
        }
    
    def validate_params(self, window: int, **kwargs) -> bool:
        return isinstance(window, (int, np.integer)) and window > 0

class MinMaxScalingModel(BaseModel):
    """Min-Max scaling model"""
    
    def __init__(self):
        super().__init__("min_max_scaling")
    
    def generate_signal(self, data: pd.Series, window: int) -> pd.Series:
        """Generate Min-Max scaled signal"""
        rolling_min = data.rolling(window).min()
        rolling_max = data.rolling(window).max()
        return (2 * (data - rolling_min) / (rolling_max - rolling_min + 1e-9)) - 1
    
    def get_param_ranges(self) -> Dict[str, Any]:
        return {
            'window': np.arange(10, 490, 20),
            'thresholds': np.arange(0.1, 1.0, 0.1)
        }
    
    def validate_params(self, window: int, **kwargs) -> bool:
        return isinstance(window, (int, np.integer)) and window > 0

class MovingAverageDiffModel(BaseModel):
    """Moving average difference model"""
    
    def __init__(self):
        super().__init__("ma_diff")
    
    def generate_signal(self, data: pd.Series, window: int) -> pd.Series:
        """Generate moving average signal"""
        return data.rolling(window).mean()
    
    def get_param_ranges(self) -> Dict[str, Any]:
        return {
            'window': np.arange(10, 200, 10),
            'thresholds': np.arange(0.00, 0.05, 0.01)
        }

class DualMovingAverageModel(BaseModel):
    """Dual moving average model"""
    
    def __init__(self):
        super().__init__("ma_double")
    
    def generate_signal(self, data: pd.Series, short_window: int, long_window: int) -> pd.Series:
        """Generate dual moving average signal"""
        short_ma = data.rolling(short_window).mean()
        long_ma = data.rolling(long_window).mean()
        return short_ma - long_ma
    
    def get_param_ranges(self) -> Dict[str, Any]:
        return {
            'short_window': np.arange(10, 100, 10),
            'long_window': np.arange(50, 500, 50),
            'thresholds': np.arange(0.00, 0.05, 0.01)
        }
    
    def validate_params(self, short_window: int, long_window: int, **kwargs) -> bool:
        return (isinstance(short_window, (int, np.integer)) and 
                isinstance(long_window, (int, np.integer)) and 
                short_window < long_window and 
                short_window > 0)

class EWMADiffModel(BaseModel):
    """EWMA difference model"""
    
    def __init__(self):
        super().__init__("ewma_diff")
    
    def generate_signal(self, data: pd.Series, fast_span: int, slow_span: int, alpha: float = 0.1) -> pd.Series:
        """Generate EWMA difference signal"""
        fast_ewma = data.ewm(span=fast_span, adjust=False).mean()
        slow_ewma = data.ewm(span=slow_span, adjust=False).mean()
        return fast_ewma - slow_ewma
    
    def get_param_ranges(self) -> Dict[str, Any]:
        return {
            'fast_span': np.arange(10, 200, 10),
            'slow_span': np.arange(50, 500, 50),
            'alpha': np.arange(0.0, 1.0, 0.05),
            'thresholds': np.arange(0.00, 0.05, 0.01)
        }
    
    def validate_params(self, fast_span: int, slow_span: int, alpha: float = 0.1, **kwargs) -> bool:
        return (isinstance(fast_span, (int, np.integer)) and 
                isinstance(slow_span, (int, np.integer)) and 
                fast_span < slow_span and 
                0 <= alpha <= 1)

class RSIModel(BaseModel):
    """RSI model"""
    
    def __init__(self):
        super().__init__("rsi")
    
    def generate_signal(self, data: pd.Series, window: int) -> pd.Series:
        """Generate RSI signal"""
        try:
            rsi_values = tb.RSI(data.values, timeperiod=window)
            return pd.Series(rsi_values, index=data.index)
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}")
            return pd.Series(np.nan, index=data.index)
    
    def get_param_ranges(self) -> Dict[str, Any]:
        return {
            'window': np.arange(10, 50, 5),
            'thresholds': [(20, 80), (25, 75), (30, 70)]
        }

class PercentileModel(BaseModel):
    """Percentile ranking model"""
    
    def __init__(self):
        super().__init__("percentile")
    
    def generate_signal(self, data: pd.Series, window: int) -> pd.Series:
        """Generate percentile ranking signal"""
        def _rank(arr):
            if len(arr) == 0:
                return np.nan
            return np.searchsorted(np.sort(arr), arr[-1]) / len(arr)
        
        return data.rolling(window).apply(_rank, raw=True)
    
    def get_param_ranges(self) -> Dict[str, Any]:
        return {
            'window': np.arange(20, 500, 10),
            'thresholds': [(0.1, 0.9), (0.15, 0.85), (0.2, 0.8), (0.25, 0.75)]
        }

class MACDModel(BaseModel):
    """MACD model"""
    
    def __init__(self):
        super().__init__("macd")
    
    def generate_signal(self, data: pd.Series, fast_period: int = 12, 
                       slow_period: int = 26, signal_period: int = 9) -> pd.Series:
        """Generate MACD signal"""
        try:
            macd_line, signal_line, histogram = tb.MACD(
                data.values, 
                fastperiod=fast_period,
                slowperiod=slow_period, 
                signalperiod=signal_period
            )
            return pd.Series(histogram, index=data.index)
        except Exception as e:
            logger.warning(f"MACD calculation failed: {e}")
            return pd.Series(np.nan, index=data.index)
    
    def get_param_ranges(self) -> Dict[str, Any]:
        return {
            'fast_period': np.arange(8, 20, 2),
            'slow_period': np.arange(20, 35, 3),
            'signal_period': np.arange(7, 15, 2),
            'thresholds': np.arange(0.0, 0.05, 0.01)
        }

class ModelRegistry:
    """Registry for all available models"""
    
    def __init__(self):
        self._models = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register all default models"""
        models = [
            ZScoreModel(),
            MinMaxScalingModel(),
            MovingAverageDiffModel(),
            DualMovingAverageModel(),
            EWMADiffModel(),
            RSIModel(),
            PercentileModel(),
            MACDModel()
        ]
        
        for model in models:
            self.register_model(model)
    
    def register_model(self, model: BaseModel):
        """Register a new model"""
        self._models[model.name] = model
        logger.info(f"Registered model: {model.name}")
    
    def get_model(self, name: str) -> BaseModel:
        """Get a model by name"""
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found. Available models: {list(self._models.keys())}")
        return self._models[name]
    
    def get_all_models(self) -> Dict[str, BaseModel]:
        """Get all registered models"""
        return self._models.copy()
    
    def get_model_names(self) -> List[str]:
        """Get list of all model names"""
        return list(self._models.keys())

# Global model registry
MODEL_REGISTRY = ModelRegistry()

# Backward compatibility functions
def compute_zscore(series: pd.Series, window: int) -> pd.Series:
    """Backward compatibility function for Z-score"""
    return MODEL_REGISTRY.get_model('zscore')(series, window=window)

def compute_minmax(series: pd.Series, window: int) -> pd.Series:
    """Backward compatibility function for Min-Max scaling"""
    return MODEL_REGISTRY.get_model('min_max_scaling')(series, window=window)

def compute_ma_diff(series: pd.Series, window: int) -> pd.Series:
    """Backward compatibility function for MA diff"""
    return MODEL_REGISTRY.get_model('ma_diff')(series, window=window)

def compute_ma_double(series: pd.Series, short_w: int, long_w: int) -> pd.Series:
    """Backward compatibility function for dual MA"""
    return MODEL_REGISTRY.get_model('ma_double')(series, short_window=short_w, long_window=long_w)

def compute_ewma_diff(series: pd.Series, fast_span: int, slow_span: int, alpha: float = 0.1) -> pd.Series:
    """Backward compatibility function for EWMA diff"""
    return MODEL_REGISTRY.get_model('ewma_diff')(series, fast_span=fast_span, slow_span=slow_span, alpha=alpha)

def compute_RSI(series: pd.Series, window: int) -> pd.Series:
    """Backward compatibility function for RSI"""
    return MODEL_REGISTRY.get_model('rsi')(series, window=window)

def compute_percentile(series: pd.Series, window: int) -> pd.Series:
    """Backward compatibility function for percentile"""
    return MODEL_REGISTRY.get_model('percentile')(series, window=window)