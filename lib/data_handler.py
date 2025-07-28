"""
Data handling module for the backtesting system.
Handles data loading, validation, and splitting for backtesting and forward testing.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import logging
from .config import BACKTEST_CONFIG

logger = logging.getLogger(__name__)

class DataHandler:
    """Handles all data operations for the backtesting system"""
    
    def __init__(self, config=None):
        self.config = config or BACKTEST_CONFIG
        self.data = None
        self.splits = {}
        
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from various file formats
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded and validated DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, **kwargs)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, **kwargs)
            elif file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
            logger.info(f"Loaded data from {file_path}: {df.shape}")
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
            
        # Validate and process the data
        df = self._validate_and_process_data(df)
        self.data = df
        return df
    
    def _validate_and_process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and process the loaded data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Processed and validated DataFrame
        """
        # Check required columns
        required_cols = [self.config.date_column, self.config.price_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Parse datetime column
        if not pd.api.types.is_datetime64_any_dtype(df[self.config.date_column]):
            df[self.config.date_column] = pd.to_datetime(df[self.config.date_column])
            
        # Sort by datetime
        df = df.sort_values(self.config.date_column).reset_index(drop=True)
        
        # Set datetime as index
        df.set_index(self.config.date_column, inplace=True)
        
        # Check for missing values in price column
        if df[self.config.price_column].isna().any():
            logger.warning("Found missing values in price column, forward filling...")
            df[self.config.price_column] = df[self.config.price_column].fillna(method='ffill')
            
        # Remove any remaining rows with missing price data
        df = df.dropna(subset=[self.config.price_column])
        
        logger.info(f"Data validation complete. Final shape: {df.shape}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def split_data(self, df: Optional[pd.DataFrame] = None, 
                   method: str = 'ratio') -> Dict[str, pd.DataFrame]:
        """
        Split data into backtest, validation, and forward test sets
        
        Args:
            df: DataFrame to split (uses self.data if None)
            method: Split method ('ratio' or 'date')
            
        Returns:
            Dictionary with split data
        """
        if df is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            df = self.data
            
        if method == 'ratio':
            splits = self._split_by_ratio(df)
        elif method == 'date':
            splits = self._split_by_date(df)
        else:
            raise ValueError(f"Unknown split method: {method}")
            
        self.splits = splits
        
        # Log split information
        for name, data in splits.items():
            logger.info(f"{name} set: {data.shape[0]} rows, "
                       f"{data.index.min()} to {data.index.max()}")
            
        return splits
    
    def _split_by_ratio(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data by ratio"""
        n = len(df)
        
        backtest_end = int(n * self.config.backtest_ratio)
        val_end = int(n * (self.config.backtest_ratio + self.config.validation_ratio))
        
        return {
            'backtest': df.iloc[:backtest_end].copy(),
            'validation': df.iloc[backtest_end:val_end].copy(),
            'forward': df.iloc[val_end:].copy(),
            'full': df.copy()
        }
    
    def _split_by_date(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data by specific dates (to be implemented based on needs)"""
        # This can be customized based on specific date requirements
        return self._split_by_ratio(df)
    
    def get_factor_data(self, factor_name: str, split: str = 'full') -> pd.Series:
        """
        Get factor data for a specific split
        
        Args:
            factor_name: Name of the factor column
            split: Which data split to use
            
        Returns:
            Factor data as Series
        """
        if not self.splits:
            raise ValueError("Data not split yet. Call split_data() first.")
            
        if split not in self.splits:
            raise ValueError(f"Unknown split: {split}")
            
        if factor_name not in self.splits[split].columns:
            raise ValueError(f"Factor '{factor_name}' not found in data")
            
        return self.splits[split][factor_name]
    
    def get_price_data(self, split: str = 'full') -> pd.Series:
        """
        Get price data for a specific split
        
        Args:
            split: Which data split to use
            
        Returns:
            Price data as Series
        """
        if not self.splits:
            raise ValueError("Data not split yet. Call split_data() first.")
            
        if split not in self.splits:
            raise ValueError(f"Unknown split: {split}")
            
        return self.splits[split][self.config.price_column]
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data"""
        if self.data is None:
            return {"status": "No data loaded"}
            
        info = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "date_range": {
                "start": self.data.index.min(),
                "end": self.data.index.max()
            },
            "missing_values": self.data.isna().sum().to_dict(),
            "splits_available": bool(self.splits)
        }
        
        if self.splits:
            info["split_info"] = {
                name: {
                    "shape": data.shape,
                    "date_range": (data.index.min(), data.index.max())
                }
                for name, data in self.splits.items()
            }
            
        return info
    
    def save_splits(self, output_dir: str):
        """Save data splits to files"""
        if not self.splits:
            raise ValueError("No splits to save. Call split_data() first.")
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, data in self.splits.items():
            file_path = output_path / f"data_{name}.csv"
            data.to_csv(file_path)
            logger.info(f"Saved {name} split to {file_path}")