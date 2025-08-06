import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression
import talib
from itertools import combinations
import logging

warnings.filterwarnings('ignore')

class AutoFeatureEngineer:
    """
    Automated Feature Engineering for Quantitative Trading with Information Coefficient Analysis
    
    This class provides comprehensive feature generation, evaluation using Information Coefficient,
    and automatic feature selection for trading strategies.
    """
    
    def __init__(self, 
                 lookback_periods: List[int] = [5, 10, 20, 50, 100],
                 min_ic_threshold: float = 0.02,
                 max_features: int = 50,
                 correlation_threshold: float = 0.8):
        """
        Initialize the AutoFeatureEngineer
        
        Args:
            lookback_periods: List of periods for technical indicators
            min_ic_threshold: Minimum IC absolute value to keep a feature
            max_features: Maximum number of features to select
            correlation_threshold: Maximum correlation between features
        """
        self.lookback_periods = lookback_periods
        self.min_ic_threshold = min_ic_threshold
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.features_metadata = {}
        self.ic_scores = {}
        self.selected_features = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_returns(self, prices: pd.Series, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """Calculate forward returns for different periods"""
        returns_df = pd.DataFrame(index=prices.index)
        
        for period in periods:
            returns_df[f'return_{period}d'] = prices.pct_change(period).shift(-period)
        
        return returns_df
    
    def generate_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate price-based features"""
        features = pd.DataFrame(index=data.index)
        
        # Basic price features
        features['price'] = data['close']
        features['log_price'] = np.log(data['close'])
        
        # Price momentum features
        for period in self.lookback_periods:
            features[f'momentum_{period}'] = data['close'].pct_change(period)
            features[f'log_momentum_{period}'] = np.log(data['close'] / data['close'].shift(period))
            
        # Price mean reversion features
        for period in self.lookback_periods:
            ma = data['close'].rolling(period).mean()
            features[f'price_to_ma_{period}'] = data['close'] / ma
            features[f'distance_to_ma_{period}'] = (data['close'] - ma) / ma
            
        # Price volatility features
        for period in self.lookback_periods:
            features[f'volatility_{period}'] = data['close'].pct_change().rolling(period).std()
            features[f'log_volatility_{period}'] = np.log(features[f'volatility_{period}'])
            
        return features
    
    def generate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based features"""
        features = pd.DataFrame(index=data.index)
        
        if 'volume' not in data.columns:
            return features
        
        # Volume momentum
        for period in self.lookback_periods:
            features[f'volume_momentum_{period}'] = data['volume'].pct_change(period)
            
        # Volume moving averages
        for period in self.lookback_periods:
            vol_ma = data['volume'].rolling(period).mean()
            features[f'volume_to_ma_{period}'] = data['volume'] / vol_ma
            
        # Price-Volume relationship
        features['pv_trend'] = (data['close'].pct_change() * data['volume']).rolling(20).mean()
        
        # On-Balance Volume
        features['obv'] = (data['volume'] * np.sign(data['close'].diff())).cumsum()
        features['obv_momentum_20'] = features['obv'].pct_change(20)
        
        return features
    
    def generate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicator features using TA-Lib"""
        features = pd.DataFrame(index=data.index)
        
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values if 'volume' in data.columns else None
        
        try:
            # Trend indicators
            for period in [14, 20, 50]:
                features[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)
                features[f'cci_{period}'] = talib.CCI(high, low, close, timeperiod=period)
                
            # Moving averages
            for period in self.lookback_periods:
                features[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
                features[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
                features[f'price_to_sma_{period}'] = close / features[f'sma_{period}']
                features[f'price_to_ema_{period}'] = close / features[f'ema_{period}']
            
            # Bollinger Bands
            for period in [20, 50]:
                upper, middle, lower = talib.BBANDS(close, timeperiod=period)
                features[f'bb_upper_{period}'] = upper
                features[f'bb_lower_{period}'] = lower
                features[f'bb_position_{period}'] = (close - lower) / (upper - lower)
                features[f'bb_width_{period}'] = (upper - lower) / middle
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_hist
            
            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close)
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd
            
            # Williams %R
            features['williams_r'] = talib.WILLR(high, low, close)
            
            # Average True Range
            features['atr'] = talib.ATR(high, low, close)
            features['atr_ratio'] = features['atr'] / close
            
        except Exception as e:
            self.logger.warning(f"Error generating technical indicators: {e}")
        
        return features
    
    def generate_cross_sectional_features(self, data: pd.DataFrame, universe_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate cross-sectional ranking and percentile features"""
        features = pd.DataFrame(index=data.index)
        
        if not universe_data:
            return features
        
        # For each date, calculate cross-sectional ranks
        all_symbols = list(universe_data.keys())
        
        for feature_name in ['momentum_20', 'volatility_20', 'rsi_14']:
            cross_sectional_data = pd.DataFrame()
            
            for symbol, symbol_data in universe_data.items():
                if feature_name in symbol_data.columns:
                    cross_sectional_data[symbol] = symbol_data[feature_name]
            
            if not cross_sectional_data.empty:
                # Calculate percentile ranks
                percentile_ranks = cross_sectional_data.rank(axis=1, pct=True)
                
                # Get the percentile rank for current symbol
                symbol_name = data.get('symbol', list(universe_data.keys())[0])
                if symbol_name in percentile_ranks.columns:
                    features[f'{feature_name}_percentile'] = percentile_ranks[symbol_name]
        
        return features
    
    def generate_interaction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features between existing features"""
        interaction_features = pd.DataFrame(index=features_df.index)
        
        # Select key features for interactions
        key_features = [col for col in features_df.columns if any(x in col for x in 
                       ['momentum', 'volatility', 'rsi', 'price_to_ma'])][:10]
        
        # Generate pairwise interactions
        for feat1, feat2 in combinations(key_features, 2):
            if feat1 in features_df.columns and feat2 in features_df.columns:
                # Multiplication
                interaction_features[f'{feat1}_x_{feat2}'] = features_df[feat1] * features_df[feat2]
                
                # Ratio (avoid division by zero)
                denominator = features_df[feat2].replace(0, np.nan)
                interaction_features[f'{feat1}_div_{feat2}'] = features_df[feat1] / denominator
        
        return interaction_features
    
    def calculate_information_coefficient(self, features: pd.DataFrame, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate Information Coefficient (IC) for each feature
        
        IC measures the correlation between feature values and forward returns
        """
        ic_scores = {}
        
        for feature_name in features.columns:
            feature_values = features[feature_name].dropna()
            
            # Align feature values with returns
            common_index = feature_values.index.intersection(returns.index)
            if len(common_index) < 30:  # Minimum observations
                continue
                
            aligned_features = feature_values.loc[common_index]
            aligned_returns = returns.loc[common_index]
            
            # Remove any remaining NaN values
            mask = ~(aligned_features.isna() | aligned_returns.isna())
            if mask.sum() < 30:
                continue
            
            clean_features = aligned_features[mask]
            clean_returns = aligned_returns[mask]
            
            # Calculate Spearman correlation (rank IC)
            ic, p_value = stats.spearmanr(clean_features, clean_returns)
            
            if not np.isnan(ic):
                ic_scores[feature_name] = {
                    'ic': ic,
                    'abs_ic': abs(ic),
                    'p_value': p_value,
                    'n_obs': len(clean_features)
                }
        
        return ic_scores
    
    def calculate_ic_statistics(self, features: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive IC statistics across multiple return periods"""
        ic_stats = []
        
        for return_col in returns.columns:
            ic_scores = self.calculate_information_coefficient(features, returns[return_col])
            
            for feature_name, scores in ic_scores.items():
                ic_stats.append({
                    'feature': feature_name,
                    'return_period': return_col,
                    'ic': scores['ic'],
                    'abs_ic': scores['abs_ic'],
                    'p_value': scores['p_value'],
                    'n_obs': scores['n_obs'],
                    'significant': scores['p_value'] < 0.05
                })
        
        return pd.DataFrame(ic_stats)
    
    def select_features_by_ic(self, ic_stats: pd.DataFrame) -> List[str]:
        """Select best features based on IC analysis"""
        
        # Filter by minimum IC threshold and significance
        filtered_stats = ic_stats[
            (ic_stats['abs_ic'] >= self.min_ic_threshold) & 
            (ic_stats['significant'] == True) &
            (ic_stats['n_obs'] >= 50)
        ].copy()
        
        if filtered_stats.empty:
            self.logger.warning("No features meet the IC threshold criteria")
            return []
        
        # Calculate average IC across return periods for each feature
        feature_ic_summary = filtered_stats.groupby('feature').agg({
            'abs_ic': ['mean', 'std'],
            'significant': 'sum',
            'n_obs': 'mean'
        }).round(4)
        
        feature_ic_summary.columns = ['avg_abs_ic', 'std_abs_ic', 'significant_count', 'avg_obs']
        feature_ic_summary['ic_stability'] = feature_ic_summary['avg_abs_ic'] / (feature_ic_summary['std_abs_ic'] + 0.001)
        
        # Rank features by IC quality
        feature_ic_summary['rank_score'] = (
            0.5 * feature_ic_summary['avg_abs_ic'] + 
            0.3 * feature_ic_summary['ic_stability'] + 
            0.2 * (feature_ic_summary['significant_count'] / filtered_stats['return_period'].nunique())
        )
        
        # Select top features
        top_features = feature_ic_summary.nlargest(self.max_features * 2, 'rank_score')
        
        return top_features.index.tolist()
    
    def remove_correlated_features(self, features: pd.DataFrame, selected_features: List[str]) -> List[str]:
        """Remove highly correlated features to avoid multicollinearity"""
        
        feature_subset = features[selected_features].dropna()
        
        if feature_subset.empty:
            return selected_features
        
        # Calculate correlation matrix
        corr_matrix = feature_subset.corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns if 
                  any(upper_triangle[column] > self.correlation_threshold)]
        
        final_features = [f for f in selected_features if f not in to_drop]
        
        self.logger.info(f"Removed {len(to_drop)} highly correlated features")
        
        return final_features[:self.max_features]
    
    def fit_transform(self, 
                     data: pd.DataFrame, 
                     universe_data: Optional[Dict[str, pd.DataFrame]] = None,
                     return_periods: List[int] = [1, 5, 10]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Main method to generate features, calculate IC, and select best features
        
        Args:
            data: Price data with OHLCV columns
            universe_data: Dictionary of data for all symbols in universe (for cross-sectional features)
            return_periods: List of forward return periods to calculate
            
        Returns:
            Tuple of (selected_features_df, all_features_df, ic_analysis_df)
        """
        
        self.logger.info("Starting automated feature engineering...")
        
        # Calculate forward returns
        returns = self.calculate_returns(data['close'], return_periods)
        
        # Generate all feature categories
        self.logger.info("Generating price-based features...")
        price_features = self.generate_price_features(data)
        
        self.logger.info("Generating volume-based features...")
        volume_features = self.generate_volume_features(data)
        
        self.logger.info("Generating technical indicators...")
        technical_features = self.generate_technical_indicators(data)
        
        self.logger.info("Generating cross-sectional features...")
        cross_sectional_features = self.generate_cross_sectional_features(data, universe_data or {})
        
        # Combine all features
        all_features = pd.concat([
            price_features,
            volume_features, 
            technical_features,
            cross_sectional_features
        ], axis=1)
        
        # Generate interaction features (limited to avoid explosion)
        self.logger.info("Generating interaction features...")
        interaction_features = self.generate_interaction_features(all_features)
        all_features = pd.concat([all_features, interaction_features], axis=1)
        
        # Remove constant and infinite values
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.loc[:, all_features.std() > 0]
        
        self.logger.info(f"Generated {len(all_features.columns)} total features")
        
        # Calculate IC statistics
        self.logger.info("Calculating Information Coefficient statistics...")
        ic_analysis = self.calculate_ic_statistics(all_features, returns)
        
        # Select features based on IC
        self.logger.info("Selecting features based on IC analysis...")
        ic_selected_features = self.select_features_by_ic(ic_analysis)
        
        # Remove highly correlated features
        self.logger.info("Removing highly correlated features...")
        final_selected_features = self.remove_correlated_features(all_features, ic_selected_features)
        
        self.selected_features = final_selected_features
        self.ic_scores = ic_analysis
        
        self.logger.info(f"Final feature selection: {len(final_selected_features)} features")
        
        # Return selected features dataframe
        selected_features_df = all_features[final_selected_features]
        
        return selected_features_df, all_features, ic_analysis
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        """Generate a comprehensive feature importance report"""
        
        if self.ic_scores.empty:
            return pd.DataFrame()
        
        # Create summary by feature
        feature_summary = self.ic_scores.groupby('feature').agg({
            'ic': ['mean', 'std'],
            'abs_ic': ['mean', 'max'],
            'significant': 'sum',
            'n_obs': 'mean'
        }).round(4)
        
        feature_summary.columns = [
            'avg_ic', 'ic_std', 'avg_abs_ic', 'max_abs_ic', 
            'significant_periods', 'avg_observations'
        ]
        
        # Add selection status
        feature_summary['selected'] = feature_summary.index.isin(self.selected_features)
        
        # Sort by average absolute IC
        feature_summary = feature_summary.sort_values('avg_abs_ic', ascending=False)
        
        return feature_summary
    
    def plot_ic_analysis(self, save_path: Optional[str] = None):
        """Plot IC analysis results"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if self.ic_scores.empty:
                print("No IC scores available for plotting")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # IC distribution
            axes[0, 0].hist(self.ic_scores['abs_ic'], bins=50, alpha=0.7)
            axes[0, 0].axvline(self.min_ic_threshold, color='red', linestyle='--', 
                              label=f'Threshold: {self.min_ic_threshold}')
            axes[0, 0].set_xlabel('Absolute IC')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Absolute IC Values')
            axes[0, 0].legend()
            
            # IC by return period
            ic_by_period = self.ic_scores.groupby('return_period')['abs_ic'].mean()
            axes[0, 1].bar(ic_by_period.index, ic_by_period.values)
            axes[0, 1].set_xlabel('Return Period')
            axes[0, 1].set_ylabel('Average Absolute IC')
            axes[0, 1].set_title('IC by Return Period')
            
            # Top features IC
            top_features = self.get_feature_importance_report().head(20)
            axes[1, 0].barh(range(len(top_features)), top_features['avg_abs_ic'])
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features.index, fontsize=8)
            axes[1, 0].set_xlabel('Average Absolute IC')
            axes[1, 0].set_title('Top 20 Features by IC')
            
            # Selected vs non-selected features
            selected_ic = self.ic_scores[self.ic_scores['feature'].isin(self.selected_features)]['abs_ic']
            non_selected_ic = self.ic_scores[~self.ic_scores['feature'].isin(self.selected_features)]['abs_ic']
            
            axes[1, 1].boxplot([selected_ic, non_selected_ic], labels=['Selected', 'Not Selected'])
            axes[1, 1].set_ylabel('Absolute IC')
            axes[1, 1].set_title('IC Distribution: Selected vs Non-Selected Features')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            print("Matplotlib and seaborn required for plotting. Install with: pip install matplotlib seaborn")


# Example usage and utility functions
def load_sample_data() -> pd.DataFrame:
    """Load or generate sample market data for testing"""
    # Generate sample data - replace with your actual data loading logic
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        'open': prices + np.random.normal(0, 0.5, len(dates)),
        'volume': np.random.lognormal(10, 1, len(dates))
    })
    
    data.set_index('date', inplace=True)
    return data


def main():
    """Example usage of AutoFeatureEngineer"""
    
    # Load sample data
    print("Loading sample data...")
    data = load_sample_data()
    
    # Initialize feature engineer
    feature_engineer = AutoFeatureEngineer(
        lookback_periods=[5, 10, 20, 50],
        min_ic_threshold=0.01,
        max_features=30,
        correlation_threshold=0.8
    )
    
    # Generate features and analyze IC
    print("Generating features and analyzing Information Coefficient...")
    selected_features, all_features, ic_analysis = feature_engineer.fit_transform(data)
    
    # Print results
    print(f"\nGenerated {len(all_features.columns)} total features")
    print(f"Selected {len(selected_features.columns)} features based on IC analysis")
    
    # Show feature importance report
    importance_report = feature_engineer.get_feature_importance_report()
    print("\nTop 10 Features by IC:")
    print(importance_report.head(10)[['avg_abs_ic', 'significant_periods', 'selected']])
    
    # Show IC analysis summary
    print(f"\nIC Analysis Summary:")
    print(f"Features above IC threshold: {len(ic_analysis[ic_analysis['abs_ic'] >= feature_engineer.min_ic_threshold])}")
    print(f"Significant features (p < 0.05): {len(ic_analysis[ic_analysis['significant']])}")
    print(f"Average IC: {ic_analysis['abs_ic'].mean():.4f}")
    print(f"Max IC: {ic_analysis['abs_ic'].max():.4f}")
    
    # Save results
    selected_features.to_csv('/workspace/selected_features.csv')
    importance_report.to_csv('/workspace/feature_importance_report.csv')
    ic_analysis.to_csv('/workspace/ic_analysis.csv')
    
    print("\nResults saved to:")
    print("- selected_features.csv")
    print("- feature_importance_report.csv") 
    print("- ic_analysis.csv")


if __name__ == "__main__":
    main()