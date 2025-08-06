import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import logging

warnings.filterwarnings('ignore')

class BasicAutoFeatureEngineer:
    """
    Simplified Automated Feature Engineering for Quantitative Trading
    (Version without TA-Lib dependency for testing)
    """
    
    def __init__(self, 
                 lookback_periods: List[int] = [5, 10, 20, 50],
                 min_ic_threshold: float = 0.02,
                 max_features: int = 20):
        self.lookback_periods = lookback_periods
        self.min_ic_threshold = min_ic_threshold
        self.max_features = max_features
        self.ic_scores = {}
        self.selected_features = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_returns(self, prices: pd.Series, periods: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """Calculate forward returns for different periods"""
        returns_df = pd.DataFrame(index=prices.index)
        
        for period in periods:
            returns_df[f'return_{period}d'] = prices.pct_change(period).shift(-period)
        
        return returns_df
    
    def generate_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate basic features without TA-Lib"""
        features = pd.DataFrame(index=data.index)
        
        # Basic price features
        features['price'] = data['close']
        features['log_price'] = np.log(data['close'])
        
        # Price momentum features
        for period in self.lookback_periods:
            features[f'momentum_{period}'] = data['close'].pct_change(period)
            features[f'log_momentum_{period}'] = np.log(data['close'] / data['close'].shift(period))
            
        # Simple moving averages and ratios
        for period in self.lookback_periods:
            sma = data['close'].rolling(period).mean()
            features[f'sma_{period}'] = sma
            features[f'price_to_sma_{period}'] = data['close'] / sma
            features[f'distance_to_sma_{period}'] = (data['close'] - sma) / sma
            
        # Volatility features
        for period in self.lookback_periods:
            features[f'volatility_{period}'] = data['close'].pct_change().rolling(period).std()
            
        # Simple RSI calculation
        for period in [14, 20]:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # High-Low ratio
        features['hl_ratio'] = data['high'] / data['low']
        features['hl_volatility'] = (data['high'] - data['low']) / data['close']
        
        # Volume features (if available)
        if 'volume' in data.columns:
            for period in self.lookback_periods:
                vol_sma = data['volume'].rolling(period).mean()
                features[f'volume_to_sma_{period}'] = data['volume'] / vol_sma
                features[f'volume_momentum_{period}'] = data['volume'].pct_change(period)
            
            # Price-Volume trend
            features['pv_trend'] = (data['close'].pct_change() * data['volume']).rolling(20).mean()
        
        return features
    
    def calculate_information_coefficient(self, features: pd.DataFrame, returns: pd.Series) -> Dict[str, dict]:
        """Calculate Information Coefficient for each feature"""
        ic_scores = {}
        
        for feature_name in features.columns:
            feature_values = features[feature_name].dropna()
            
            # Align feature values with returns
            common_index = feature_values.index.intersection(returns.index)
            if len(common_index) < 30:
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
                    'n_obs': len(clean_features),
                    'significant': p_value < 0.05
                }
        
        return ic_scores
    
    def select_best_features(self, ic_scores: Dict[str, dict]) -> List[str]:
        """Select best features based on IC analysis"""
        
        # Filter by minimum IC threshold and significance
        valid_features = {}
        for feature_name, scores in ic_scores.items():
            if (scores['abs_ic'] >= self.min_ic_threshold and 
                scores['significant'] and 
                scores['n_obs'] >= 30):
                valid_features[feature_name] = scores
        
        if not valid_features:
            self.logger.warning("No features meet the IC criteria, selecting top features by absolute IC")
            # Fallback: select top features by absolute IC
            sorted_features = sorted(ic_scores.items(), key=lambda x: x[1]['abs_ic'], reverse=True)
            return [f[0] for f in sorted_features[:min(self.max_features, len(sorted_features))]]
        
        # Sort by IC quality score
        sorted_features = sorted(valid_features.items(), 
                               key=lambda x: x[1]['abs_ic'], reverse=True)
        
        return [f[0] for f in sorted_features[:self.max_features]]
    
    def fit_transform(self, data: pd.DataFrame, return_periods: List[int] = [5]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Main method to generate and select features"""
        
        self.logger.info("Starting basic feature engineering...")
        
        # Calculate forward returns
        returns = self.calculate_returns(data['close'], return_periods)
        
        # Generate features
        self.logger.info("Generating basic features...")
        all_features = self.generate_basic_features(data)
        
        # Remove constant and infinite values
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.loc[:, all_features.std() > 0]
        
        self.logger.info(f"Generated {len(all_features.columns)} total features")
        
        # Calculate IC for primary return period
        primary_return = returns[f'return_{return_periods[0]}d']
        ic_scores = self.calculate_information_coefficient(all_features, primary_return)
        
        self.logger.info(f"Calculated IC for {len(ic_scores)} features")
        
        # Select best features
        selected_feature_names = self.select_best_features(ic_scores)
        self.logger.info(f"Selected {len(selected_feature_names)} features")
        
        # Store results
        self.ic_scores = ic_scores
        self.selected_features = selected_feature_names
        
        # Return selected features
        selected_features_df = all_features[selected_feature_names] if selected_feature_names else pd.DataFrame()
        
        return selected_features_df, all_features, ic_scores


def load_sample_data() -> pd.DataFrame:
    """Generate sample market data for testing"""
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic price data with trends and volatility
    n_days = len(dates)
    
    # Base trend
    trend = np.linspace(0, 0.3, n_days)  # 30% growth over period
    
    # Random walk component
    random_returns = np.random.normal(0.0005, 0.02, n_days)
    
    # Add some autocorrelation for realism
    for i in range(1, n_days):
        random_returns[i] += 0.1 * random_returns[i-1]
    
    # Combine trend and random walk
    log_prices = trend + np.cumsum(random_returns)
    prices = 100 * np.exp(log_prices)
    
    # Generate OHLC data
    daily_returns = np.diff(np.concatenate([[0], log_prices]))
    volatility = np.abs(daily_returns) + np.random.normal(0, 0.005, n_days)
    
    high_prices = prices * (1 + volatility * np.random.uniform(0.3, 0.7, n_days))
    low_prices = prices * (1 - volatility * np.random.uniform(0.3, 0.7, n_days))
    open_prices = prices + np.random.normal(0, prices * 0.01)
    
    # Generate volume data
    volume = np.random.lognormal(10, 1, n_days)
    
    data = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': prices,
        'volume': volume
    })
    
    data.set_index('date', inplace=True)
    return data


def main():
    """Test the basic feature engineering system"""
    
    print("=" * 60)
    print("BASIC AUTO FEATURE ENGINEERING TEST")
    print("=" * 60)
    
    # Load sample data
    print("Loading sample data...")
    data = load_sample_data()
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Initialize feature engineer
    feature_engineer = BasicAutoFeatureEngineer(
        lookback_periods=[5, 10, 20, 50],
        min_ic_threshold=0.01,  # Lower threshold for demo
        max_features=15
    )
    
    # Generate and select features
    print("\nGenerating features and calculating IC...")
    selected_features, all_features, ic_scores = feature_engineer.fit_transform(
        data, 
        return_periods=[5, 10]
    )
    
    print(f"\nResults:")
    print(f"- Generated {len(all_features.columns)} total features")
    print(f"- Selected {len(selected_features.columns)} best features")
    print(f"- Calculated IC for {len(ic_scores)} features")
    
    # Show IC analysis
    print(f"\nIC Analysis Summary:")
    if ic_scores:
        ic_values = [scores['ic'] for scores in ic_scores.values()]
        abs_ic_values = [scores['abs_ic'] for scores in ic_scores.values()]
        significant_count = sum(1 for scores in ic_scores.values() if scores['significant'])
        
        print(f"- Average IC: {np.mean(ic_values):.4f}")
        print(f"- Average Absolute IC: {np.mean(abs_ic_values):.4f}")
        print(f"- Max Absolute IC: {np.max(abs_ic_values):.4f}")
        print(f"- Significant features: {significant_count}/{len(ic_scores)}")
    
    # Show top features
    print(f"\nTop Features by IC:")
    if ic_scores:
        sorted_features = sorted(ic_scores.items(), key=lambda x: x[1]['abs_ic'], reverse=True)
        for i, (feature_name, scores) in enumerate(sorted_features[:10]):
            status = "✓" if feature_name in feature_engineer.selected_features else " "
            significance = "*" if scores['significant'] else " "
            print(f"{status} {i+1:2d}. {feature_name:25s} IC: {scores['ic']:7.4f} ({scores['abs_ic']:6.4f}) {significance}")
    
    # Save results
    if not selected_features.empty:
        selected_features.to_csv('/workspace/basic_selected_features.csv')
        print(f"\nSelected features saved to: basic_selected_features.csv")
    
    # Create IC summary
    if ic_scores:
        ic_df = pd.DataFrame(ic_scores).T
        ic_df = ic_df.sort_values('abs_ic', ascending=False)
        ic_df.to_csv('/workspace/basic_ic_analysis.csv')
        print(f"IC analysis saved to: basic_ic_analysis.csv")
    
    print("\n" + "=" * 60)
    print("BASIC TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return selected_features, all_features, ic_scores


if __name__ == "__main__":
    main()