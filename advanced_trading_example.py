import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
from auto_feature_engineer import AutoFeatureEngineer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class QuantTradingPipeline:
    """
    Advanced Quantitative Trading Pipeline with Automated Feature Engineering
    
    This class demonstrates how to use the AutoFeatureEngineer in a complete
    quantitative trading workflow including backtesting and performance analysis.
    """
    
    def __init__(self, 
                 feature_engineer: AutoFeatureEngineer,
                 prediction_horizon: int = 5,
                 rebalance_frequency: int = 20):
        """
        Initialize the trading pipeline
        
        Args:
            feature_engineer: Instance of AutoFeatureEngineer
            prediction_horizon: Days ahead to predict returns
            rebalance_frequency: How often to retrain models (days)
        """
        self.feature_engineer = feature_engineer
        self.prediction_horizon = prediction_horizon
        self.rebalance_frequency = rebalance_frequency
        self.models = {}
        self.feature_importance_history = []
        self.predictions_history = []
        
    def load_multiple_assets_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple assets - replace this with your actual data source
        
        For demo purposes, this generates synthetic data for multiple symbols
        """
        universe_data = {}
        
        for symbol in symbols:
            # Generate synthetic data for each symbol
            dates = pd.date_range(start_date, end_date, freq='D')
            np.random.seed(hash(symbol) % 2**32)  # Different seed for each symbol
            
            # Generate correlated returns (some market factor)
            market_returns = np.random.normal(0.0005, 0.015, len(dates))
            idiosyncratic_returns = np.random.normal(0, 0.01, len(dates))
            
            # Combine market and idiosyncratic returns
            total_returns = 0.7 * market_returns + 0.3 * idiosyncratic_returns
            prices = 100 * np.exp(np.cumsum(total_returns))
            
            data = pd.DataFrame({
                'date': dates,
                'close': prices,
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'open': prices + np.random.normal(0, 0.5, len(dates)),
                'volume': np.random.lognormal(10, 1, len(dates)),
                'symbol': symbol
            })
            
            data.set_index('date', inplace=True)
            universe_data[symbol] = data
            
        return universe_data
    
    def generate_features_for_universe(self, universe_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate features for all assets in the universe"""
        features_universe = {}
        
        print("Generating features for universe...")
        for symbol, data in universe_data.items():
            print(f"Processing {symbol}...")
            
            # Generate features for this symbol
            selected_features, all_features, ic_analysis = self.feature_engineer.fit_transform(
                data, 
                universe_data=universe_data,
                return_periods=[self.prediction_horizon]
            )
            
            # Store results
            features_universe[symbol] = {
                'features': selected_features,
                'all_features': all_features,
                'ic_analysis': ic_analysis,
                'returns': self.feature_engineer.calculate_returns(
                    data['close'], 
                    [self.prediction_horizon]
                )
            }
            
        return features_universe
    
    def create_training_dataset(self, features_universe: Dict[str, pd.DataFrame]) -> tuple:
        """Create a unified training dataset from all symbols"""
        
        all_features = []
        all_returns = []
        all_symbols = []
        all_dates = []
        
        for symbol, data in features_universe.items():
            features = data['features']
            returns = data['returns'][f'return_{self.prediction_horizon}d']
            
            # Align features and returns
            common_index = features.index.intersection(returns.index)
            if len(common_index) < 50:  # Minimum data requirement
                continue
                
            aligned_features = features.loc[common_index]
            aligned_returns = returns.loc[common_index]
            
            # Remove NaN values
            mask = ~(aligned_features.isna().any(axis=1) | aligned_returns.isna())
            if mask.sum() < 30:
                continue
                
            clean_features = aligned_features[mask]
            clean_returns = aligned_returns[mask]
            
            all_features.append(clean_features)
            all_returns.extend(clean_returns.values)
            all_symbols.extend([symbol] * len(clean_features))
            all_dates.extend(clean_features.index.tolist())
        
        if not all_features:
            raise ValueError("No valid training data found")
            
        # Combine all features
        combined_features = pd.concat(all_features, axis=0)
        combined_returns = pd.Series(all_returns, index=combined_features.index)
        
        # Create metadata DataFrame
        metadata = pd.DataFrame({
            'symbol': all_symbols,
            'date': all_dates
        }, index=combined_features.index)
        
        return combined_features, combined_returns, metadata
    
    def train_models(self, features: pd.DataFrame, returns: pd.Series) -> Dict:
        """Train multiple prediction models"""
        
        # Remove any remaining NaN values
        mask = ~(features.isna().any(axis=1) | returns.isna())
        clean_features = features[mask]
        clean_returns = returns[mask]
        
        if len(clean_features) < 100:
            raise ValueError("Insufficient training data")
        
        models = {}
        
        # Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(clean_features, clean_returns)
        models['random_forest'] = rf_model
        
        # Linear Regression
        print("Training Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(clean_features, clean_returns)
        models['linear_regression'] = lr_model
        
        return models
    
    def backtest_strategy(self, 
                         features_universe: Dict[str, pd.DataFrame],
                         start_date: str,
                         end_date: str) -> pd.DataFrame:
        """
        Perform time-series cross-validation backtest
        """
        
        print("Starting backtest...")
        
        # Create combined dataset
        all_features, all_returns, metadata = self.create_training_dataset(features_universe)
        
        # Prepare for time series split
        dates = sorted(metadata['date'].unique())
        backtest_results = []
        
        # Use expanding window for training
        min_train_size = 252  # 1 year of daily data
        
        for i in range(min_train_size, len(dates), self.rebalance_frequency):
            train_end_date = dates[i]
            test_start_date = dates[min(i + 1, len(dates) - 1)]
            test_end_date = dates[min(i + self.rebalance_frequency, len(dates) - 1)]
            
            print(f"Training period: {dates[0]} to {train_end_date}")
            print(f"Testing period: {test_start_date} to {test_end_date}")
            
            # Split data
            train_mask = metadata['date'] <= train_end_date
            test_mask = (metadata['date'] >= test_start_date) & (metadata['date'] <= test_end_date)
            
            train_features = all_features[train_mask]
            train_returns = all_returns[train_mask]
            test_features = all_features[test_mask]
            test_returns = all_returns[test_mask]
            test_metadata = metadata[test_mask]
            
            if len(train_features) < 100 or len(test_features) < 10:
                continue
            
            # Train models
            try:
                models = self.train_models(train_features, train_returns)
            except Exception as e:
                print(f"Error training models: {e}")
                continue
            
            # Make predictions
            for model_name, model in models.items():
                try:
                    predictions = model.predict(test_features)
                    
                    # Calculate metrics
                    mse = mean_squared_error(test_returns, predictions)
                    mae = mean_absolute_error(test_returns, predictions)
                    
                    # Calculate IC for predictions
                    ic_pred = np.corrcoef(predictions, test_returns)[0, 1]
                    
                    # Store results
                    for j, (idx, row) in enumerate(test_metadata.iterrows()):
                        backtest_results.append({
                            'date': row['date'],
                            'symbol': row['symbol'],
                            'model': model_name,
                            'prediction': predictions[j],
                            'actual_return': test_returns.iloc[j],
                            'mse': mse,
                            'mae': mae,
                            'ic': ic_pred
                        })
                        
                except Exception as e:
                    print(f"Error making predictions with {model_name}: {e}")
                    continue
        
        return pd.DataFrame(backtest_results)
    
    def analyze_results(self, backtest_results: pd.DataFrame) -> Dict:
        """Analyze backtest results and generate performance metrics"""
        
        if backtest_results.empty:
            return {}
        
        analysis = {}
        
        # Overall performance by model
        model_performance = backtest_results.groupby('model').agg({
            'ic': 'mean',
            'mse': 'mean',
            'mae': 'mean',
            'prediction': 'count'
        }).round(4)
        model_performance.columns = ['avg_ic', 'avg_mse', 'avg_mae', 'n_predictions']
        
        analysis['model_performance'] = model_performance
        
        # IC time series
        ic_timeseries = backtest_results.groupby(['date', 'model'])['ic'].first().unstack(fill_value=0)
        analysis['ic_timeseries'] = ic_timeseries
        
        # Feature importance (from Random Forest if available)
        if 'random_forest' in backtest_results['model'].values:
            # This would need to be stored during training - simplified here
            analysis['feature_importance'] = "Feature importance analysis would go here"
        
        # Prediction accuracy by symbol
        symbol_performance = backtest_results.groupby(['symbol', 'model']).agg({
            'ic': 'mean',
            'prediction': 'count'
        }).round(4)
        
        analysis['symbol_performance'] = symbol_performance
        
        return analysis
    
    def plot_results(self, backtest_results: pd.DataFrame, analysis: Dict):
        """Create visualization of backtest results"""
        
        if backtest_results.empty:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Model performance comparison
        model_perf = analysis['model_performance']
        axes[0, 0].bar(model_perf.index, model_perf['avg_ic'])
        axes[0, 0].set_title('Average IC by Model')
        axes[0, 0].set_ylabel('Information Coefficient')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # IC time series
        if 'ic_timeseries' in analysis and not analysis['ic_timeseries'].empty:
            ic_ts = analysis['ic_timeseries']
            for col in ic_ts.columns:
                axes[0, 1].plot(ic_ts.index, ic_ts[col], label=col, alpha=0.7)
            axes[0, 1].set_title('IC Over Time')
            axes[0, 1].set_ylabel('Information Coefficient')
            axes[0, 1].legend()
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Prediction vs Actual scatter
        sample_data = backtest_results.sample(min(1000, len(backtest_results)))
        for model in sample_data['model'].unique():
            model_data = sample_data[sample_data['model'] == model]
            axes[1, 0].scatter(model_data['prediction'], model_data['actual_return'], 
                             alpha=0.5, label=model)
        axes[1, 0].plot(axes[1, 0].get_xlim(), axes[1, 0].get_ylim(), 'k--', alpha=0.75, zorder=0)
        axes[1, 0].set_xlabel('Predicted Return')
        axes[1, 0].set_ylabel('Actual Return')
        axes[1, 0].set_title('Predictions vs Actual Returns')
        axes[1, 0].legend()
        
        # MSE by model
        axes[1, 1].bar(model_perf.index, model_perf['avg_mse'])
        axes[1, 1].set_title('Average MSE by Model')
        axes[1, 1].set_ylabel('Mean Squared Error')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('/workspace/backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_complete_pipeline():
    """Run the complete quantitative trading pipeline"""
    
    print("=" * 60)
    print("QUANTITATIVE TRADING PIPELINE WITH AUTO FEATURE ENGINEERING")
    print("=" * 60)
    
    # Initialize feature engineer
    feature_engineer = AutoFeatureEngineer(
        lookback_periods=[5, 10, 20, 50],
        min_ic_threshold=0.015,  # Lower threshold for demo
        max_features=25,
        correlation_threshold=0.8
    )
    
    # Initialize trading pipeline
    pipeline = QuantTradingPipeline(
        feature_engineer=feature_engineer,
        prediction_horizon=5,  # Predict 5-day returns
        rebalance_frequency=30  # Retrain every 30 days
    )
    
    # Load data for multiple assets
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    universe_data = pipeline.load_multiple_assets_data(
        symbols=symbols,
        start_date='2022-01-01',
        end_date='2023-12-31'
    )
    
    print(f"Loaded data for {len(symbols)} symbols")
    
    # Generate features for all symbols
    features_universe = pipeline.generate_features_for_universe(universe_data)
    
    print("Feature generation completed")
    
    # Run backtest
    backtest_results = pipeline.backtest_strategy(
        features_universe=features_universe,
        start_date='2022-06-01',  # Start backtest after enough training data
        end_date='2023-12-31'
    )
    
    print(f"Backtest completed with {len(backtest_results)} predictions")
    
    # Analyze results
    analysis = pipeline.analyze_results(backtest_results)
    
    # Print key results
    print("\n" + "=" * 40)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 40)
    
    if 'model_performance' in analysis:
        print("\nModel Performance:")
        print(analysis['model_performance'])
    
    # Save results
    if not backtest_results.empty:
        backtest_results.to_csv('/workspace/backtest_results.csv', index=False)
        print(f"\nBacktest results saved to: backtest_results.csv")
        
        # Plot results
        pipeline.plot_results(backtest_results, analysis)
    
    # Generate feature importance report
    print("\n" + "=" * 40)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 40)
    
    # Get feature importance from one symbol as example
    example_symbol = symbols[0]
    if example_symbol in features_universe:
        importance_report = feature_engineer.get_feature_importance_report()
        if not importance_report.empty:
            print(f"\nTop 10 Features for {example_symbol}:")
            print(importance_report.head(10)[['avg_abs_ic', 'significant_periods', 'selected']])
            
            importance_report.to_csv('/workspace/feature_importance_final.csv')
            print(f"\nFeature importance report saved to: feature_importance_final.csv")
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_complete_pipeline()