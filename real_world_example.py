"""
Real-World Example: Automated Feature Engineering for Quantitative Trading

This example demonstrates how to:
1. Load real market data (or connect to data sources)
2. Generate and select features using IC analysis
3. Build a simple prediction model
4. Implement a basic trading strategy
5. Evaluate performance

For production use, replace the data loading with your actual data source
(e.g., Yahoo Finance, Alpha Vantage, Bloomberg, etc.)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
from test_basic_features import BasicAutoFeatureEngineer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class TradingStrategy:
    """
    Simple trading strategy based on predicted returns
    """
    
    def __init__(self, 
                 feature_engineer: BasicAutoFeatureEngineer,
                 rebalance_frequency: int = 20,
                 transaction_cost: float = 0.001):
        """
        Initialize trading strategy
        
        Args:
            feature_engineer: Trained feature engineering instance
            rebalance_frequency: Days between portfolio rebalancing
            transaction_cost: Cost per trade (percentage)
        """
        self.feature_engineer = feature_engineer
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.model = None
        self.positions = []
        self.returns = []
        
    def train_model(self, features: pd.DataFrame, returns: pd.Series, model_type: str = 'rf'):
        """Train prediction model"""
        
        # Remove NaN values
        mask = ~(features.isna().any(axis=1) | returns.isna())
        clean_features = features[mask]
        clean_returns = returns[mask]
        
        if model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
        else:  # linear regression
            self.model = LinearRegression()
        
        self.model.fit(clean_features, clean_returns)
        
        # Calculate training metrics
        train_pred = self.model.predict(clean_features)
        train_ic = np.corrcoef(train_pred, clean_returns)[0, 1]
        train_mse = mean_squared_error(clean_returns, train_pred)
        
        print(f"Training Results ({model_type}):")
        print(f"- IC: {train_ic:.4f}")
        print(f"- MSE: {train_mse:.6f}")
        print(f"- Samples: {len(clean_features)}")
        
        return train_ic, train_mse
    
    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on predictions"""
        
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Make predictions
        predictions = self.model.predict(features)
        
        # Convert predictions to signals
        # Simple strategy: buy if predicted return > threshold, sell if < -threshold
        threshold = np.std(predictions) * 0.5  # Dynamic threshold
        
        signals = pd.Series(index=features.index, dtype=float)
        signals[predictions > threshold] = 1.0    # Long
        signals[predictions < -threshold] = -1.0  # Short
        signals[np.abs(predictions) <= threshold] = 0.0  # Neutral
        
        return signals, pd.Series(predictions, index=features.index)
    
    def backtest(self, 
                 data: pd.DataFrame,
                 features: pd.DataFrame,
                 returns: pd.Series,
                 start_date: str = None,
                 end_date: str = None) -> Dict:
        """
        Backtest the trading strategy
        
        Args:
            data: Original price data
            features: Feature matrix
            returns: Forward returns
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Dictionary with backtest results
        """
        
        # Filter data by date range
        if start_date:
            mask = data.index >= start_date
            data = data[mask]
            features = features[mask]
            returns = returns[mask]
        
        if end_date:
            mask = data.index <= end_date
            data = data[mask]
            features = features[mask]
            returns = returns[mask]
        
        # Align all data
        common_index = data.index.intersection(features.index).intersection(returns.index)
        data = data.loc[common_index]
        features = features.loc[common_index]
        returns = returns.loc[common_index]
        
        # Initialize results
        backtest_results = []
        portfolio_value = 1.0  # Start with $1
        current_position = 0.0
        
        # Time series split for walk-forward analysis
        min_train_size = 200  # Minimum training samples
        
        for i in range(min_train_size, len(data), self.rebalance_frequency):
            
            # Define training and testing periods
            train_end = i
            test_start = i
            test_end = min(i + self.rebalance_frequency, len(data))
            
            # Split data
            train_features = features.iloc[:train_end]
            train_returns = returns.iloc[:train_end]
            test_features = features.iloc[test_start:test_end]
            test_returns = returns.iloc[test_start:test_end]
            test_data = data.iloc[test_start:test_end]
            
            # Skip if insufficient data
            if len(train_features) < 50 or len(test_features) < 1:
                continue
            
            try:
                # Train model
                self.train_model(train_features, train_returns, model_type='rf')
                
                # Generate signals for test period
                signals, predictions = self.generate_signals(test_features)
                
                # Simulate trading
                for j, (date, signal) in enumerate(signals.items()):
                    
                    # Calculate position change
                    position_change = signal - current_position
                    
                    # Calculate transaction costs
                    transaction_cost = abs(position_change) * self.transaction_cost
                    
                    # Update portfolio value with return
                    if j < len(test_returns):
                        period_return = test_returns.iloc[j]
                        portfolio_return = current_position * period_return - transaction_cost
                        portfolio_value *= (1 + portfolio_return)
                    
                    # Update position
                    current_position = signal
                    
                    # Store results
                    backtest_results.append({
                        'date': date,
                        'portfolio_value': portfolio_value,
                        'position': current_position,
                        'prediction': predictions.loc[date] if date in predictions.index else 0,
                        'actual_return': test_returns.iloc[j] if j < len(test_returns) else 0,
                        'signal': signal
                    })
                    
            except Exception as e:
                print(f"Error during backtest period {i}: {e}")
                continue
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(backtest_results)
        
        if results_df.empty:
            return {'error': 'No backtest results generated'}
        
        # Calculate performance metrics
        results_df.set_index('date', inplace=True)
        
        # Calculate returns
        results_df['strategy_return'] = results_df['portfolio_value'].pct_change()
        results_df['cumulative_return'] = results_df['portfolio_value'] - 1.0
        
        # Performance metrics
        total_return = results_df['cumulative_return'].iloc[-1]
        volatility = results_df['strategy_return'].std() * np.sqrt(252)  # Annualized
        sharpe_ratio = results_df['strategy_return'].mean() / results_df['strategy_return'].std() * np.sqrt(252)
        max_drawdown = (results_df['portfolio_value'] / results_df['portfolio_value'].expanding().max() - 1).min()
        
        # Win rate
        winning_trades = (results_df['strategy_return'] > 0).sum()
        total_trades = (results_df['strategy_return'] != 0).sum()
        win_rate = winning_trades / max(total_trades, 1)
        
        performance_metrics = {
            'total_return': total_return,
            'annualized_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_portfolio_value': results_df['portfolio_value'].iloc[-1]
        }
        
        return {
            'results_df': results_df,
            'performance_metrics': performance_metrics
        }


def load_real_world_data(symbol: str = 'AAPL', start_date: str = '2020-01-01', end_date: str = '2023-12-31') -> pd.DataFrame:
    """
    Load real market data - replace this with your actual data source
    
    For production, you might use:
    - yfinance: yf.download(symbol, start=start_date, end=end_date)
    - Alpha Vantage API
    - Bloomberg API
    - Quandl
    - Your broker's API
    """
    
    # For this example, we'll generate realistic synthetic data
    # In practice, replace this with actual data loading
    
    print(f"Loading data for {symbol} from {start_date} to {end_date}")
    print("(Using synthetic data for demo - replace with real data source)")
    
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Generate more realistic market data with regime changes
    np.random.seed(hash(symbol) % 2**32)
    n_days = len(dates)
    
    # Create different market regimes
    regime_length = n_days // 4
    regimes = []
    
    for i in range(4):
        if i == 0:  # Bull market
            trend = np.linspace(0, 0.4, regime_length)
            volatility = 0.15
        elif i == 1:  # Volatile market
            trend = np.linspace(0, 0.1, regime_length)
            volatility = 0.25
        elif i == 2:  # Bear market
            trend = np.linspace(0, -0.2, regime_length)
            volatility = 0.20
        else:  # Recovery
            trend = np.linspace(0, 0.3, n_days - 3 * regime_length)
            volatility = 0.18
            
        regime_returns = np.random.normal(0.0005, volatility/np.sqrt(252), len(trend))
        regimes.extend(trend + np.cumsum(regime_returns))
    
    # Ensure we have the right length
    regimes = regimes[:n_days]
    prices = 100 * np.exp(regimes)
    
    # Generate OHLC with realistic intraday movements
    daily_vol = np.abs(np.diff(np.concatenate([[0], regimes]))) + np.random.normal(0, 0.01, n_days)
    
    high_prices = prices * (1 + daily_vol * np.random.uniform(0.2, 0.8, n_days))
    low_prices = prices * (1 - daily_vol * np.random.uniform(0.2, 0.8, n_days))
    open_prices = prices + np.random.normal(0, prices * 0.005)
    
    # Volume with mean reversion
    volume_base = np.random.lognormal(12, 0.5, n_days)
    volume = volume_base * (1 + 0.3 * daily_vol)  # Higher volume on volatile days
    
    data = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': prices,
        'volume': volume,
        'symbol': symbol
    })
    
    data.set_index('date', inplace=True)
    return data


def main():
    """Run complete real-world example"""
    
    print("=" * 70)
    print("REAL-WORLD QUANTITATIVE TRADING EXAMPLE")
    print("=" * 70)
    
    # 1. Load market data
    print("\n1. Loading Market Data...")
    data = load_real_world_data('AAPL', '2020-01-01', '2023-12-31')
    print(f"   Loaded {len(data)} days of data")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # 2. Generate features
    print("\n2. Generating Features...")
    feature_engineer = BasicAutoFeatureEngineer(
        lookback_periods=[5, 10, 20, 50],
        min_ic_threshold=0.005,  # Lower threshold for real market data
        max_features=20
    )
    
    selected_features, all_features, ic_scores = feature_engineer.fit_transform(
        data, 
        return_periods=[5, 10, 20]  # Multiple horizons
    )
    
    print(f"   Generated {len(all_features.columns)} features")
    print(f"   Selected {len(selected_features.columns)} best features")
    
    # 3. Show top features
    print("\n3. Top Features by Information Coefficient:")
    if ic_scores:
        sorted_features = sorted(ic_scores.items(), key=lambda x: x[1]['abs_ic'], reverse=True)
        for i, (feature_name, scores) in enumerate(sorted_features[:10]):
            status = "✓" if feature_name in feature_engineer.selected_features else " "
            print(f"   {status} {feature_name:25s} IC: {scores['ic']:7.4f} (p={scores['p_value']:.3f})")
    
    # 4. Build trading strategy
    print("\n4. Building Trading Strategy...")
    strategy = TradingStrategy(
        feature_engineer=feature_engineer,
        rebalance_frequency=10,  # Rebalance every 10 days
        transaction_cost=0.001   # 0.1% transaction cost
    )
    
    # 5. Run backtest
    print("\n5. Running Backtest...")
    returns_5d = feature_engineer.calculate_returns(data['close'], [5])['return_5d']
    
    backtest_results = strategy.backtest(
        data=data,
        features=selected_features,
        returns=returns_5d,
        start_date='2021-01-01',  # Use 2020 data for initial training
        end_date='2023-12-31'
    )
    
    # 6. Display results
    print("\n6. Backtest Results:")
    print("=" * 50)
    
    if 'error' in backtest_results:
        print(f"   Error: {backtest_results['error']}")
        return
    
    perf = backtest_results['performance_metrics']
    results_df = backtest_results['results_df']
    
    print(f"   Total Return:        {perf['total_return']:8.2%}")
    print(f"   Annualized Volatility: {perf['annualized_volatility']:6.2%}")
    print(f"   Sharpe Ratio:        {perf['sharpe_ratio']:8.2f}")
    print(f"   Max Drawdown:        {perf['max_drawdown']:8.2%}")
    print(f"   Win Rate:            {perf['win_rate']:8.2%}")
    print(f"   Total Trades:        {perf['total_trades']:8.0f}")
    print(f"   Final Portfolio:     ${perf['final_portfolio_value']:8.2f}")
    
    # 7. Calculate benchmark (buy and hold)
    buy_hold_return = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1
    print(f"   Buy & Hold Return:   {buy_hold_return:8.2%}")
    
    # 8. Save results
    print("\n7. Saving Results...")
    selected_features.to_csv('/workspace/strategy_features.csv')
    results_df.to_csv('/workspace/strategy_backtest.csv')
    
    # Save performance summary
    with open('/workspace/strategy_performance.txt', 'w') as f:
        f.write("QUANTITATIVE TRADING STRATEGY PERFORMANCE\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Return:          {perf['total_return']:8.2%}\n")
        f.write(f"Annualized Volatility: {perf['annualized_volatility']:8.2%}\n")
        f.write(f"Sharpe Ratio:          {perf['sharpe_ratio']:8.2f}\n")
        f.write(f"Max Drawdown:          {perf['max_drawdown']:8.2%}\n")
        f.write(f"Win Rate:              {perf['win_rate']:8.2%}\n")
        f.write(f"Total Trades:          {perf['total_trades']:8.0f}\n")
        f.write(f"Buy & Hold Return:     {buy_hold_return:8.2%}\n")
    
    print("   Results saved to:")
    print("   - strategy_features.csv")
    print("   - strategy_backtest.csv")
    print("   - strategy_performance.txt")
    
    # 9. Create simple visualization
    print("\n8. Creating Performance Chart...")
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Portfolio value over time
        ax1.plot(results_df.index, results_df['portfolio_value'], label='Strategy', linewidth=2)
        
        # Buy and hold comparison
        buy_hold_values = data['close'] / data['close'].iloc[0]
        common_dates = results_df.index.intersection(buy_hold_values.index)
        ax1.plot(common_dates, buy_hold_values.loc[common_dates], 
                label='Buy & Hold', alpha=0.7, linewidth=2)
        
        ax1.set_title('Portfolio Performance Comparison')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Positions over time
        ax2.plot(results_df.index, results_df['position'], label='Position', alpha=0.8)
        ax2.set_title('Trading Positions Over Time')
        ax2.set_ylabel('Position (-1: Short, 0: Neutral, 1: Long)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/workspace/strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   Performance chart saved to: strategy_performance.png")
        
    except Exception as e:
        print(f"   Could not create chart: {e}")
    
    print("\n" + "=" * 70)
    print("REAL-WORLD EXAMPLE COMPLETED!")
    print("=" * 70)
    
    print("\nNext Steps for Production Implementation:")
    print("1. Replace synthetic data with real market data API")
    print("2. Add more sophisticated risk management")
    print("3. Implement proper position sizing")
    print("4. Add regime detection and adaptive parameters")
    print("5. Include multiple assets for portfolio construction")
    print("6. Add real-time prediction and execution system")


if __name__ == "__main__":
    main()