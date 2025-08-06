# Automated Feature Engineering for Quantitative Trading

A comprehensive Python system for automated feature generation, Information Coefficient (IC) analysis, and feature selection specifically designed for quantitative trading strategies.

## 🎯 Overview

This system automates the process of creating and evaluating features for quantitative trading models by:

- **Generating 100+ technical features** from price and volume data
- **Calculating Information Coefficient (IC)** to measure predictive power
- **Automatically selecting the best features** based on statistical significance
- **Removing highly correlated features** to avoid multicollinearity
- **Cross-sectional ranking** across multiple assets
- **Backtesting framework** with time-series validation

## 🏗️ Key Components

### 1. AutoFeatureEngineer Class
The core class that handles all feature engineering tasks:

```python
from auto_feature_engineer import AutoFeatureEngineer

# Initialize with custom parameters
feature_engineer = AutoFeatureEngineer(
    lookback_periods=[5, 10, 20, 50, 100],    # Periods for technical indicators
    min_ic_threshold=0.02,                     # Minimum IC to keep a feature
    max_features=50,                           # Maximum features to select
    correlation_threshold=0.8                  # Max correlation between features
)
```

### 2. Feature Categories Generated

#### Price-Based Features
- Momentum indicators (various periods)
- Mean reversion features (price to moving average ratios)
- Volatility measures
- Log transformations

#### Technical Indicators
- RSI, CCI, MACD, Stochastic
- Bollinger Bands positions and widths
- Williams %R, ATR
- Multiple moving averages (SMA, EMA)

#### Volume-Based Features
- Volume momentum and trends
- On-Balance Volume (OBV)
- Price-volume relationships

#### Cross-Sectional Features
- Percentile rankings across universe
- Relative performance metrics

#### Interaction Features
- Multiplicative combinations of key features
- Ratio features between important indicators

### 3. Information Coefficient Analysis

The system uses Spearman rank correlation (Information Coefficient) to measure the relationship between features and forward returns:

```python
# IC measures how well features predict future returns
ic_scores = feature_engineer.calculate_information_coefficient(features, returns)

# Features are selected based on:
# - Statistical significance (p < 0.05)
# - Minimum IC threshold
# - Stability across time periods
# - Low correlation with other features
```

## 📊 Information Coefficient in Quantitative Finance

**Information Coefficient (IC)** is a crucial metric in quantitative finance that measures the correlation between predicted and actual returns:

- **IC > 0.05**: Excellent predictive power
- **IC > 0.03**: Good predictive power  
- **IC > 0.01**: Modest but potentially useful
- **IC < 0.01**: Limited predictive value

The system automatically:
1. Calculates IC for multiple return horizons
2. Tests statistical significance
3. Measures IC stability over time
4. Ranks features by IC quality

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install -r requirements.txt

# Note: TA-Lib requires separate installation
# On Ubuntu/Debian:
sudo apt-get install libta-lib-dev
pip install TA-Lib

# On macOS:
brew install ta-lib
pip install TA-Lib

# On Windows:
# Download TA-Lib from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# pip install path/to/TA_Lib-0.4.25-cp39-cp39-win_amd64.whl
```

### Basic Usage

```python
import pandas as pd
from auto_feature_engineer import AutoFeatureEngineer

# Load your market data (OHLCV format)
data = pd.read_csv('your_market_data.csv')
data.set_index('date', inplace=True)

# Initialize feature engineer
feature_engineer = AutoFeatureEngineer(
    min_ic_threshold=0.02,
    max_features=30
)

# Generate and select features
selected_features, all_features, ic_analysis = feature_engineer.fit_transform(data)

# View results
importance_report = feature_engineer.get_feature_importance_report()
print(importance_report.head(10))
```

### Advanced Pipeline Usage

```python
# Run the complete trading pipeline
python advanced_trading_example.py
```

This will:
1. Generate features for multiple symbols
2. Train prediction models
3. Perform time-series backtesting
4. Generate performance reports and visualizations

## 📈 Expected Data Format

Your data should be a pandas DataFrame with these columns:

```python
data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=1000),
    'open': [...],     # Opening prices
    'high': [...],     # High prices  
    'low': [...],      # Low prices
    'close': [...],    # Closing prices
    'volume': [...]    # Volume (optional but recommended)
})
data.set_index('date', inplace=True)
```

## 🔧 Customization Options

### Feature Engineering Parameters

```python
feature_engineer = AutoFeatureEngineer(
    lookback_periods=[5, 10, 20, 50],      # Custom indicator periods
    min_ic_threshold=0.015,                 # Lower for more features
    max_features=25,                        # Limit feature count
    correlation_threshold=0.85              # Allow higher correlation
)
```

### Return Prediction Horizons

```python
# Predict returns for different time horizons
selected_features, all_features, ic_analysis = feature_engineer.fit_transform(
    data,
    return_periods=[1, 5, 10, 20]  # 1-day, 5-day, 10-day, 20-day returns
)
```

### Cross-Sectional Analysis

```python
# Include universe data for cross-sectional features
universe_data = {
    'AAPL': aapl_data,
    'GOOGL': googl_data,
    'MSFT': msft_data
}

selected_features, all_features, ic_analysis = feature_engineer.fit_transform(
    data,
    universe_data=universe_data  # Enables cross-sectional ranking
)
```

## 📊 Output Files

The system generates several output files:

1. **`selected_features.csv`** - Final selected features with values
2. **`feature_importance_report.csv`** - Detailed IC analysis for all features
3. **`ic_analysis.csv`** - Raw IC statistics across return periods
4. **`backtest_results.csv`** - Backtesting results (from advanced pipeline)
5. **`backtest_results.png`** - Performance visualization charts

## 🏆 Best Practices for Quantitative Trading

### 1. Feature Selection Strategy
- Start with IC threshold of 0.02 for conservative selection
- Use multiple return horizons to validate feature stability
- Regularly retrain models (monthly/quarterly)
- Monitor feature performance degradation over time

### 2. Risk Management
- Always check for look-ahead bias in feature construction
- Use proper time-series cross-validation
- Test features on out-of-sample data
- Monitor correlation between features and market regimes

### 3. Implementation Tips
- Use robust scaling for features with outliers
- Consider regime-dependent feature importance
- Implement feature decay for older observations
- Regular feature importance monitoring in production

## ⚠️ Important Considerations

### Data Quality
- Ensure data is clean and adjusted for splits/dividends
- Handle missing values appropriately
- Check for survivor bias in historical data

### Statistical Validity
- IC significance doesn't guarantee future performance
- Features may degrade over time due to market evolution
- Always validate on truly out-of-sample data

### Computational Efficiency
- Feature generation can be time-intensive for large universes
- Consider parallel processing for multiple symbols
- Cache feature computations when possible

## 🔬 Advanced Features

### Custom Feature Functions

```python
def custom_feature_generator(data):
    """Add your custom features here"""
    features = pd.DataFrame(index=data.index)
    
    # Example: Price momentum with volume weighting
    features['volume_weighted_momentum'] = (
        data['close'].pct_change(20) * data['volume'].rolling(20).mean()
    )
    
    return features

# Integrate custom features into the pipeline
# (Modify the generate_custom_features method in AutoFeatureEngineer)
```

### Alternative IC Calculations

The system uses Spearman rank correlation by default, but you can modify it to use:
- Pearson correlation for linear relationships
- Mutual information for non-linear relationships
- Custom scoring functions

## 📚 References and Further Reading

1. **"Advances in Financial Machine Learning"** by Marcos López de Prado
2. **"Quantitative Portfolio Management"** - Various academic papers on IC analysis
3. **"Machine Learning for Asset Managers"** by Marcos López de Prado
4. **TA-Lib Documentation**: https://ta-lib.org/
5. **Pandas Financial Analysis**: https://pandas.pydata.org/

## 🤝 Contributing

Feel free to contribute by:
- Adding new feature generators
- Improving IC calculation methods
- Adding more sophisticated feature selection algorithms
- Enhancing the backtesting framework

## 📄 License

This project is provided as-is for educational and research purposes. Please ensure compliance with your institution's policies and applicable regulations when using in production trading systems.

---

**Disclaimer**: This system is for educational purposes only. Past performance does not guarantee future results. Always perform thorough backtesting and risk assessment before deploying any trading strategy.