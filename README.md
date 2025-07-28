# Comprehensive Backtesting System

A flexible and extensible backtesting framework for quantitative trading strategies with advanced analytics, statistical significance testing, and comprehensive reporting.

## 🚀 Features

### Core Capabilities
- **Modular Architecture**: Pluggable models and strategies with easy extensibility
- **Multiple Models**: Z-score, Min-Max scaling, Moving averages, EWMA, RSI, MACD, and more
- **Diverse Strategies**: Trend following, mean reversion, with various exit conditions
- **Statistical Testing**: Permutation tests for strategy significance
- **Walk-Forward Analysis**: Out-of-sample testing with rolling windows
- **Comprehensive Metrics**: 20+ performance metrics including Sharpe, Sortino, Calmar ratios

### Advanced Analytics
- **Risk Analysis**: Drawdown analysis, VaR, CVaR calculations
- **Performance Attribution**: Detailed trade-level analysis
- **Statistical Significance**: Permutation testing with multiple comparison corrections
- **Forward Testing**: Consistency checks between backtest and forward periods

### Visualization & Reporting
- **Interactive Dashboards**: Plotly-based performance dashboards
- **Parameter Heatmaps**: Visual optimization results across parameter spaces
- **Equity Curves**: Detailed performance visualization with entry/exit points
- **Automated Reports**: JSON, CSV, and HTML output formats

### Data Handling
- **Flexible Input**: Support for CSV, Excel, Parquet formats
- **Data Validation**: Automatic data cleaning and validation
- **Time Series Management**: Proper handling of datetime indices and missing data

## 📁 Project Structure

```
├── lib/                          # Core library modules
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│   ├── data_handler.py          # Data loading and preprocessing
│   ├── models.py                # Signal generation models
│   ├── strategies.py            # Trading strategies
│   ├── backtester.py            # Backtesting engine
│   ├── visualization.py         # Plotting and visualization
│   ├── statistical_tests.py     # Statistical significance testing
│   └── orchestrator.py          # Main system orchestrator
├── main_new.py                  # New system demonstration
├── main.py                      # Original implementation (legacy)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd backtesting-system
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install TA-Lib** (if not already installed):
```bash
# On Ubuntu/Debian
sudo apt-get install libta-lib-dev
pip install TA-Lib

# On macOS
brew install ta-lib
pip install TA-Lib

# On Windows
# Download TA-Lib from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install <downloaded-whl-file>
```

## 🎯 Quick Start

### Basic Usage

```python
from lib import BacktestOrchestrator

# Initialize the system
orchestrator = BacktestOrchestrator()

# Run comprehensive backtest
results = orchestrator.run_full_pipeline(
    data_file='your_data.csv',
    factor_column='your_factor',
    output_dir='results',
    interval='1h',
    run_permutation_tests=True,
    generate_reports=True
)

print(f"Tested {results['total_combinations_tested']} combinations")
print(f"Found {results['significant_strategies']} significant strategies")
```

### Manual Strategy Selection

```python
# Get strategy selector for manual filtering
strategy_selector = orchestrator.get_strategy_selector()

# Filter strategies by criteria
def select_best_strategies(selector, min_sharpe=2.0, max_dd=-0.3):
    selected = []
    for key, data in selector.items():
        if (data['sharpe_ratio'] >= min_sharpe and 
            data['max_drawdown'] >= max_dd):
            selected.append(key)
    return selected

best_strategies = select_best_strategies(strategy_selector)
print(f"Found {len(best_strategies)} strategies meeting criteria")
```

### Custom Models and Strategies

```python
from lib.models import BaseModel, MODEL_REGISTRY
from lib.strategies import BaseStrategy, STRATEGY_REGISTRY
import pandas as pd
import numpy as np

# Custom model
class MyCustomModel(BaseModel):
    def __init__(self):
        super().__init__("my_model")
    
    def generate_signal(self, data: pd.Series, window: int) -> pd.Series:
        return data.rolling(window).std()  # Volatility signal
    
    def get_param_ranges(self):
        return {
            'window': np.arange(10, 100, 10),
            'thresholds': np.arange(0.01, 0.1, 0.01)
        }

# Custom strategy
class MyCustomStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("my_strategy")
    
    def generate_positions(self, signal: pd.Series, threshold: float) -> pd.Series:
        # Your custom logic here
        return pd.Series(np.where(signal > threshold, 1, 0), index=signal.index)

# Register custom components
MODEL_REGISTRY.register_model(MyCustomModel())
STRATEGY_REGISTRY.register_strategy(MyCustomStrategy())
```

## 📊 Available Models

| Model | Description | Parameters |
|-------|-------------|------------|
| **Z-Score** | Standardized signal using rolling mean/std | `window` |
| **Min-Max Scaling** | Normalized signal to [-1, 1] range | `window` |
| **Moving Average** | Simple moving average signal | `window` |
| **Dual MA** | Difference between short and long MA | `short_window`, `long_window` |
| **EWMA Diff** | Exponentially weighted MA difference | `fast_span`, `slow_span`, `alpha` |
| **RSI** | Relative Strength Index | `window` |
| **MACD** | Moving Average Convergence Divergence | `fast_period`, `slow_period`, `signal_period` |
| **Percentile** | Percentile ranking within window | `window` |

## 🎯 Available Strategies

| Strategy | Description | Exit Condition |
|----------|-------------|----------------|
| **Trend** | Basic trend following | Hold until signal reverses |
| **Trend Close** | Trend following | Close when signal exits threshold |
| **Trend Zero** | Trend following | Close when signal returns to zero |
| **Mean Reversion** | Contrarian strategy | Hold until signal reverses |
| **MR Close** | Mean reversion | Close when signal exits threshold |
| **MR Zero** | Mean reversion | Close when signal returns to zero |
| **RSI** | RSI-based strategy | Overbought/oversold levels |
| **Percentile** | Percentile-based strategy | High/low percentile thresholds |

## 📈 Performance Metrics

The system calculates comprehensive performance metrics:

### Return Metrics
- Total Return
- Annualized Return
- Compound Annual Growth Rate (CAGR)

### Risk Metrics
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Maximum Drawdown
- Volatility
- Value at Risk (VaR)
- Conditional VaR (CVaR)

### Trade Statistics
- Number of Trades
- Win Rate
- Average Win/Loss
- Profit Factor
- Trade Frequency

### Advanced Metrics
- Skewness
- Kurtosis
- Recovery Time Analysis
- Statistical Significance (p-values)

## 🔬 Statistical Testing

### Permutation Tests
The system includes robust permutation testing to assess strategy significance:

```python
# Automatic permutation testing
results = orchestrator.run_full_pipeline(
    ...,
    run_permutation_tests=True  # Enable statistical testing
)

# Manual permutation test
from lib import PermutationTester
tester = PermutationTester()
perm_result = tester.run_permutation_test(
    data=factor_data,
    model_func=model,
    strategy_func=strategy,
    model_params={'window': 50},
    strategy_params={'threshold': 0.5},
    price=price_data,
    original_sharpe=2.5,
    num_permutations=1000
)
```

### Multiple Testing Correction
```python
from lib import StatisticalAnalyzer
analyzer = StatisticalAnalyzer()

# Apply Bonferroni correction
corrected_p_values = analyzer.multiple_testing_correction(
    p_values=[0.01, 0.03, 0.05],
    method='bonferroni'
)
```

## 📊 Visualization

### Equity Curves
```python
from lib import Visualizer
viz = Visualizer()

# Plot equity curves
viz.plot_equity_curves(
    results=[(name, backtest_result), ...],
    output_path='equity_curves.png'
)
```

### Parameter Heatmaps
```python
# Generate parameter heatmaps
viz.plot_parameter_heatmaps(
    results_df=results_dataframe,
    model='zscore',
    strategy='trend',
    output_dir='heatmaps'
)
```

### Interactive Dashboard
```python
# Create interactive Plotly dashboard
viz.create_performance_dashboard(
    results=strategy_results,
    output_path='dashboard.html'
)
```

## ⚙️ Configuration

### Backtest Configuration
```python
from lib.config import BACKTEST_CONFIG

# Modify configuration
BACKTEST_CONFIG.fee = 0.001  # Trading fee
BACKTEST_CONFIG.min_sharpe = 1.5  # Minimum Sharpe for filtering
BACKTEST_CONFIG.max_drawdown = -0.4  # Maximum drawdown threshold
```

### Model Parameters
```python
from lib.config import MODEL_PARAMS

# Customize parameter ranges
MODEL_PARAMS.zscore_windows = np.arange(10, 200, 5)
MODEL_PARAMS.zscore_thresholds = np.arange(0.5, 3.0, 0.1)
```

## 📝 Output Files

The system generates comprehensive outputs:

### JSON Reports
- `factor_backtest_results_1h.json`: All backtest results
- `factor_significant_strategies_1h.json`: Statistically significant strategies

### CSV Files
- `strategy_summary.csv`: Performance summary table
- `significance_analysis.csv`: Statistical significance analysis

### Visualizations
- `equity_curves.png`: Strategy performance comparison
- `performance_dashboard.html`: Interactive dashboard
- `strategy_detail_*.png`: Individual strategy analysis
- Parameter heatmaps for each model/strategy combination

## 🧪 Walk-Forward Analysis

```python
from lib.backtester import WalkForwardAnalyzer

analyzer = WalkForwardAnalyzer()
wf_results = analyzer.run_walk_forward(
    data=full_dataset,
    model_func=model,
    strategy_func=strategy,
    model_params={'window': 50},
    strategy_params={'threshold': 1.0},
    factor_column='factor',
    window_size=252,  # 1 year
    step_size=63     # 3 months
)
```

## 🔧 Advanced Usage

### Batch Processing
```python
# Process multiple factors
factors = ['factor1', 'factor2', 'factor3']
for factor in factors:
    results = orchestrator.run_full_pipeline(
        data_file='data.csv',
        factor_column=factor,
        output_dir=f'results_{factor}',
        interval='1h'
    )
```

### Custom Filtering
```python
# Custom result filtering
def custom_filter(result):
    return (result.sharpe_ratio > 2.0 and 
            result.max_drawdown_pct > -0.3 and
            result.num_trades > 20)

filtered_results = [r for r in all_results if custom_filter(r)]
```

## 🐛 Troubleshooting

### Common Issues

1. **TA-Lib Installation**: Follow platform-specific installation instructions
2. **Memory Usage**: For large datasets, consider reducing parameter ranges
3. **Data Format**: Ensure datetime column is properly formatted
4. **Missing Data**: The system handles missing data automatically

### Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with pandas, numpy, matplotlib, seaborn, plotly
- Technical indicators powered by TA-Lib
- Statistical testing using scipy

---

For more examples and detailed documentation, check the `main_new.py` file and explore the `/lib` modules.