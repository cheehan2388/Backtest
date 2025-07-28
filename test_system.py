"""
Test script for the comprehensive backtesting system.
Generates sample data and runs a quick test to verify system functionality.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from lib import BacktestOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(n_periods=1000, start_date='2020-01-01'):
    """Generate sample financial data for testing"""
    
    # Generate datetime index
    start = pd.to_datetime(start_date)
    dates = pd.date_range(start=start, periods=n_periods, freq='H')
    
    # Generate realistic price data (geometric Brownian motion)
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0.0001, 0.02, n_periods)  # Small positive drift with volatility
    
    # Generate price series
    price = 100 * np.exp(np.cumsum(returns))
    
    # Generate a factor that has some predictive power
    # Add some trend and mean-reverting components
    trend_component = np.sin(np.arange(n_periods) * 2 * np.pi / 100) * 0.5
    noise = np.random.normal(0, 0.3, n_periods)
    factor = trend_component + noise
    
    # Add some correlation with future returns (to make it somewhat predictive)
    factor_with_signal = factor + np.roll(returns, -5) * 10  # 5-period forward-looking component
    
    # Create DataFrame
    df = pd.DataFrame({
        'datetime': dates,
        'close': price,
        'high': price * (1 + np.abs(np.random.normal(0, 0.01, n_periods))),
        'low': price * (1 - np.abs(np.random.normal(0, 0.01, n_periods))),
        'volume': np.random.exponential(1000, n_periods),
        'test_factor': factor_with_signal
    })
    
    # Ensure high >= close >= low
    df['high'] = np.maximum(df['high'], df['close'])
    df['low'] = np.minimum(df['low'], df['close'])
    
    return df

def test_basic_functionality():
    """Test basic system functionality"""
    
    logger.info("Starting basic functionality test...")
    
    # Generate sample data
    df = generate_sample_data(500)  # Smaller dataset for quick testing
    
    # Save to CSV
    test_data_file = 'test_data.csv'
    df.to_csv(test_data_file, index=False)
    
    try:
        # Initialize orchestrator
        orchestrator = BacktestOrchestrator()
        
        # Run a quick test with limited models/strategies
        results = orchestrator.run_full_pipeline(
            data_file=test_data_file,
            factor_column='test_factor',
            output_dir='test_results',
            interval='1h',
            selected_models=['zscore', 'min_max_scaling'],  # Limited set for speed
            selected_strategies=['trend', 'mr'],  # Limited set for speed
            run_permutation_tests=False,  # Skip for speed
            generate_reports=True
        )
        
        logger.info("✓ Basic functionality test passed!")
        logger.info(f"Results: {results}")
        
        # Test strategy selector
        strategy_selector = orchestrator.get_strategy_selector()
        logger.info(f"✓ Generated {len(strategy_selector)} strategy combinations")
        
        # Show top strategies
        if strategy_selector:
            sorted_strategies = sorted(
                strategy_selector.items(),
                key=lambda x: x[1]['sharpe_ratio'],
                reverse=True
            )
            
            print("\nTop 5 strategies:")
            for i, (key, data) in enumerate(sorted_strategies[:5], 1):
                print(f"{i}. {data['model']} + {data['strategy']}: "
                      f"Sharpe={data['sharpe_ratio']:.2f}, "
                      f"Return={data['total_return']:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Basic functionality test failed: {e}")
        return False
    
    finally:
        # Cleanup
        if os.path.exists(test_data_file):
            os.remove(test_data_file)

def test_custom_components():
    """Test custom model and strategy registration"""
    
    logger.info("Testing custom component registration...")
    
    from lib.models import BaseModel, MODEL_REGISTRY
    from lib.strategies import BaseStrategy, STRATEGY_REGISTRY
    
    # Test custom model
    class TestModel(BaseModel):
        def __init__(self):
            super().__init__("test_model")
        
        def generate_signal(self, data: pd.Series, window: int = 20) -> pd.Series:
            return data.rolling(window).mean()
        
        def get_param_ranges(self):
            return {
                'window': np.arange(10, 50, 10),
                'thresholds': np.arange(0.01, 0.1, 0.01)
            }
    
    # Test custom strategy
    class TestStrategy(BaseStrategy):
        def __init__(self):
            super().__init__("test_strategy")
        
        def generate_positions(self, signal: pd.Series, threshold: float) -> pd.Series:
            return pd.Series(np.where(signal > threshold, 1, 0), index=signal.index)
    
    try:
        # Register components
        test_model = TestModel()
        test_strategy = TestStrategy()
        
        MODEL_REGISTRY.register_model(test_model)
        STRATEGY_REGISTRY.register_strategy(test_strategy)
        
        # Verify registration
        assert 'test_model' in MODEL_REGISTRY.get_model_names()
        assert 'test_strategy' in STRATEGY_REGISTRY.get_strategy_names()
        
        logger.info("✓ Custom component registration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Custom component test failed: {e}")
        return False

def test_data_handling():
    """Test data handling capabilities"""
    
    logger.info("Testing data handling...")
    
    from lib.data_handler import DataHandler
    
    try:
        # Generate test data with various scenarios
        df = generate_sample_data(100)
        
        # Add some missing values
        df.loc[50:55, 'close'] = np.nan
        
        # Save test data
        test_file = 'test_data_handling.csv'
        df.to_csv(test_file, index=False)
        
        # Test data handler
        handler = DataHandler()
        loaded_df = handler.load_data(test_file, parse_dates=['datetime'])
        
        # Test data splitting
        splits = handler.split_data()
        
        # Verify splits
        assert 'backtest' in splits
        assert 'validation' in splits
        assert 'forward' in splits
        assert 'full' in splits
        
        # Verify no missing values in price column
        assert not splits['full']['close'].isna().any()
        
        logger.info("✓ Data handling test passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data handling test failed: {e}")
        return False
    
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)

def run_all_tests():
    """Run all system tests"""
    
    print("="*60)
    print("COMPREHENSIVE BACKTESTING SYSTEM - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Custom Components", test_custom_components),
        ("Data Handling", test_data_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n" + "="*60)
        print("SYSTEM READY FOR USE!")
        print("="*60)
        print("You can now run the main system with:")
        print("  python main_new.py")
        print("\nOr use the system programmatically:")
        print("  from lib import BacktestOrchestrator")
        print("  orchestrator = BacktestOrchestrator()")
        print("  results = orchestrator.run_full_pipeline(...)")
    else:
        print("\n" + "="*60)
        print("SYSTEM NEEDS ATTENTION")
        print("="*60)
        print("Please fix the failing tests before using the system.")
        
    # Cleanup test directories
    import shutil
    for test_dir in ['test_results']:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)