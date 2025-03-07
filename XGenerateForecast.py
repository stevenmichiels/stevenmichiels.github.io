import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from XForecastIndicator import ForecastIndicator, plot_forecast
import argparse
import json

def find_latest_nan_date(df, instrument):
    """Find the latest date where there are NaN values in the instrument column."""
    nan_dates = df[df[instrument].isna()].index
    if len(nan_dates) > 0:
        return nan_dates[-1]
    return None

# Custom JSON encoder to handle NaN values
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def calculate_metrics(prices, forecasts):
    """Calculate performance metrics for a single instrument"""
    # Calculate daily returns
    daily_returns = prices.pct_change()

    # Create position signals (1 for long, 0 for cash)
    positions = (forecasts > 0).astype(int)
    positions = positions.shift(1)  # Shift by 1 day to avoid look-ahead bias
    positions.iloc[0] = 0  # Set first day position to 0

    # Calculate strategy returns
    strategy_returns = daily_returns * positions
    cumulative_returns = (1 + strategy_returns).cumprod()
    buy_hold_returns = (1 + daily_returns).cumprod()

    # Convert returns to percentages
    cumulative_returns = (cumulative_returns - 1) * 100
    buy_hold_returns = (buy_hold_returns - 1) * 100

    # Calculate drawdowns
    def calculate_drawdown(returns):
        cumulative = (1 + returns/100)
        rolling_max = cumulative.cummax()
        drawdowns = (cumulative - rolling_max) / rolling_max * 100
        return drawdowns

    strategy_dd = calculate_drawdown(cumulative_returns)
    bh_dd = calculate_drawdown(buy_hold_returns)

    # Calculate annualized metrics
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    strategy_cagr = (1 + cumulative_returns.iloc[-1]/100) ** (1/years) - 1
    bh_cagr = (1 + buy_hold_returns.iloc[-1]/100) ** (1/years) - 1

    # Calculate annualized volatilities
    strategy_vol = np.std(strategy_returns) * np.sqrt(252)
    bh_vol = np.std(daily_returns) * np.sqrt(252)

    # Calculate Sharpe ratios (assuming 0% risk-free rate for simplicity)
    strategy_sharpe = strategy_cagr / strategy_vol
    bh_sharpe = bh_cagr / bh_vol

    return {
        'cumulative_returns': cumulative_returns,
        'buy_hold_returns': buy_hold_returns,
        'strategy_drawdown': strategy_dd,
        'bh_drawdown': bh_dd,
        'positions': positions,
        'metrics': {
            'strategy_final_return': f"{cumulative_returns.iloc[-1]:.1f}",
            'bh_final_return': f"{buy_hold_returns.iloc[-1]:.1f}",
            'avg_strategy_dd': f"{strategy_dd.mean():.1f}",
            'avg_bh_dd': f"{bh_dd.mean():.1f}",
            'strategy_sharpe': f"{strategy_sharpe:.2f}",
            'bh_sharpe': f"{bh_sharpe:.2f}",
            'num_trades': f"{positions.diff().abs().sum() / 2:.0f}",
            'max_dd_strategy': f"{strategy_dd.min():.1f}",
            'max_dd_bh': f"{bh_dd.min():.1f}"
        }
    }

def run_forecasts(instruments=['SPX', 'NDX', 'GC', 'XLE'], start_year=1962, subfolder='stevenmichiels.github.io'):
    """Run forecasts for multiple instruments and combine results into a single JSON file"""
    # Read the CSV file
    base_path = f'/Users/stevenmichiels/Repos/{subfolder}'
    df = pd.read_csv(os.path.join(base_path, 'Xdaily.csv'))
    
    # Verify if all instruments exist in the DataFrame
    missing_instruments = [instr for instr in instruments if instr not in df.columns]
    if missing_instruments:
        raise ValueError(f"Instruments not found in data: {', '.join(missing_instruments)}")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Filter data from start_year onwards
    df = df[df.index.year >= start_year]

    # Create output filename for JSON
    output_path = os.path.join(base_path, 'forecast_data.json')
    print(f"Will save forecast data to: {output_path}")

    # Create a forecast indicator
    indicator = ForecastIndicator(base_period=8, include='16-64-256')

    # Initialize dictionary to store all forecast data
    all_forecasts = {
        'metadata': {
            'last_update': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start_year': start_year,
            'instruments': instruments
        },
        'instruments': {}
    }

    def clean_series(series):
        return [float(x) if not np.isnan(x) else None for x in series]

    # Process each instrument
    for instrument in instruments:
        print(f"\nProcessing {instrument}...")
        
        # Check for latest NaN date
        latest_nan = find_latest_nan_date(df, instrument)
        if latest_nan is not None:
            print(f"Latest NaN value found on: {latest_nan.strftime('%Y-%m-%d')}")

        # Extract instrument prices and calculate forecasts
        prices = df[instrument]
        forecasts = indicator.calculate_forecast(prices)
        
        # Calculate metrics
        metrics = calculate_metrics(prices, forecasts)
        
        # Store data for this instrument
        all_forecasts['instruments'][instrument] = {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'prices': clean_series(prices),
            'forecasts': clean_series(forecasts),
            'cumulative_returns': clean_series(metrics['cumulative_returns']),
            'buy_hold_returns': clean_series(metrics['buy_hold_returns']),
            'strategy_drawdown': clean_series(metrics['strategy_drawdown']),
            'bh_drawdown': clean_series(metrics['bh_drawdown']),
            'positions': clean_series(metrics['positions']),
            'metrics': metrics['metrics']
        }

        # Print performance statistics
        print(f"\nPerformance Summary for {instrument}:")
        for key, value in metrics['metrics'].items():
            print(f"{key}: {value}")

    # Save to JSON file
    try:
        with open(output_path, 'w') as f:
            json.dump(all_forecasts, f, cls=NpEncoder)
        print(f"\n✅ Forecast data has been saved as '{output_path}'")
    except Exception as e:
        print(f"\n❌ Error saving forecast data: {str(e)}")
        return

if __name__ == "__main__":
    try:
        # Check if running in Jupyter
        get_ipython()
        is_jupyter = True
    except:
        is_jupyter = False

    if is_jupyter:
        # Default values when running in Jupyter
        run_forecasts(instruments=['SPX', 'NDX', 'GC', 'CVX', 'XOM'], start_year=1962)
    else:
        # Command line argument parsing when running as script
        parser = argparse.ArgumentParser(description='Run forecast strategy on financial instruments')
        parser.add_argument('--instruments', type=str, nargs='+', default=['SPX', 'NDX', 'GC', 'XLE'],
                          help='List of instruments to analyze (must be columns in Xdaily.csv)')
        parser.add_argument('--start_year', type=int, default=1962,
                          help='Start year for analysis')
        args = parser.parse_args()
        run_forecasts(args.instruments, args.start_year)
