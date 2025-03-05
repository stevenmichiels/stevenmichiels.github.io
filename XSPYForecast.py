import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from XForecastIndicator import ForecastIndicator, plot_forecast
import argparse

def run_forecast(instrument='SPX', start_year=1962, subfolder='stevenmichiels.github.io'):
    # Read the CSV file
    df = pd.read_csv('/Users/stevenmichiels/Repos/'+subfolder+'/Xdaily.csv')
    
    # Verify if instrument exists in the DataFrame
    if instrument not in df.columns:
        raise ValueError(f"Instrument '{instrument}' not found in data. Available instruments: {', '.join(df.columns)}")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Filter data from start_year onwards
    df = df[df.index.year >= start_year]

    # Extract instrument prices
    prices = df[instrument]
    
    # Create output filename based on instrument
    output_path = f'/Users/stevenmichiels/Repos/' +subfolder+'/{instrument}_strategy_performance.html'

    # Create a forecast indicator and calculate forecasts
    indicator = ForecastIndicator(base_period=8, include='16-64-256')
    forecasts = indicator.calculate_forecast(prices)

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
    years = (df.index[-1] - df.index[0]).days / 365.25
    strategy_cagr = (1 + cumulative_returns.iloc[-1]/100) ** (1/years) - 1
    bh_cagr = (1 + buy_hold_returns.iloc[-1]/100) ** (1/years) - 1

    # Calculate annualized volatilities
    strategy_vol = np.std(strategy_returns) * np.sqrt(252)  # 252 trading days
    bh_vol = np.std(daily_returns) * np.sqrt(252)

    # Calculate Sharpe ratios (assuming 0% risk-free rate for simplicity)
    strategy_sharpe = strategy_cagr / strategy_vol
    bh_sharpe = bh_cagr / bh_vol

    # Create subplot figure with three rows
    fig = make_subplots(rows=3, cols=1, 
                        subplot_titles=('Cumulative Returns (%)', 'Drawdown (%)', 'Position Signal'),
                        vertical_spacing=0.1,
                        row_heights=[0.4, 0.3, 0.3])

    # Add cumulative return traces
    fig.add_trace(
        go.Scatter(x=cumulative_returns.index, y=cumulative_returns, 
                   name='Strategy Returns', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=buy_hold_returns.index, y=buy_hold_returns, 
                   name='Buy & Hold Returns', line=dict(color='gray')),
        row=1, col=1
    )

    # Add drawdown traces
    fig.add_trace(
        go.Scatter(x=strategy_dd.index, y=strategy_dd, 
                   name='Strategy Drawdown', line=dict(color='red')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=bh_dd.index, y=bh_dd, 
                   name='Buy & Hold Drawdown', line=dict(color='orange')),
        row=2, col=1
    )

    # Add position signal trace (now in third row)
    fig.add_trace(
        go.Scatter(x=positions.index, y=positions, 
                   name='Position (1=Long, 0=Cash)', line=dict(color='green')),
        row=3, col=1
    )

    # Calculate average drawdowns
    avg_strategy_dd = strategy_dd.mean()
    avg_bh_dd = bh_dd.mean()

    # Update layout with instrument name
    fig.update_layout(
        title=f'{instrument} Portfolio Performance with Forecast Signals<br>'
              f'Strategy Sharpe: {strategy_sharpe:.2f} | Buy&Hold Sharpe: {bh_sharpe:.2f}<br>'
              f'Avg DD Strategy: {avg_strategy_dd:.1f}% | Avg DD Buy&Hold: {avg_bh_dd:.1f}%',
        height=1000,
        showlegend=True,
        yaxis_title="Returns (%)",
        yaxis2_title="Drawdown (%)",
        yaxis3_title="Position (1=Long, 0=Cash)"
    )

    # Save the figure
    fig.write_html(output_path)

    # Print performance statistics
    print(f"\nPerformance Summary for {instrument}:")
    print(f"Strategy Total Return: {cumulative_returns.iloc[-1]:.1f}%")
    print(f"Buy & Hold Return: {buy_hold_returns.iloc[-1]:.1f}%")
    print(f"Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
    print(f"Buy & Hold Sharpe Ratio: {bh_sharpe:.2f}")
    print(f"Number of Trades: {positions.diff().abs().sum() / 2:.0f}")
    print(f"Max Drawdown Strategy: {strategy_dd.min():.1f}%")
    print(f"Max Drawdown Buy&Hold: {bh_dd.min():.1f}%")
    print(f"\n✅ Performance plot has been saved as '{output_path}'")

if __name__ == "__main__":
    try:
        # Check if running in Jupyter
        get_ipython()
        is_jupyter = True
    except:
        is_jupyter = False

    if is_jupyter:
        # Default values when running in Jupyter
        run_forecast(instrument='GC', start_year=1962)
    else:
        # Command line argument parsing when running as script
        parser = argparse.ArgumentParser(description='Run forecast strategy on financial instrument')
        parser.add_argument('--instrument', type=str, default='SPX',
                          help='Instrument to analyze (must be a column in Xdaily.csv)')
        parser.add_argument('--start_year', type=int, default=1962,
                          help='Start year for analysis')
        args = parser.parse_args()
        run_forecast(args.instrument, args.start_year)
