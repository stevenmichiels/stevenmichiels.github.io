import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def read_forecast_data(json_path=None):
    """
    Read forecast data from JSON file and convert it to pandas DataFrames.
    
    Parameters:
    -----------
    json_path : str, optional
        Path to the JSON file. If None, will look in the current directory
        and the GitHub repository directory.
        
    Returns:
    --------
    dict
        Dictionary containing:
        - data_df: DataFrame with time series data (dates, prices, forecasts, etc.)
        - metrics: Dict with performance metrics
        - metadata: Dict with metadata about the forecast
    """
    # Try to find the JSON file
    if json_path is None:
        possible_paths = [
            'forecast_data.json',
            os.path.join(os.path.dirname(__file__), 'forecast_data.json'),
            '/Users/stevenmichiels/Repos/stevenmichiels.github.io/forecast_data.json'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                json_path = path
                break
        
        if json_path is None:
            raise FileNotFoundError("Could not find forecast_data.json")
    
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create the main DataFrame with time series data
    df = pd.DataFrame({
        'date': pd.to_datetime(data['dates']),
        'price': data['prices'],
        'forecast': data['forecasts'],
        'cumulative_return': data['cumulative_returns'],
        'buy_hold_return': data['buy_hold_returns'],
        'strategy_drawdown': data['strategy_drawdown'],
        'buy_hold_drawdown': data['bh_drawdown'],
        'position': data['positions']
    }).set_index('date')
    
    # Extract performance metrics
    metrics = {
        'strategy_final_return': float(data['strategy_final_return']),
        'buy_hold_final_return': float(data['bh_final_return']),
        'avg_strategy_drawdown': float(data['avg_strategy_dd']),
        'avg_buy_hold_drawdown': float(data['avg_bh_dd']),
        'strategy_sharpe': float(data['strategy_sharpe']),
        'buy_hold_sharpe': float(data['bh_sharpe'])
    }
    
    # Return dictionary with all components
    return {
        'data_df': df,
        'metrics': metrics,
        'metadata': data['metadata']
    }

def plot_price_and_forecast(data_df, metadata, save_path=None):
    """
    Create a matplotlib figure with price and forecast data.
    
    Parameters:
    -----------
    data_df : pandas.DataFrame
        DataFrame containing the price and forecast data
    metadata : dict
        Dictionary containing metadata about the forecast
    save_path : str, optional
        If provided, save the figure to this path
    """
    # Create figure and axis objects with a single subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
    plt.subplots_adjust(hspace=0.3)

    # Plot price data
    ax1.plot(data_df.index, data_df['price'], label='Price', color='blue', linewidth=1)
    ax1.set_title(f"{metadata['instrument']} Price History\nLast Update: {metadata['last_update']}")
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))  # Show a tick every 5 years
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Plot forecast signal
    ax2.plot(data_df.index, data_df['forecast'], label='Forecast Signal', color='red', linewidth=1)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.axhline(y=20, color='gray', linestyle='--', alpha=0.3)
    ax2.axhline(y=-20, color='gray', linestyle='--', alpha=0.3)
    ax2.set_ylim(-22, 22)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Forecast Signal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Format x-axis for forecast
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()

if __name__ == "__main__":
    try:
        # Read the forecast data
        forecast = read_forecast_data()
        
        # Print some information about the data
        print("\nForecast Data Summary:")
        print("=====================")
        print(f"\nMetadata:")
        for key, value in forecast['metadata'].items():
            print(f"{key}: {value}")
            
        print(f"\nTime Series Data Shape: {forecast['data_df'].shape}")
        print(f"Date Range: {forecast['data_df'].index[0]} to {forecast['data_df'].index[-1]}")
        
        print("\nPerformance Metrics:")
        for key, value in forecast['metrics'].items():
            print(f"{key}: {value:.2f}")
            
        print("\nFirst few rows of the data:")
        print(forecast['data_df'].head())
        
        # Create and show the plot
        plot_price_and_forecast(forecast['data_df'], forecast['metadata'])
        
    except Exception as e:
        print(f"Error reading forecast data: {str(e)}") 