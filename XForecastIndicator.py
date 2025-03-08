import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import sys
import os

# Create logs directory if it doesn't exist
log_file = '/Users/stevenmichiels/Repos/stevenmichiels.github.io/forecast.log'
try:
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging with error handling
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode='a')
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    # Test the logger
    logger.info("Logging initialized successfully")
    
except Exception as e:
    print(f"Error setting up logging: {str(e)}")
    # Fallback to console-only logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.warning(f"Falling back to console-only logging due to error: {str(e)}")

class ForecastIndicator:
    def __init__(self, base_period=8, include='16-64-256', fc_cap=20, ema1=2, ema2=4):
        logger.info(f"Initializing ForecastIndicator with base_period={base_period}, include={include}")
        self.base_period = base_period
        self.include = include
        self.fc_cap = fc_cap
        self.ema1 = ema1
        self.ema2 = ema2
        
        # Initialize periods
        self.period1 = base_period
        self.period2 = base_period * 2
        self.period3 = base_period * 4
        
        # Breakout periods - revert to original
        self.bo_period1 = int(round(80 * base_period/16, 0))
        self.bo_period2 = int(round(160 * base_period/16, 0))
        self.bo_period3 = int(round(320 * base_period/16, 0))
        
        # scalars
        self.ewmac_scalars = self._initialize_ewmac_scalars()
        self.breakout_scalars = self._initialize_breakout_scalars()
        self.ewmac_FDM = self._initialize_ewmac_FDM()
        self.breakout_FDM = self._initialize_breakout_FDM()

    def _initialize_ewmac_FDM(self):
        if self.include == '16-64':
            return 1
        else:
            return 1.08
    
    def _initialize_breakout_FDM(self):
        if self.include == '16-64':
            return 1
        else:
            return 1.1
        
    def _initialize_ewmac_scalars(self):
        ewmac8_scalar = 5.95
        ewmac16_scalar = 4.1
        ewmac32_scalar = 2.79
        ewmac64_scalar = 1.91
        
        if self.base_period == 8:
            return [ewmac8_scalar, ewmac16_scalar, ewmac32_scalar]
        elif self.base_period == 16:
            return [ewmac16_scalar, ewmac32_scalar, ewmac64_scalar]
        else:
            scalar1 = ewmac8_scalar - (((self.base_period-8)/8) * (ewmac8_scalar-ewmac16_scalar))
            scalar2 = ewmac16_scalar - (((self.base_period*4-32)/32) * (ewmac16_scalar-ewmac32_scalar))
            scalar3 = ewmac32_scalar - (((self.base_period*16-128)/128) * (ewmac32_scalar-ewmac64_scalar))
            return [scalar1, scalar2, scalar3]

    def _initialize_breakout_scalars(self):
        bo40_scalar = 0.7
        bo80_scalar = 0.73
        bo160_scalar = 0.74
        bo320_scalar = 0.74
        
        if self.base_period == 16:
            return [bo80_scalar, bo160_scalar, bo320_scalar]
        else:
            scalar1 = bo40_scalar - (((self.base_period-8)/8) * (bo40_scalar-bo80_scalar))
            scalar2 = bo80_scalar - (((self.base_period*4-32)/32) * (bo80_scalar-bo160_scalar))
            scalar3 = bo160_scalar - (((self.base_period*16-128)/128) * (bo160_scalar-bo320_scalar))
            return [scalar1, scalar2, scalar3]

    def calculate_ewma_volatility(self, prices, span=32, blend_ratio=0):
        returns = prices.pct_change() * 100
        lambda_ = 2 / (span + 1)
        
        ewma = returns.ewm(alpha=lambda_).mean()
        ewvar = (returns - ewma).pow(2).ewm(alpha=lambda_).mean()
        
        sigma_day = np.sqrt(ewvar)
        sigma_ann = 16 * sigma_day
        
        if blend_ratio > 0:
            sigma_10y = sigma_ann.rolling(2560).mean()
            sigma_blend = ((100-blend_ratio)/100 * sigma_ann) + (blend_ratio/100 * sigma_10y)
        else:
            sigma_blend = sigma_ann
            
        return sigma_blend

    def calculate_ewmac(self, prices):
        ewmac16 = prices.ewm(span=self.period1).mean() - prices.ewm(span=4*self.period1).mean()
        ewmac32 = prices.ewm(span=self.period2).mean() - prices.ewm(span=4*self.period2).mean()
        ewmac64 = prices.ewm(span=self.period3).mean() - prices.ewm(span=4*self.period3).mean()
        
        return ewmac16, ewmac32, ewmac64

    def calculate_breakout(self, prices):
        def calculate_bo(prices, period):
            high = prices.rolling(period).max()
            low = prices.rolling(period).min()
            mid = (high + low) / 2
            return 40 * (prices - mid) / (high - low)
        
        bo80 = calculate_bo(prices, self.bo_period1).ewm(span=self.bo_period1/4).mean()
        bo160 = calculate_bo(prices, self.bo_period2).ewm(span=self.bo_period2/4).mean()
        bo320 = calculate_bo(prices, self.bo_period3).ewm(span=self.bo_period3/4).mean()
        
        # Apply scalars
        bo80 = bo80 * self.breakout_scalars[0]
        bo160 = bo160 * self.breakout_scalars[1]
        bo320 = bo320 * self.breakout_scalars[2]
        
        return bo80, bo160, bo320

    def calculate_forecast(self, prices, include_breakout=True):
        logger.info("Starting forecast calculation")
        sigma_blend = self.calculate_ewma_volatility(prices)
        sigma_price = sigma_blend / 16 / 100 * prices
        
        ewmac16, ewmac32, ewmac64 = self.calculate_ewmac(prices)
        bo80, bo160, bo320 = self.calculate_breakout(prices)
        
        ewmac_signals = [ewmac16, ewmac32, ewmac64]
        bo_signals = [bo80, bo160, bo320]
        scaled_signals = []
        
        for i, (ewmac, scalar) in enumerate(zip(ewmac_signals, self.ewmac_scalars)):
            normalized = ewmac / sigma_price
            scaled = normalized * scalar
            capped = np.clip(scaled, -self.fc_cap, self.fc_cap)
            scaled_signals.append(capped)
            logger.info(f"EWMAC signal {i+1}: scaled={scalar:.2f}, min={capped.min():.2f}, max={capped.max():.2f}")
        
        weights = self._calculate_weights()
        combined_forecast = sum(w * s for w, s in zip(weights, scaled_signals))
        
        logger.info(f"EWMAC FDM: {self.ewmac_FDM}")
        combined_forecast = combined_forecast * self.ewmac_FDM
        # cap the combined forecast
        combined_forecast = np.clip(combined_forecast, -self.fc_cap, self.fc_cap)
        logger.info(f"Combined forecast: min={combined_forecast.min():.2f}, max={combined_forecast.max():.2f}")
        
        if include_breakout:
            if self.include == '16-64':
                breakout_forecast = bo_signals[0]
                logger.info(f"Using single breakout signal: min={bo_signals[0].min():.2f}, max={bo_signals[0].max():.2f}")
            else:
                breakout_forecast = (bo_signals[0] + bo_signals[1] + bo_signals[2]) / 3 * self.breakout_FDM
                logger.info(f"Using combined breakout signals: min={breakout_forecast.min():.2f}, max={breakout_forecast.max():.2f}")
            final_forecast = (combined_forecast + breakout_forecast) / 2
        else:
            final_forecast = combined_forecast
            
        logger.info(f"Final forecast: min={final_forecast.min():.2f}, max={final_forecast.max():.2f}")
        return final_forecast

    def _calculate_weights(self):
        if self.include == '16-64-256':
            return [1/3, 1/3, 1/3]
        else:
            return [1/2, 0, 0]

def plot_forecast(indicator, prices, include_breakout=False, save_html=False, output_path='/Users/stevenmichiels/pst/forecast_plot.html'):
    forecast = indicator.calculate_forecast(prices, include_breakout=include_breakout)
    
    # Ensure we're working with aligned data
    common_index = prices.index.intersection(forecast.index)
    prices_aligned = prices.loc[common_index]
    forecast_aligned = forecast.loc[common_index]
    
    # Create a DataFrame with both price and forecast for hover data
    hover_df = pd.DataFrame({
        'Price': prices_aligned,
        'Forecast': forecast_aligned.round(1)  # Round to 1 decimal place
    })
    
    hover_template = (
        '<b>Date:</b> %{x|%Y-%m-%d}<br>' +
        '<b>Price:</b> %{customdata[0]:.1f}<br>' +  # Changed from .2f to .1f
        '<b>Forecast:</b> %{customdata[1]:.1f}'
    )
    
    forecast_hover_template = (
        '<b>Date:</b> %{x|%Y-%m-%d}<br>' +
        '<b>Price:</b> %{customdata[0]:.1f}<br>' +  # Changed from .2f to .1f
        '<b>Forecast:</b> %{customdata[1]:.1f}'
    )
    
    # Create subplots with shared x-axis - back to 2 rows
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Price", "Forecast"),
        row_heights=[0.6, 0.4]  # Adjust row heights
    )
    
    # Main price plot with linear y-scale
    fig.add_trace(
        go.Scatter(
            x=hover_df.index, 
            y=hover_df['Price'],
            name="Price",
            customdata=hover_df[['Price', 'Forecast']].values,
            hovertemplate=hover_template,
            line=dict(color='blue', width=2),
            hoverlabel=dict(bgcolor='blue', font_color='white')
        ),
        row=1, col=1
    )
    
    # Forecast line in row 2
    fig.add_trace(
        go.Scatter(
            x=hover_df.index, 
            y=hover_df['Forecast'],
            name="Forecast", 
            line=dict(color='gray'),
            customdata=hover_df[['Price', 'Forecast']].values,
            hovertemplate=forecast_hover_template,
            hoverlabel=dict(bgcolor='gray', font_color='white')
        ),
        row=2, col=1
    )
    
    # Add horizontal lines to forecast plot
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2)
    fig.add_hline(y=20, line_dash="dash", line_color="black", row=2)
    fig.add_hline(y=-20, line_dash="dash", line_color="black", row=2)
    
    # Calculate reasonable y-axis range for price plot
    price_min = prices_aligned.min()
    price_max = prices_aligned.max()
    price_range = price_max - price_min
    
    # Add some padding to the range (5% on each side)
    price_min_padded = price_min - 0.05 * price_range
    price_max_padded = price_max + 0.05 * price_range
    
    # Update layout
    fig.update_layout(
        title_text="Price and Forecast",
        height=800,  # Adjusted height for two plots
        showlegend=True,
        xaxis_title="",
        xaxis2_title="Date",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update price y-axis
    fig.update_yaxes(
        title_text="Price", 
        range=[price_min_padded, price_max_padded],
        row=1, col=1,
        showgrid=True,
        gridcolor='lightgrey'
    )
    
    # Update forecast y-axis
    fig.update_yaxes(
        title_text="Forecast",
        range=[-21, 21],
        row=2, col=1,
        showgrid=True,
        gridcolor='lightgrey'
    )
    
    # Update x-axes to ensure they stay in sync
    fig.update_xaxes(matches='x')
    
    if save_html:
        fig.write_html(output_path)
    
    fig.show()

def plot_breakout_scalars(base_period=8):
    indicator = ForecastIndicator(base_period=base_period)
    
    # Calculate scalars for different base periods
    periods = np.linspace(8, 24, 100)
    scalars = [ForecastIndicator(base_period=p)._initialize_breakout_scalars() for p in periods]
    
    scalar1 = [s[0] for s in scalars]
    scalar2 = [s[1] for s in scalars]
    scalar3 = [s[2] for s in scalars]
    
    # Create plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=periods, y=scalar1, name='Scalar 1 (40/80)'))
    fig.add_trace(go.Scatter(x=periods, y=scalar2, name='Scalar 2 (80/160)'))
    fig.add_trace(go.Scatter(x=periods, y=scalar3, name='Scalar 3 (160/320)'))
    
    # Add vertical line at base_period
    fig.add_vline(x=base_period, line_dash="dash", line_color="red")
    
    # Add point markers for current scalars
    current_scalars = indicator.breakout_scalars
    fig.add_trace(go.Scatter(
        x=[base_period]*3,
        y=current_scalars,
        mode='markers',
        marker=dict(size=10, color='red'),
        name=f'Scalars at period={base_period}'
    ))
    
    fig.update_layout(
        title=f'Breakout Scalars vs Base Period (current={base_period})',
        xaxis_title='Base Period',
        yaxis_title='Scalar Value',
        showlegend=True
    )
    
    fig.show()
