import numpy as np
from sklearn.utils import resample
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import sys
import seaborn as sns
import plotly.express as px
import logging
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='etf_initialization.log'
)

# Get credentials from environment variables
username = os.getenv('TV_USERNAME')
password = os.getenv('TV_PASSWORD')
datadir = os.getenv('DATA_DIR')

from tvScrape import TvScrape, Interval
tv = TvScrape(username, password)
logging.info("Initializing ETF data")

# Constants
DEFAULT_BARS = 30000
CUTOFF_YEAR = 1900
DEFAULT_INTERVAL = '1D'

def quick_yf(ticker, log_=False):
    yf_=yf.Ticker(ticker)
    yf_=yf_.history(period="max")
    if log_:
        plt.plot(np.log(yf_['Close']))
    else:
        plt.plot(yf_['Close'])
    plt.title(ticker)
    plt.show()
    return yf_

def quick_tv(ticker_, exchange_):
    tv_=tv.get_hist(symbol=ticker_,exchange=exchange_,interval='1D',n_bars=30000)
    tv_.rename(columns={'close':ticker_}, inplace=True)
    plt.plot(np.log(tv_[ticker_]))
    plt.title(ticker_)
    plt.show()
    return tv_

def wrangle_tradingview(symbol_, exchange_, tickername_final, interval_='1D', n_bars_=30000, fut_contract_=0, cutoff_=1900):
    try:
        print(f"Fetching data for {tickername_final}")
        df = tv.get_hist(
            symbol=symbol_,
            exchange=exchange_,
            interval=interval_,
            n_bars=n_bars_,
            fut_contract=fut_contract_ if fut_contract_ else None
        )
        
        if df.empty:
            raise ValueError(f"No data received for {tickername_final}")
            
        df = df[df.index.year >= cutoff_]
        if df.empty:
            raise ValueError(f"No data after cutoff year {cutoff_} for {tickername_final}")
            
        df['Date'] = df.index.date
        df.set_index('Date', inplace=True, drop=True)
        df.index = pd.to_datetime(df.index)
        df.drop(columns=['symbol','open','high','low', 'volume'], inplace=True)
        df.columns = [tickername_final] * len(df.columns)
        
        return df
        
    except Exception as e:
        print(f"Error processing {tickername_final}: {str(e)}")
        return pd.DataFrame()


def wrangle_barchart(tickername, tickername_final):
    try:
        file_path = os.path.join(datadir, tickername + '.csv')
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return pd.DataFrame()
            
        # Read CSV file with error handling
        try:
            df = pd.read_csv(file_path, header=1)
        except pd.errors.EmptyDataError:
            print(f"Empty or corrupt file: {file_path}")
            return pd.DataFrame()
            
        print(f"Columns in {tickername}.csv: {df.columns.tolist()}")
        
        # Try different possible date column names
        date_column = None
        for col in ['Date Time', 'Date', 'DateTime', 'Timestamp']:
            if col in df.columns:
                date_column = col
                break
                
        if date_column is None:
            print(f"No date column found in {tickername}.csv")
            return pd.DataFrame()
            
        # Rename the date column to 'Date'
        df.rename(columns={date_column: 'Date'}, inplace=True)
        
        # Convert to datetime with error handling
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            print(f"Error converting dates in {tickername}.csv: {str(e)}")
            print("First few date values:", df['Date'].head())
            return pd.DataFrame()
            
        df.set_index('Date', inplace=True, drop=True)
        
        # Drop columns if they exist
        columns_to_drop = ['Change', 'Open Interest', 'High', 'Low', 'Open', 'Volume']
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        if existing_columns:
            df.drop(columns=existing_columns, inplace=True)
            
        df.columns = [tickername_final for _ in df.columns]
        return df
    except Exception as e:
        print(f"Error processing {tickername}.csv: {str(e)}")
        return pd.DataFrame()


def update_tv_with_yf(tv, tv_ticker, yf_ticker="^GSPC", overwrite_=True):
    yf_ = yf.Ticker(yf_ticker)
    yf_ = yf_.history(period="max")
    yf_ = yf_[['Close']]
    yf_.columns = [tv_ticker]
    yf_.index = pd.to_datetime(yf_.index)
    yf_.index = yf_.index.date
    tv.update(yf_, overwrite=overwrite_)
    return tv

# daily and weekly
tv_SPX_init = wrangle_tradingview('SPX','SP','SPX')
tv_SPX = update_tv_with_yf(tv_SPX_init, 'SPX')
tv_NDX_init = wrangle_tradingview('NDX','NASDAQ','NDX')
tv_NDX = update_tv_with_yf(tv_NDX_init, 'NDX')

tv_CL = wrangle_tradingview('USOIL','TVC','CL')
tv_GC = wrangle_tradingview('GC','COMEX','GC', fut_contract_=1)
tv_IEF = wrangle_tradingview('IEF','NASDAQ','IEF')
tv_XAUUSD = wrangle_tradingview('XAUUSD','OANDA','GC')
tv_XAGUSD = wrangle_tradingview('XAGUSD','OANDA','SI')
tv_EURUSD = wrangle_tradingview('EURUSD','OANDA','EURUSD')
tv_US10 = wrangle_tradingview('US10Y','TVC','US10')
# shift the index of tv_XAUUSD with one row
tv_XAUUSD.index = tv_XAUUSD.index.shift(1, freq='D')
tv_XAGUSD.index = tv_XAGUSD.index.shift(1, freq='D')
tv_EURUSD.index = tv_EURUSD.index.shift(1, freq='D')
tv_US10.index = tv_US10.index.shift(1, freq='D')

tv_VIX = wrangle_tradingview('VIX','TVC','VIX')
tv_DXY = wrangle_tradingview('DXY','INDEX','DXY')
tv_DJI = wrangle_tradingview('DJI','TVC','DJI')
tv_HG = wrangle_tradingview('HG1!','COMEX','HG')
tv_BTCUSD = wrangle_tradingview('BTCUSD','BITSTAMP','BTCUSD')
tv_SX5E = wrangle_tradingview('SX5E','TVC','SX5E')
tv_CORN = wrangle_tradingview('CORN','MIL','CORN')

tv_KOSDAQ = wrangle_tradingview('KOSDAQ','KRX','KOSDAQ')
tv_KOSPI = wrangle_tradingview('KOSPI','KRX','KOSPI')
tv_SOX = wrangle_tradingview('SOX','NASDAQ','SOX')
tv_IWDA=wrangle_tradingview('EUNL','XETR','IWDA')
tv_DAX = wrangle_tradingview('DAX','XETR','DAX')

tv_list = [tv_SPX,tv_NDX,tv_SOX, tv_CL, tv_XAUUSD,tv_IEF,tv_XAGUSD, tv_EURUSD,tv_VIX, tv_US10, tv_DXY, tv_DJI,tv_SX5E, tv_HG, tv_BTCUSD, tv_CORN]
tv_list_tickers = ['SPX','NDX','SOX','CL','GC','IEF','SI','EURUSD','VIX','US10','DXY','DJI','SX5E','HG','BTCUSD', 'CORN']

yf_list = ["IWM","SMH","XLK","XLE","XLF","XLV","XLC","XLY","DBC","COPX", "^KQ11","^KS11", 'EUNL.DE','^GDAXI','TDIV.AS','EXH5.DE','XOM','CVX','SCCO','MSFT', 'AAPL', 'AMD', 'NVDA', 'TSLA','BRK-B', 'AGG', 'NUE', 'RS', 'STLD', 'CCJ']
yf_list_final = ['IWM','SMH','XLK','XLE','XLF',"XLV",'XLC','XLY','DBC','COPX', 'KOSDAQ','KOSPI','IWDA', 'DAX','TDIV','EXH5','XOM','CVX','SCCO','MSFT', 'AAPL', 'AMD', 'NVDA','TSLA','BRK.B', 'AGG', 'NUE', 'RS', 'STLD', 'CCJ']

ticker = 'SPX'


# join tv_SPX, tv_VIX, tv_TNX, tv_DXY, tv_DJI, tv_HG, tv_BTCUSD, tv_IWDA using the pd.join
daily = tv_SPX.copy()
for index,df in enumerate(tv_list[1:]):
    print(index+1)
    daily[tv_list_tickers[index+1]] = df[tv_list_tickers[index+1]]

# in daily, fill the na-values using interpolation
daily = daily.interpolate(method='linear')

tickerlist_map = dict(zip(yf_list, yf_list_final))
tickers = yf.Tickers(yf_list)
tickers_daily = tickers.history(period="max")


# make a list of the columns that contain either 'Capital Gains' or 'Dividends'
to_drop=[]
for drop_string in ['Capital Gains','Dividends','Volume','Stock Splits', 'High','Low','Open']:
    to_drop += [num for num in tickers_daily if drop_string in num]

tickers_daily.drop(columns=to_drop, inplace=True)
tickers_daily.columns = tickers_daily.columns.map(' '.join)
tickers_daily.columns = [col.split(' ')[1] for col in tickers_daily.columns]


for column in tickers_daily.columns:
    # if any key of tickerlist_map is in the column name, replace it by the value of that key
    for key in tickerlist_map.keys():
        if key in column:
            tickers_daily.rename(columns={column: column.replace(key, tickerlist_map[key])}, inplace=True)

XLE = wrangle_barchart('XLE', 'XLE')

XLE_adj = wrangle_barchart('XLE_adj','XLE')

DBC = wrangle_barchart('XDBC','DBC')

###### XLE ########
ticker='XLE'
##dividend=tickers_daily['Dividends ' + ticker][~tickers_daily['Close '+ticker].isna()]
XLE_yf=tickers_daily[[ticker]][~tickers_daily[ticker].isna()]
# select all rows of XLE_adj that are not in XLE_yf
XLEx = XLE_adj.loc[~XLE_adj.index.isin(XLE_yf.index)]

###### DBC ########
ticker='DBC'
##dividend=tickers_daily['Dividends ' + ticker][~tickers_daily['Close '+ticker].isna()]
DBC_yf=tickers_daily[[ticker]][~tickers_daily[ticker].isna()]
# select all rows of XLE_adj that are not in XLE_yf
DBCx = DBC.loc[~DBC.index.isin(DBC_yf.index)]

### SOX ###
first_SMH_index = tickers_daily['SMH'][~tickers_daily['SMH'].isna()].index[0].date()
# write first_SMH_index as string 'YYYY-MM-DD'
first_SMH_index = first_SMH_index.strftime('%Y-%m-%d')

SMH_connect= tickers_daily['SMH'][~tickers_daily['SMH'].isna()].iloc[0]
SOX_connect= daily.loc[first_SMH_index,'SOX']
ratio = SMH_connect/SOX_connect
tv_SOX['SMH'] = tv_SOX['SOX']*ratio

tickers_daily.update(XLEx)
tickers_daily.update(tv_SOX, overwrite=False)
tickers_daily.update(DBCx, overwrite=True)

# drop NDX
tickers_daily.drop(columns=[col for col in tickers_daily.columns if 'NDX' in col], inplace=True)

# merge or join daily and ticker_daily
daily = daily.join(tickers_daily, how='outer')
daily = daily.interpolate(method='linear')

#write daily to csv
daily['ZN_reg'] = -7.72190816*daily['US10'] + 144.3629930747892
daily['CASH']=1


def create_cash_returns(df, spread=-0.02):
    """
    Create a CASH column based on US10Y rates minus a spread, with 0% floor
    
    Args:
        df: DataFrame with datetime index and US10Y column
        spread: Spread to subtract from US10Y rate (default -0.02 for -2%)
    
    Returns:
        Series with daily values based on US10Y
    """
    # Calculate the annualized return on cash
    annualized_return = df['US10'] / 100 + spread
    
    # Floor at 0%
    annualized_return = annualized_return.clip(lower=0)
    
    # Convert to daily rate (using ACT/365)
    daily_rates = (1 + annualized_return)**(1/365) - 1
    
    # Calculate cumulative returns starting from 1
    cash_values = (1 + daily_rates).cumprod()
    
    return pd.Series(cash_values, index=df.index, name='CASH')

# Replace the existing CASH column
daily['CASH'] = create_cash_returns(daily)

# Verify the yearly returns
def verify_cash_returns(cash_series):
    """Print yearly returns to verify they're close to 2.5%"""
    yearly_returns = (cash_series
                     .resample('Y')
                     .last()
                     .pct_change()
                     .dropna())
    
    print("Yearly returns:")
    print(yearly_returns.apply(lambda x: f"{x*100:.2f}%"))
    
    mean_return = yearly_returns.mean()
    print(f"\nMean yearly return: {mean_return*100:.2f}%")

verify_cash_returns(daily['CASH'])

daily.to_csv('Xdaily.csv')

daily_to_separate = daily.copy()
daily_to_separate.insert(0, 'DATETIME', daily_to_separate.index.strftime('%Y-%m-%d 23:00:00'))
for column in daily_to_separate.columns[1:]:
    to_write = pd.DataFrame()
    to_write['DATETIME'] = daily_to_separate['DATETIME']
    to_write['price'] = daily_to_separate[column]
    # drop all the first rows with NaN
    to_write = to_write[~to_write['price'].isna()]
    to_write.to_csv(os.path.join(datadir,column+'.csv'), index=False)

plot_separate, plot_log = 0,0

if plot_separate:
    for column in daily.columns:
        test=pd.read_csv(os.path.join(datadir,column+'.csv'))
        if plot_log:
            plt.plot(np.log(test.price), label=column)
        else:
            plt.plot(test.price, label=column)
        plt.title(column)
        plt.show()


# df_daily = pd.read_csv('Xdaily.csv')
# df_daily.rename(columns={df_daily.columns[0]:'Date'}, inplace=True)
# df_daily['Date'] = pd.to_datetime(df_daily['Date'])
# df_daily.set_index('Date', inplace=True, drop=True)

# #  add calendar data
# df_daily['Year'] = pd.DatetimeIndex(daily.index).isocalendar().year
# df_daily['Month'] = daily.index.month
# df_daily['Week'] = pd.DatetimeIndex(daily.index).isocalendar().week
# df_daily['Weekday'] = daily.index.weekday
# df_daily['DayOfYear'] = daily.index.dayofyear

# open_columns = [col for col in tickers_daily.columns if 'Open' in col]
# close_columns = [col for col in tickers_daily.columns if 'Close' in col]
# # get the Open and Close for each year
# yearly_open=tickers_daily.groupby('Year')[open_columns].first()
# yearly_open.columns=[num.split(' ')[1] for num in yearly_open.columns]
# # replace all the 0's with NaN
# yearly_open[yearly_open==0] = np.nan

# yearly_close=tickers_daily.groupby('Year')[close_columns].last()
# yearly_close.columns=[num.split(' ')[1] for num in yearly_close.columns]

# # fill the NaN with the first Close of the year
# yearly_open.update(yearly_close, overwrite=False)


# yearly_returns = pd.DataFrame()
# yearly_returns_perc = pd.DataFrame()
# # create additional columns with the yearly return for each asset
# for ticker in tickerlist_final:
#     yearly_returns[ticker] = (yearly_close[ticker] - yearly_open[ticker]) / yearly_open[ticker]
#     yearly_returns_perc[ticker] = yearly_returns[ticker]*100
# yearly_returns = yearly_returns[1:]
# yearly_returns_perc = yearly_returns_perc[1:]

# # impute SMH with NDX
# for year in np.arange(yearly_returns.NDX[~yearly_returns.NDX.isna()].index[0],yearly_returns.SMH[~yearly_returns.SMH.isna()].index[0]):
#     yearly_returns.loc[1985,'SMH'] = yearly_returns.loc[1985,'NDX']
#     yearly_returns_perc.loc[1985,'SMH'] = yearly_returns_perc.loc[1985,'NDX']


# # calculate the geometric mean return
# geometric_mean_return_exact = pd.Series()
# yearly_sigma = yearly_returns.std()
# for column in yearly_returns.columns:
#     geometric_mean_return_exact[column] = (yearly_returns[column]+1).prod()**(1/len(yearly_returns[column][~yearly_returns[column].isna()]))-1

# # approximation
# geometric_mean_return_approx = yearly_returns.mean()-yearly_returns.var()/2

# # calculate the geometric mean return by approx on 100 bootstrap samples

# def bootstrap_performance(horizon,yearly, n=1000):
#     bootstrap = pd.DataFrame()
#     for i in range(n):
#         yearly_sample = resample(yearly, n_samples=horizon,replace=True) if horizon<len(yearly) else resample(yearly,replace=True)
#         bootstrap.loc[i, 'Geom_mean_return'] = yearly_sample.Return.mean()-yearly_sample.Return.var()/2
#         bootstrap.loc[i, 'Std'] = yearly_sample.Return.std()
#     sns.histplot(100*bootstrap.Geom_mean_return, linewidth=1, edgecolor='black')
#     # xlabel: 'Test'
#     plt.xlabel('Yearly geometric mean return (%)')
#     plt.ylabel('Frequency [-]')
#     plt.title('SP500, horizon = '+str(horizon)+' years')
#     plt.show()
#     return bootstrap


# # histogram of yearly returns using seaborn
# sns.set_theme()
# sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# sns.histplot(yearly['Return'], bins=30, linewidth=1, edgecolor='black') 
# plt.show()
# # barplot of yearly returns using seaborn
# sns.barplot(x=yearly.index, y=yearly['Return'], linewidth=1, edgecolor='black')
# plt.show()


# # show meta information about the history (requires history() to be called first)
# spx.history_metadata

# # show actions (dividends, splits, capital gains)
# spx.actions
# spx.dividends
# spx.splits
# spx.capital_gains  # only for mutual funds & etfs


# data_USDJPY = tv.get_hist(symbol='USDJPY',exchange='OANDA',interval=Interval.in_1D,n_bars=5000)
# data_US02Y = tv.get_hist(symbol='US02Y',exchange='TVC',interval=Interval.in_1D,n_bars=5000)
# data_USCBBS = tv.get_hist(symbol='USCBBS',exchange='ECONOMICS',interval=Interval.in_1D,n_bars=5000)
# # data_USIRYY= tv.get_hist(symbol='USIRYY',exchange='ECONOMICS',interval=Interval.in_daily,n_bars=5000)
# # data_USINTR= tv.get_hist(symbol='USINTR',exchange='ECONOMICS',interval=Interval.in_daily,n_bars=5000)
# # data_USM0= tv.get_hist(symbol='USM0',exchange='ECONOMICS',interval=Interval.in_daily,n_bars=5000)
# data_USCBBS.close.plot(figsize=(15,10))


def export_TV_csv(ticker='NVDA', exchange_='NASDAQ', full=True, n_bars_=2750):
    data_ticker = tv.get_hist(symbol=ticker,exchange=exchange_, n_bars=n_bars_)
    data_ticker_weekly=tv.get_hist(symbol=ticker,exchange=exchange_,interval='1W', n_bars=n_bars_)
    data_ticker = data_ticker if full==False else data_ticker.close.resample('D').fillna('nearest')
    
    if full==False:
        data_ticker.set_index(data_ticker.index.strftime('%-m/%-d/%Y'), inplace=True)
    data_ticker_weekly.set_index(data_ticker_weekly.index.strftime('%-m/%-d/%Y'), inplace=True)
    data_daily=pd.DataFrame()
    data_daily['time']=data_ticker.index
    data_daily['value']=data_ticker.close.values if full==False else data_ticker.values
    data_weekly=pd.DataFrame()
    data_weekly['time']=data_ticker_weekly.index
    data_weekly['value']=data_ticker_weekly.close.values
    appendix='-F' if full == True else ''
    #data_daily.iloc[-5000:].to_csv(ticker+'-D'+appendix+'.csv', sep=',', index=False)
    #data_weekly.iloc[-5000:].to_csv(ticker+'-W.csv', sep=',', index=False)

    data_daily.iloc[-5000:].to_csv(os.path.join(os.getcwd(), 'Tickers', ticker+'-D'+'.csv'), sep=',', index=False)
    data_weekly.iloc[-5000:].to_csv(os.path.join(os.getcwd(), 'Tickers', ticker+'-W.csv'), sep=',', index=False)
    return data_daily, data_weekly

plot_all = 1
if plot_all:
    for ticker in daily.columns:
        plt.figure(figsize=(12, 6))
        # Ensure we're using the datetime index properly
        plt.plot(daily.index, daily[ticker], label=ticker)
        
        # Format x-axis to show dates properly
        plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
        
        # Add grid and proper date formatting
        plt.grid(True, alpha=0.3)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{ticker} Price History')
        plt.legend()
        
        # Use tight layout to prevent date labels from being cut off
        plt.tight_layout()
        plt.show()
        plt.close()  # Close the figure to free memory

# For the individual plots like SPX, you can use:
def plot_ticker(df, ticker_col, title=None):
    """
    Plot a single ticker with proper date formatting
    
    Args:
        df (DataFrame): DataFrame containing the data
        ticker_col (str): Column name to plot
        title (str): Optional custom title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[ticker_col])
    plt.gcf().autofmt_xdate()
    plt.grid(True, alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title or f'{ticker_col} Price History')
    plt.tight_layout()
    plt.show()
    plt.close()

# Example usage:
plot_ticker(tv_SPX, 'SPX')

def validate_data(df, ticker):
    """Validate DataFrame for common issues."""
    if df.empty:
        raise ValueError(f"Empty DataFrame for {ticker}")
    
    if df.isnull().any().any():
        logging.warning(f"Missing values found in {ticker}")
        
    if (df < 0).any().any():
        logging.warning(f"Negative values found in {ticker}")
        
    return df

def fetch_all_tickers(ticker_list):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(wrangle_tradingview, *ticker_params) 
                  for ticker_params in ticker_list]
        results = [f.result() for f in futures]
    return results

def is_test_environment():
    return os.getenv('ENVIRONMENT') == 'test'

if is_test_environment():
    # Use mock data or reduced dataset
    parquet_selection = parquet_selection[:3]
    n_bars = 100

def plot_ticker_interactive(df, ticker_col, title=None):
    """
    Create an interactive plot using plotly
    
    Args:
        df (DataFrame): DataFrame containing the data
        ticker_col (str): Column name to plot
        title (str): Optional custom title
    """
    fig = px.line(
        df,
        x=df.index,
        y=ticker_col,
        title=title or f'{ticker_col} Price History'
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified'
    )
    
    fig.show()

# Example usage:
plot_ticker_interactive(tv_SPX, 'SPX')

# SPX and CASH on the same plot
plot_ticker_interactive(daily, ['SPX', 'CASH'], title='SPX and CASH Prices')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ForecastIndicator:
    def __init__(self, base_period=8, include='16-64-256', fc_cap=20, ema1=2, ema2=4):
        self.base_period = base_period
        self.include = include
        self.fc_cap = fc_cap
        self.ema1 = ema1
        self.ema2 = ema2
        
        # Initialize periods
        self.period1 = base_period
        self.period2 = base_period * 2
        self.period3 = base_period * 4
        
        # Breakout periods
        self.bo_period1 = int(round(80 * base_period/16, 0))
        self.bo_period2 = int(round(160 * base_period/16, 0))
        self.bo_period3 = int(round(320 * base_period/16, 0))
        
        # EWMAC scalars
        self.ewmac_scalars = self._initialize_ewmac_scalars()
        
    def _initialize_ewmac_scalars(self):
        ewmac8_scalar = 5.95
        ewmac16_scalar = 4.1
        ewmac32_scalar = 2.79
        ewmac64_scalar = 1.91
        
        # Interpolate scalars based on base period
        if self.base_period == 8:
            return [ewmac8_scalar, ewmac16_scalar, ewmac32_scalar]
        elif self.base_period == 16:
            return [ewmac16_scalar, ewmac32_scalar, ewmac64_scalar]
        else:
            # Linear interpolation
            scalar1 = ewmac8_scalar - (((self.base_period-8)/8) * (ewmac8_scalar-ewmac16_scalar))
            scalar2 = ewmac16_scalar - (((self.base_period*4-32)/32) * (ewmac16_scalar-ewmac32_scalar))
            scalar3 = ewmac32_scalar - (((self.base_period*16-128)/128) * (ewmac32_scalar-ewmac64_scalar))
            return [scalar1, scalar2, scalar3]

    def calculate_ewma_volatility(self, prices, span=32, blend_ratio=0):
        """Calculate EWMA volatility"""
        returns = prices.pct_change() * 100
        lambda_ = 2 / (span + 1)
        
        # Calculate EWMA and variance
        ewma = returns.ewm(alpha=lambda_).mean()
        ewvar = (returns - ewma).pow(2).ewm(alpha=lambda_).mean()
        
        # Calculate annualized volatility
        sigma_day = np.sqrt(ewvar)
        sigma_ann = 16 * sigma_day
        
        if blend_ratio > 0:
            sigma_10y = sigma_ann.rolling(2560).mean()
            sigma_blend = ((100-blend_ratio)/100 * sigma_ann) + (blend_ratio/100 * sigma_10y)
        else:
            sigma_blend = sigma_ann
            
        return sigma_blend

    def calculate_ewmac(self, prices):
        """Calculate EWMAC indicators"""
        ewmac16 = prices.ewm(span=self.period1).mean() - prices.ewm(span=4*self.period1).mean()
        ewmac32 = prices.ewm(span=self.period2).mean() - prices.ewm(span=4*self.period2).mean()
        ewmac64 = prices.ewm(span=self.period3).mean() - prices.ewm(span=4*self.period3).mean()
        
        return ewmac16, ewmac32, ewmac64

    def calculate_breakout(self, prices):
        """Calculate breakout component"""
        def calculate_bo(prices, period):
            high = prices.rolling(period).max()
            low = prices.rolling(period).min()
            mid = (high + low) / 2
            return 40 * (prices - mid) / (high - low)
        
        bo80 = calculate_bo(prices, self.bo_period1).ewm(span=self.bo_period1/4).mean()
        bo160 = calculate_bo(prices, self.bo_period2).ewm(span=self.bo_period2/4).mean()
        bo320 = calculate_bo(prices, self.bo_period3).ewm(span=self.bo_period3/4).mean()
        
        return (bo80 + bo160 + bo320) / 3

    def calculate_forecast(self, prices, include_breakout=True):
        """Calculate final forecast"""
        # Calculate volatility
        sigma_blend = self.calculate_ewma_volatility(prices)
        sigma_price = sigma_blend / 16 / 100 * prices
        
        # Calculate EWMAC components
        ewmac16, ewmac32, ewmac64 = self.calculate_ewmac(prices)
        
        # Normalize and scale EWMAC signals
        ewmac_signals = [ewmac16, ewmac32, ewmac64]
        scaled_signals = []
        
        for ewmac, scalar in zip(ewmac_signals, self.ewmac_scalars):
            normalized = ewmac / sigma_price
            scaled = normalized * scalar
            capped = np.clip(scaled, -self.fc_cap, self.fc_cap)
            scaled_signals.append(capped)
        
        # Combine forecasts
        weights = self._calculate_weights()
        combined_forecast = sum(w * s for w, s in zip(weights, scaled_signals))
        
        if include_breakout:
            breakout_forecast = self.calculate_breakout(prices)
            final_forecast = (combined_forecast + breakout_forecast) / 2
        else:
            final_forecast = combined_forecast
            
        return final_forecast

    def _calculate_weights(self):
        """Calculate weights based on included signals"""
        if self.include == '16-64-256':
            return [1/3, 1/3, 1/3]
        else:
            return [1/2, 1/2, 0]

def plot_forecast(indicator, prices):
    """Plot the forecast with the price series"""
    forecast = indicator.calculate_forecast(prices)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot prices
    ax1.plot(prices.index, prices)
    ax1.set_title('Price')
    
    # Plot forecast
    ax2.plot(forecast.index, forecast)
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.axhline(y=20, color='gray', linestyle='-')
    ax2.axhline(y=-20, color='gray', linestyle='-')
    ax2.set_title('Forecast')
    
    plt.tight_layout()
    plt.show()


# Load your price data into a pandas Series
prices=daily.SPX

# Create indicator instance
indicator = ForecastIndicator(base_period=8)

# Calculate and plot forecast
plot_forecast(indicator, prices)