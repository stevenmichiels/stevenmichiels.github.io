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


username = 'bb.trendfollowing@gmail.com'
password = 'AnikaSeol1920!!'
datadir = '/Users/stevenmichiels/ETF'

from tvScrape import TvScrape, Interval
tv = TvScrape(username, password)
print("Initializing ETF data")

#For capital of 250000, 33 instruments, Selected order: 
parquet_selection = ['AUD', 'BOBL', 'BRE', 'BTP', 'CHF', 'CNH', 'CORN', 'CRUDE_W_mini', 'DOW', 'EU-OIL', 'EU-TRAVEL', 'EU-UTILS', 'EURCHF', 'FTSECHINAA','GBP', 'GBPEUR', 'IRON', 'KOSDAQ', 'KR10', 'LEANHOG', 'MUMMY','NIKKEI', 'NZD', 'PLAT', 'SOYMEAL', 'VIX', 'WHEAT', 'YENEUR'] 
#For capital of 25000000, 73 instruments, Selected order: ['NASDAQ_micro', 'BRENT-LAST', 'EURCHF', 'CAD', 'REDWHEAT', 'PLAT', 'BUND', 'LEANHOG', 'US10U', 'GAS-LAST', 'NIFTY', 'KOSPI', 'COPPER', 'FTSECHINAA', 'CORN', 'NZD', 'SILVER', 'BOBL', 'GAS_US_mini', 'PALLAD', 'IRON', 'SOYMEAL', 'EU-TRAVEL', 'ETHEREUM', 'EUR', 'MSCISING', 'CNH', 'BTP', 'USIRS10', 'EU-UTILS', 'KOSDAQ', 'LIVECOW', 'MUMMY', 'FTSECHINAH', 'GBPEUR', 'EU-DIV30', 'BONO', 'V2X', 'NIKKEI', 'KR10', 'BRE', 'SOYOIL', 'GOLD_micro', 'YENEUR', 'EU-BASIC', 'FEEDCOW', 'EU-HEALTH', 'CHF', 'JPY', 'FTSETAIWAN', 'GBP', 'BUXL', 'MXP', 'WHEAT', 'DAX', 'SOYBEAN', 'RUSSELL', 'SP400', 'US30', 'DOW', 'EU-TECH', 'US10', 'GASOILINE', 'EU-OIL', 'MSCIASIA', 'US5', 'AUD', 'USIRS5ERIS', 'DJSTX-SMALL', 'OAT', 'HEATOIL', 'AEX', 'BITCOIN']
parquet_selection2 = ['AEX', 'AUD', 'BITCOIN', 'BOBL', 'BONO', 'BRE', 'BRENT-LAST', 'BTP', 'BUND', 'BUXL', 'CAD', 'CHF', 'CNH', 'COPPER', 'CORN', 'DAX', 'DJSTX-SMALL', 'DOW', 'ETHEREUM', 'EU-BASIC', 'EU-DIV30', 'EU-HEALTH', 'EU-OIL', 'EU-TECH', 'EU-TRAVEL', 'EU-UTILS', 'EUR', 'EURCHF', 'FEEDCOW', 'FTSECHINAA', 'FTSECHINAH', 'FTSETAIWAN', 'GAS-LAST', 'GASOILINE', 'GAS_US_mini', 'GBP', 'GBPEUR', 'GOLD_micro', 'HEATOIL', 'IRON', 'JPY', 'KOSDAQ', 'KOSPI', 'KR10', 'LEANHOG', 'LIVECOW', 'MSCIASIA', 'MSCISING', 'MUMMY', 'MXP', 'NASDAQ_micro', 'NIFTY', 'NIKKEI', 'NZD', 'OAT', 'PALLAD', 'PLAT', 'REDWHEAT', 'RUSSELL', 'SILVER', 'SOYBEAN', 'SOYMEAL', 'SOYOIL', 'SP400', 'US10', 'US10U', 'US30', 'US5', 'USIRS10', 'USIRS5ERIS', 'V2X', 'WHEAT', 'YENEUR'] 

parquet_dir_adjusted_ = '/Users/stevenmichiels/Data/parquet/futures_adjusted_prices'
instrumentlist = [num.split('.')[0] for num in os.listdir(parquet_dir_adjusted_)]
instrumentlist=[num for num in instrumentlist if num not in ['.parquet']]
instrumentlist=parquet_selection
#instrumentlist=['CORN.parquet']
df_parquet=pd.DataFrame()

load_parquet=0

def quick_yf(ticker,ticker_final, log_=False):
    yf_=yf.Ticker(ticker)
    yf_=yf_.history(period="max")
    if log_:
        plt.plot(np.log(yf_['Close']))
    else:
        plt.plot(yf_['Close'])
    yf_.index = pd.to_datetime(yf_.index)
    yf_[ticker_final] = yf_['Close']
    to_drop = [col for col in yf_.columns if col not in [ticker_final]]
    yf_.drop(columns=to_drop, inplace=True)

    plt.title(ticker_final)
    plt.show()
    return yf_

def quick_tv(ticker_, exchange_):
    tv_=tv.get_hist(symbol=ticker_,exchange=exchange_,interval='1D',n_bars=30000)
    tv_.rename(columns={'close':ticker_}, inplace=True)
    plt.plot(np.log(tv_[ticker_]))
    plt.title(ticker_)
    plt.show()
    return tv_



def wrangle_parquet(instrument, parquet_dir_adjusted=parquet_dir_adjusted_):
    datafile=os.path.join(parquet_dir_adjusted,instrument+'.parquet')
    df = pd.read_parquet(datafile)
    # drop the time in the index
    df.index = pd.to_datetime(df.index.date)
    # for each date, take only the last price
    df = df.groupby(df.index).last() 
    df.rename(columns={'price':instrument.split('.parquet')[0].capitalize()}, inplace=True)
    # create a duplicate of df, that takes only the last value of each week
    df2 = df.resample('W').last()
    return df, df2


if load_parquet==1:
    for instrument in instrumentlist:
        df,df2 = wrangle_parquet(instrument)
        df_parquet = pd.concat([df_parquet, df], axis=1)

def wrangle_tradingview(symbol_,exchange_, tickername_final,interval_='1D', n_bars_=30000, fut_contract_=0, cutoff_=1900):
    print(tickername_final)
    df = tv.get_hist(symbol=symbol_,exchange=exchange_,interval=interval_,n_bars=n_bars_) if fut_contract_==0 else tv.get_hist(symbol=symbol_,exchange=exchange_,interval=interval_,n_bars=n_bars_,fut_contract=1)
    df = df[df.index.year >= cutoff_]
    df['Date'] = df.index.date
    df.set_index('Date', inplace=True, drop=True)
    df.index = pd.to_datetime(df.index)
    df.drop(columns=['symbol','open','high','low', 'volume'], inplace=True)
    # capitalize the column names
    df.columns = [col.capitalize() for col in df.columns]
    df.columns = [tickername_final for num in df.columns]
    return df


def wrangle_barchart(tickername, tickername_final):
    print(tickername_final)
    df=pd.read_csv(os.path.join(datadir,tickername+'.csv'), header=1)
    # rename Date Time to Date
    df.rename(columns={'Date Time': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True, drop=True)
    df.drop(columns=['Change', 'Open Interest', 'High', 'Low', 'Open', 'Volume'], inplace=True)
    df.columns = [tickername_final for num in df.columns]
    return df


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

tv_IEF = wrangle_tradingview('IEF','NASDAQ','IEF')
tv_US10 = wrangle_tradingview('US10Y','TVC','US10')
tv_US10.index = tv_US10.index.shift(1, freq='D')
yf_ZN = quick_yf('ZN=F', 'ZN')
# make the index of yf_ZN non-timezone aware
yf_ZN.index = yf_ZN.index.tz_localize(None)

# put tv_US10 and yf_ZN in a dataframe
df_bonds = pd.concat([tv_US10, yf_ZN], axis=1)
df_test = tv_US10.copy()
# drop the rows that contain NaN
df_bonds.dropna(inplace=True)
df_bonds['US10_inv'] = df_bonds['US10']
sns.scatterplot(x='US10_inv', y='ZN', data=df_bonds)
# regression
from sklearn.linear_model import LinearRegression
X = df_bonds['US10_inv'].values.reshape(-1,1)
y = df_bonds['ZN'].values
reg = LinearRegression().fit(X, y)
print(reg.coef_, reg.intercept_)
plt.plot(X, reg.predict(X), color='red')
plt.show()


# plot the residual errors of the regression
residuals = y - reg.predict(X)
plt.plot(residuals)
plt.title('Residual errors of the regression')
plt.show()


import seaborn as sns

sns.lmplot(x='US10_inv', y='ZN', data=df_bonds, ci=95, line_kws={'lw':1})
plt.show()

df_test['ZN'] = reg.predict(df_test['US10'].values.reshape(-1,1))
df_test['ZN_reg'] = -7.72190816*df_test['US10'] + 144.3629930747892
plt.plot(df_test.ZN_reg)

