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

datadir = '/Users/stevenmichiels/pst'
df_daily = pd.read_csv(os.path.join(datadir,'Xdaily.csv'), index_col=0)
df_daily.index = pd.to_datetime(df_daily.index, format='mixed')  # First convert to datetime
df_daily.index = pd.to_datetime(df_daily.index.date)  # Then extract date and convert back to datetime

df_weekly =  df_daily.resample('W').last()
df_monthly =  df_daily.resample('M').last()
df_yearly =  df_daily.resample('Y').last()
df_daily_returns = df_daily.pct_change()*100
df_weekly_returns = df_weekly.pct_change()*100
df_monthly_returns = df_monthly.pct_change()*100
df_yearly_returns = df_yearly.pct_change()*100

# calculate the geometric mean return from df_yearly_returns for SPX
def geom_mean(ticker, df_daily, tf='M', year_begin=1920, year_end=2023):
    tf_map=dict(zip(['D','W','M'],[256,52,12]))
    df_tf = df_daily.resample(tf).last()
    df_returns = df_tf.pct_change()*100
    df_returns = (df_returns[ticker][~df_returns[ticker].isna()]/100)
    df_returns_selection = df_returns[(df_returns.index.year>=year_begin) & (df_returns.index.year<=year_end)]
    geom_mean = (df_returns_selection+1).prod()**(1/len(df_returns_selection))-1
    sigma = df_returns_selection.std()

    return geom_mean, sigma


# calculate the geometric mean return from df_yearly_returns for SPX
def geom_mean(ticker, df_daily, tf='M', year_begin=1920, year_end=2023):
    tf_map=dict(zip(['D','W','M','Y'],[256,52,12,1]))
    df_tf = df_daily.resample(tf).last()
    df_returns = df_tf.pct_change()*100
    df_returns = (df_returns[ticker][~df_returns[ticker].isna()]/100)
    df_returns_selection = df_returns[(df_returns.index.year>=year_begin) & (df_returns.index.year<=year_end)]
    geom_mean = tf_map[tf]*((df_returns_selection+1).prod()**(1/len(df_returns_selection))-1)
    sigma = np.sqrt(tf_map[tf])*df_returns_selection.std()
    print('Geometric mean return of ' + ticker + ': ' + str(np.round(100*geom_mean,2)) + '%')
    print('Sigma : ' + str(np.round(100*sigma,2)) + '%')
    return geom_mean, sigma

def geom_mean_approx(ticker, df_returns, tf='D',year_begin=1920, year_end=2023):
    tf_map=dict(zip(['D','W','M','Y'],[256,52,12,1]))
    df_returns = (df_returns[ticker][~df_returns[ticker].isna()]/100)
    df_returns_selection = df_returns[(df_returns.index.year>=year_begin) & (df_returns.index.year<=year_end)]
    geom_mean = tf_map[tf]**(df_returns_selection+1).prod()**(1/len(df_returns_selection))-1
    geom_mean_approx = tf_map[tf]*(df_returns_selection.mean() - 0.5*df_returns_selection.var())
    sigma = np.sqrt(tf_map[tf])*df_returns_selection.std()
    print('Approximated geometric mean return of ' + ticker + ': ' + str(np.round(100*geom_mean_approx,2)) + '%')
    print('Sigma : ' + str(np.round(100*sigma,2)) + '%')
    return geom_mean_approx, sigma

tf_map=dict(zip(['M','Y'],[12,1]))
def bootstrap_return(ticker, df_daily, tf='Y', block_length = 1, horizon_length=5, year_begin=1920, year_end=2023, n_samples=10000):
    tf_map=dict(zip(['M','Y'],[12,1]))
    df_tf = df_daily.resample(tf).last()
    df_returns = df_tf.pct_change()*100
    df_returns = (df_returns[ticker][~df_returns[ticker].isna()]/100)
    df_returns_selection = df_returns[(df_returns.index.year>=year_begin) & (df_returns.index.year<=year_end)]
    dataset_length = df_returns_selection.shape[0]
    horizon_length

    bootstrap_returns = np.zeros(n_samples)
    resampled_returns = list()
    for i in range(n_samples):
        if i%1000==0:
            print("Sample "+str(i)+'/'+str(n_samples))
        resampled_single = list()
        count = 0
        while count < horizon_length:
            start_index = np.random.randint(0, dataset_length-block_length)
            resampled_single.append(df_returns_selection.iloc[start_index:start_index+block_length].values)
            count += block_length        
        resampled_single = np.hstack(resampled_single)
        resampled_returns.append(resampled_single)
        n = int(horizon_length/block_length)
        factor = tf_map[tf]/block_length
        bootstrap_returns[i] = factor*(((resampled_returns[i]+1).prod()**(1/n)-1)*100)
    return resampled_returns, bootstrap_returns

ticker_='SPX'
tf_='M'
block_length_=1
horizon_length_=240
year_begin_ = 1920
year_end=year_end_ = 2025

geom, sigma = geom_mean(ticker_, df_daily,tf_)

resampled_returns,bootstrap_returns = bootstrap_return(ticker=ticker_, df_daily=df_daily, tf=tf_, block_length = block_length_, horizon_length=horizon_length_, year_begin=year_begin_, year_end=year_end_)

print(str(np.quantile(bootstrap_returns, [0.5])))


fig, ax = plt.subplots()
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.autolayout"] = True
sns.set_style("darkgrid",{'axes.grid' : True})
sns.set_theme(font_scale = 1.5)

sns.histplot(bootstrap_returns, bins=50, kde=True)

bootstrap_quantiles = np.quantile(bootstrap_returns, [0.1,0.5,0.9])
plt.axvline(bootstrap_quantiles[1], color='blue', linestyle='dashed', linewidth=2)
# add a text box with the median value, at the top
textstr = '10th quantile'
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
# at the top of the figure, add a text box with the median value


plt.text(bootstrap_quantiles[1], 0, str(np.round(bootstrap_quantiles[1],1)), rotation=0, verticalalignment='top')
plt.axvline(bootstrap_quantiles[0], color='red', linestyle='dashed', linewidth=1)
plt.text(bootstrap_quantiles[0], 0, str(np.round(bootstrap_quantiles[0],1)), rotation=0, verticalalignment='top')
plt.axvline(bootstrap_quantiles[2], color='green', linestyle='dashed', linewidth=1)
plt.text(bootstrap_quantiles[2], 0, str(np.round(bootstrap_quantiles[2],1)), rotation=0, verticalalignment='top')
plt.title(ticker_ + ', ' + str(int(horizon_length_/tf_map[tf_])) + ' year horizon \n Yearly geometric return [%]')
# don't show x-ticks
plt.xticks([])
plt.show()

# order = np.argsort(bootstrap_returns)
# bootstrap_returns = bootstrap_returns[order]
# resampled_returns = [resampled_returns[i] for i in order]

# plt.plot(bootstrap_returns)
# plt.show()
# print("median: " + str(bootstrap_returns[5000]))
# plt.plot(resampled_returns[500].cumsum())
# plt.plot(resampled_returns[5000].cumsum())
# plt.plot(resampled_returns[9500].cumsum())
# plt.legend(['5%','50%','95%'])
# plt.show()
