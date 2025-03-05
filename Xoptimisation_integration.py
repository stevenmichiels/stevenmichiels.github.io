"""
This file shows how to do a number of different optimisations - 'one shot' and bootstrapping ;
  also entirely in sample, expanding, and rolling windows 

As in chapters 3 and 4 of "Systematic Trading" by Robert Carver (www.systematictrading.org)

Required: pandas / numpy, matplotlib

USE AT YOUR OWN RISK! No warranty is provided or implied.

Handling of NAN's and Inf's isn't done here (except within pandas), 
And there is no error handling!

The bootstrapping method here is not 'block' bootstrapping, so any time series dependence of returns will be lost 

"""

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from copy import copy
import random
import optimisation_functions as optim_func
import handcrafting_functions as handcraft_func


## Get the data

## Let's do some optimisation
## Feel free to play with these
start_year=1970

tickerselection = ['SPX', 'NDX','XOM','CVX','GC','ZN_reg']
tickerselection_port_ref = ['NDX','SPX','XLE','SMH', 'GC']
tickerselection_ref = ['SPX']

tickerselection=['NDX','SPX','XLE','SMH', 'GC']
tickerselection=['SPX','XOM','CVX','GC']

end_year=2025


df_daily=pd.read_csv('Xdaily.csv')
df_daily.index=pd.to_datetime(df_daily['Date'])
del df_daily['Date']
df_daily=df_daily[df_daily.index.year>start_year]
df_daily_returns = df_daily.pct_change()

df_weekly =  df_daily.resample('W').last()
df_weekly_returns = df_weekly.pct_change()


assets = optim_func.pd_readcsv("assetprices.csv")
assets_selected = assets[['SP500','US20']]

df_daily_returns_selection = df_daily_returns[tickerselection]
df_daily_returns_selection = df_daily_returns_selection[(df_daily_returns_selection.index.year>=start_year) & (df_daily_returns_selection.index.year<=end_year)]

df_weekly_returns_selection = df_weekly_returns[tickerselection]


bstrap=optim_func.opt_and_plot(df_daily_returns_selection, "expanding", "bootstrap", equalisemeans=False, equalisevols=True)
print(bstrap.tail(1))
bstrap_weights = bstrap.iloc[-1].to_list()


p_port_bstrap = handcraft_func.Portfolio(df_weekly_returns_selection, use_SR_estimates=False, overwrite_weights=bstrap_weights)
p_port = handcraft_func.Portfolio(df_weekly_returns_selection, use_SR_estimates=False)


"""
    Remember the arguments are:
    data, date_method, fit_method, rollyears=20, equalisemeans=False, equalisevols=True, 
                              monte_carlo=200, monte_length=250
    
    
    opt_and_plot(data, "in_sample", "one_period", equalisemeans=False, equalisevols=False)
    
    opt_and_plot(data, "in_sample", "one_period", equalisemeans=False, equalisevols=True)
    
    opt_and_plot(data, "in_sample", "one_period", equalisemeans=True, equalisevols=True)
    
    opt_and_plot(data, "in_sample", "bootstrap", equalisemeans=False, equalisevols=True, monte_carlo=500)
    
    opt_and_plot(data, "rolling", "one_period", rollyears=1, equalisemeans=False, equalisevols=True)
    
    opt_and_plot(data, "rolling", "one_period", rollyears=5, equalisemeans=False, equalisevols=True)
    
    opt_and_plot(data, "expanding", "one_period", equalisemeans=False, equalisevols=True)
    
    opt_and_plot(data, "expanding", "bootstrap", equalisemeans=False, equalisevols=True)
    
    """
