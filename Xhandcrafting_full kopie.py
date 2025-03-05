# This is the *full* handcrafting code
# It can be used for long only
# It is *not* the code actually used in pysystemtrade
# It is completely self contained with no pysystemtrade imports
# CAVEATS:
# Uses weekly returns (resample needed first)
# Doesn't deal with missing assets
from copy import copy
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
FLAG_BAD_RETURN = -9999999.9
from scipy.optimize import minimize
from collections import namedtuple
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import Xhandcrafting_full_functions as hcfunc

tickerselection = ['SPX','SMH','XOM','CVX', 'GC', 'BTCUSD']
tickerselection = ['SPX','SMH','XOM', 'CVX', 'GC','DBC','HG','CL', 'TDIV', 'ZN_reg']
tickerselection = ['NDX','SPX','XLE', 'GC', 'HG']

tickerselection_port_ref = ['NDX','SPX','XLE','SMH', 'GC']
tickerselection_ref = ['SPX']

start_year = 1966
risk_target_ = 10
use_SR_estimates_ = False

gold_risk_weight = 0.15
interest_rounding = 0.25
interest_rounding_factor = int(1/interest_rounding)
interest_delta = 2


def calc_pd_diversification_mult(corr_matrix,vol_weights ):
        """
        Calculates the diversification multiplier for a portfolio
        :return: float
        """

        #corr_matrix = self.corr_matrix.values
        #vol_weights = np.array(self.volatility_weights)

        div_mult = 1.0 / (
            (np.dot(np.dot(vol_weights, corr_matrix), vol_weights.transpose())) ** 0.5
        )

        return div_mult


def norm_weights(list_of_weights):
    norm_weights = list(np.array(list_of_weights) / np.sum(list_of_weights))
    return norm_weights



def vol_to_cash_weights(vol_weights, instrument_std):
    raw_cash_weights = [
        vweight / vol for vweight, vol in zip(vol_weights, instrument_std)
    ]
    raw_cash_weights = norm_weights(raw_cash_weights)
    return raw_cash_weights


# join all the elements in the list with a comma

datadir = '/Users/stevenmichiels/ETF'
df_daily = pd.read_csv(os.path.join(datadir,'Xdaily.csv'), index_col=0)
df_daily.index = pd.to_datetime(df_daily.index)
df_daily = df_daily[df_daily.index.year>=start_year]
df_weekly =  df_daily.resample('W').last()
df_weekly['US10r'] = np.round(df_weekly['US10']*interest_rounding_factor)/interest_rounding_factor
df_weekly['US10rd'] = (df_weekly['US10r']-interest_delta).clip(lower=0)
df_monthly =  df_daily.resample('ME').last()
df_yearly =  df_daily.resample('YE').last()
df_daily_returns = df_daily.pct_change()*100
df_weekly_returns = df_weekly.pct_change()*100
df_yearly_returns = df_yearly.pct_change()*100
weekly_returns_selection = df_weekly_returns[tickerselection]
weekly_returns_ref = df_weekly_returns[tickerselection_ref]
weekly_returns_port_ref = df_weekly_returns[tickerselection_port_ref]

df_daily_returns_selection = df_daily_returns[tickerselection]

weekly_returns_gold = df_weekly_returns[['GC']]

np.mean(df_weekly['US10rd'][df_weekly.index.year>2000])
plt.plot(df_weekly['US10rd'])


p_port_ref = hcfunc.Portfolio(weekly_returns_port_ref, use_SR_estimates=False)
gold_vol = p_port_ref._diags_as_dataframe().cash.calcs.loc['Std.','GC']


p = hcfunc.Portfolio(weekly_returns_selection, use_SR_estimates=use_SR_estimates_)
#df_p = p._diags_as_dataframe().cash.calcs
#df_p.loc['Vol weights'] = (1-gold_risk_weight)*df_p.loc['Vol weights']
# drop the final row of df_p
#df_p = df_p.drop(df_p.index[-1])
# add the gold risk weight
#df_p.loc['Vol weights','GC'] = gold_risk_weight
#df_p.loc['Std.','GC'] = gold_vol
#df_p.loc['Cash weights'] = vol_to_cash_weights(df_p.loc['Vol weights'], df_p.loc['Std.'])

p._diags_as_dataframe().calcs
p.corr_matrix
#p._diags_as_dataframe().aggregate.calcs
#p._diags_as_dataframe().cash.calcs

#vola=p._diags_as_dataframe().cash.calcs.loc['Vol weights'].values
#st=p._diags_as_dataframe().cash.calcs.loc['Std.'].values
#vol_to_cash_weights(vola,st)
#p._diags_as_dataframe().cash.calcs


port_returns = p.portfolio_returns
port_cash = p.cash_weights
port = dict(zip(tickerselection,port_cash))
port_list = [str(np.round(m*100,1))+'% '+n for m,n in zip(port_cash,tickerselection)]
port_string = 'Cash: ' +'<br>' +  ' + '.join(port_list[:4])+'<br>' +  ' + '.join(port_list[4:])
port_cumul, port_dd, port_mean, port_CAGR, port_CAGR_approx, port_total_return, port_sigma, port_sharpe, port_years = hcfunc.calc_metrics(p)
print("Portfolio weights: ")
print(p.diags.cash)
print("Portfolio: ")
print(p.show_subportfolio_tree())
print("Cash: ")
print(1-sum(p.cash_weights))
      


# make a string of the first value in port, the first key in port, the second value in port and so on
# So I want: 0.2 * SPX + 0.1 * XOM + ...



p_ref = hcfunc.Portfolio(weekly_returns_ref)
port_returns_ref = p_ref.portfolio_returns
port_cumul_ref, port_dd_ref, port_mean_ref, port_CAGR_ref, port_CAGR_approx_ref, port_total_return_ref, port_sigma_ref, port_sharpe_ref, port_years_ref = hcfunc.calc_metrics(p_ref)

# ceil to the nearest 100

port_total_return = np.ceil(port_total_return/100)*100
tickerstring = ' + '.join(tickerselection)
metricstring = 'Total return '+str(np.round(port_total_return,1)) + '%' +  '<br>' + 'Yearly CAGR ' + str(np.round(port_CAGR,2)) +  '%' + '<br>' + 'Sigma: ' + str(np.round(port_sigma,1))+'%' + '<br>' +'Sharpe: '+str(np.round(port_sharpe,2))
metricstring_ref = 'Total return '+str(np.round(port_total_return_ref,1)) + '%' +'<br>' + 'Yearly CAGR ' + str(np.round(port_CAGR_ref,2)) +  '%' + '<br>' + 'Sigma: ' + str(np.round(port_sigma_ref,1))+'%' +'<br>'+ 'Sharpe: '+str(np.round(port_sharpe_ref,2))
ddstring = 'Max drawdown: ' + str(np.round(port_dd.max(),1)) + '%' + '<br>' + 'Mean drawdown: ' + str(np.round(port_dd.mean(),1)) + '%'
ddstring_ref = 'Max drawdown: ' + str(np.round(port_dd_ref.max(),1)) + '%' + '<br>' + 'Mean drawdown: ' + str(np.round(port_dd_ref.mean(),1)) + '%'
title_text = tickerstring  + '<br>' + metricstring
title_text_small = tickerstring  + '<br>' + metricstring
title_text_small_ref = 'SPX' +'<br>' + 'Total return '+str(np.round(port_total_return_ref,1)) + '%' +'<br>' + 'Yearly CAGR ' + str(np.round(port_CAGR_ref,2)) +  '%' + '<br>' + 'Sigma: ' + str(np.round(port_sigma_ref,1))+'%' +'<br>'+ 'Sharpe: '+str(np.round(port_sharpe_ref,2))
port_cumul_ceil = int(np.ceil(np.max([np.max(port_cumul),np.max(port_cumul_ref)])/100)*100)
# using plotly.go, subplots of port_returns.cumsum() and port_returns_ref.cumsum()
# size of figure is 1000x500
# title is title_text
# hovertemplate shows date and cumulative return with 1 digit

fig = make_subplots(rows=2, cols=2, shared_xaxes=True, subplot_titles=(port_string, '1*SPX'), row_heights=[1, 0.4], vertical_spacing=0.05)
fig.add_trace(go.Scatter(x=port_cumul.index, y=port_cumul, mode='lines', name='cumulative returns', marker_color='blue'), row=1, col=1)
fig.add_trace(go.Scatter(x=port_cumul_ref.index, y=port_cumul_ref, mode='lines', name='cumulative returns', marker_color='black'), row=1, col=2)
fig.add_trace(go.Scatter(x=port_dd.index, y=-port_dd, mode='lines', name='drawdown', marker_color='blue'), row=2, col=1)
fig.add_trace(go.Scatter(x=port_dd_ref.index, y=-port_dd_ref, mode='lines', name='drawdown', marker_color='black'), row=2, col=2)
# on the second row, add a horizontal line at port_drawdown.mean()
fig.add_hline(y=-port_dd.mean(), line_dash="dot", line_color="blue", row=2, col=1)
fig.add_hline(y=-port_dd_ref.mean(), line_dash="dot", line_color="black", row=2, col=2)
# hovertemplate with 1 digit
fig.update_traces(hovertemplate='Date: %{x} <br>Cumulative return: %{y:.1f}' + '%', row=1)
fig.update_traces(hovertemplate='Date: %{x} <br>Drawdown: %{y:.1f}' + '%', row=2)
fig.update_xaxes(title_font_family="Arial")
# y-labels: 'Drawdown (%)' and 'Cumulative return (%)'
fig.update_yaxes(title_text='Cumulative return (%)', row=1, col=1)
fig.update_yaxes(title_text='Cumulative return (%)', row=1, col=2)
fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
fig.update_yaxes(title_text='Drawdown (%)', row=2, col=2)
# make the y-limits the same
fig.update_yaxes(range=[-50, port_cumul_ceil], row=1, col=1)
fig.update_yaxes(range=[-50, port_cumul_ceil], row=1, col=2)
fig.update_yaxes(range=[-100, 0], row=2, col=1)
fig.update_yaxes(range=[-100, 0], row=2, col=2)
fig.update_layout(showlegend=False, width=1200, height=800)
# in each subplot, put a top left text box with the sharpe ratio
fig.add_annotation(xref="paper", yref="paper", x=0, y=port_cumul_ceil-100,
            text=metricstring, showarrow=False, font=dict(family="Courier New, monospace", size=14, color="#ffffff"), align="left", bordercolor="black", borderwidth=1, borderpad=4, bgcolor="blue", opacity=0.5, row=1, col=1)
fig.add_annotation(xref="paper", yref="paper", x=0, y=-80,
            text=ddstring, showarrow=False, font=dict(family="Courier New, monospace", size=14, color="#ffffff"), align="left", bordercolor="black", borderwidth=1, borderpad=4, bgcolor="blue", opacity=0.5, row=2, col=1)
fig.add_annotation(xref="paper", yref="paper", x=0, y=-80,
            text=ddstring_ref, showarrow=False, font=dict(family="Courier New, monospace", size=14, color="#ffffff"), align="left", bordercolor="black", borderwidth=1, borderpad=4, bgcolor="black", opacity=0.5, row=2, col=2)
fig.add_annotation(xref="paper", yref="paper", x=0, y=port_cumul_ceil-100,
            text=metricstring_ref, showarrow=False, font=dict(family="Courier New, monospace", size=14, color="#ffffff"), align="left", bordercolor="black", borderwidth=1, borderpad=4, bgcolor="black", opacity=0.5, row=1, col=2)
# omit the legend

fig.show()
fig.write_html("Xport.html")


fig = go.Figure()
fig.add_trace(go.Scatter(x=port_returns.index, y=port_returns.cumsum(), mode='lines', name='cumulative returns'))
fig.update_layout(title=tickerstring)
fig.update_layout(title_x=0.5)
fig.update_layout(
    font_family="Courier New",
    font_color="black",
    title_font_family="Times New Roman",
    title_font_color="black",
    legend_title_font_color="green"
)
# hovertemplate with 1 digit
fig.update_traces(hovertemplate='Date: %{x} <br>Cumulative return: %{y:.1f}' + '%')
fig.update_xaxes(title_font_family="Arial")
fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.98,
            text=metricstring, showarrow=False, font=dict(family="Courier New, monospace", size=16, color="#ffffff"), align="left", bordercolor="black", borderwidth=1, borderpad=4, bgcolor="blue", opacity=0.8)
# omit the legend
fig.update_layout(showlegend=False, width=1200, height=800)
fig.show()


p.sub_portfolios
#p.sub_portfolios[0].instruments
#p.sub_portfolios[1].instruments

p.show_subportfolio_tree()

for sub in np.arange(len(p.sub_portfolios)):
    print(p.sub_portfolios[sub].instruments)
    print(str(np.round(p.sub_portfolios[sub].volatility_weights,3)))
    print(p.sub_portfolios[sub].diags)
    print(p.sub_portfolios[1].diags.aggregate)

vol_weights =dict([(instr,wt) for instr,wt in zip(p.instruments, p.volatility_weights)])
cash_weights = dict([(instr,wt) for instr,wt in zip(p.instruments, p.cash_weights)])

1-sum(p.cash_weights)

p.diags.cash


[' Contains 3 sub portfolios',
 ["[0] Contains ['NDX', 'SPX', 'SMH']"],
 ["[1] Contains ['XLE']"],
 ["[2] Contains ['GC']"]]