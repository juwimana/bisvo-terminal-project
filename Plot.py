import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from Data import *
import talib
import numpy as np 

my_year_month_fmt = mdates.DateFormatter('%m/%y')

class Plot:
	@staticmethod
	def historical_line_plot(ticker:str,start:datetime.date,end:datetime.date):
		fig, ax = plt.subplots(figsize=(16,9))

		ax.plot(Data.get_close(ticker,start,end).index,\
			Data.get_close(ticker,start,end), label=ticker)

		ax.set_xlabel('Date')
		ax.set_ylabel('Adjusted closing price ($)')
		ax.legend()

		return fig

	@staticmethod
	def rolling_plot(ticker:str,window:int,start:datetime.date,end:datetime.date):
		rolling = Data.get_close(ticker,start,end).rolling(\
				  window=window).mean()
		panel_data = Data.stock_data(ticker,start,end)
		stock = Data.get_close(ticker,start,end)
		# Plot everything by leveraging the very powerful matplotlib package
		fig, ax = plt.subplots(2,1,figsize=(13,8))
		#Calculate full sample mean
		full_sample_mean = stock.mean()
		ax[0].plot(stock.index, stock, label=ticker)
		ax[0].plot(rolling.index, rolling, label=str(window)+' days rolling')
		ax[0].axhline(full_sample_mean,linestyle='--',color='red',label='Full Sample Mean')

		ax[0].set_xlabel('Date')
		ax[0].set_ylabel('Adjusted closing price ($)')
		ax[0].legend()
		ax[1].plot(panel_data.loc[start:end,].index, panel_data.loc[start:end],label='Volume Traded')

		ax[1].set_ylabel('Volume Traded')
		ax[1].xaxis.set_major_formatter(my_year_month_fmt)

		return fig

	@staticmethod
	def momentum_plot(ticker:str,start:datetime.date,end:datetime.date):
		panel_data = Data.stock_data(ticker,start,end)
		# adj_close = Data.get_adj_close(ticker,start,end)
		# high = Data.get_close(ticker,start,end)
		# low = Data.get_low(ticker,start,end)

		rsi = talib.RSI(panel_data["Adj Close"])
		wr = talib.WILLR(panel_data["High"],panel_data["Low"],panel_data["Adj Close"], timeperiod=14)

		fig, (ax_rsi,ax_wr) = plt.subplots(2,1,figsize=(13,8))

		ax_rsi.plot(panel_data.index, [70] * len(panel_data.index), label="overbought")
		ax_rsi.plot(panel_data.index, [30] * len(panel_data.index), label="oversold")
		ax_rsi.plot(panel_data.index, rsi, label="Relative Strength Indicator (RSI)")
		ax_rsi.legend(loc='best')
		ax_rsi.plot(panel_data["Adj Close"])

		ax_wr.plot(panel_data.index, [-20] * len(panel_data.index))
		ax_wr.plot(panel_data.index, [-80] * len(panel_data.index))
		ax_wr.plot(panel_data.index, wr,color="#2413bd",label="Williams %R (WILLR)")
		ax_wr.legend(loc='best')
		ax_wr.plot(panel_data["Adj Close"])

		return fig

	@staticmethod
	def bollinger_bands(ticker:str,start:datetime.date,end:datetime.date):
		panel_data = Data.stock_data(ticker,start,end)
		upperband, middleband, lowerband = talib.BBANDS(panel_data["Adj Close"], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
		fig,ax = plt.subplots(figsize=(13,4))

		ax.plot(panel_data.index,upperband,label="Upperband",linewidth=1)
		ax.plot(panel_data.index,middleband,label="Middleband",linewidth=1)
		ax.plot(panel_data.index,lowerband,label="Lowerband",linewidth=1)
		ax.legend(loc="best")
		ax.xaxis.set_major_formatter(my_year_month_fmt)

		return fig

	@staticmethod
	def ema_plot(ticker:str,span:int, start:datetime.date,end:datetime.date):
		panel_data = Data.stock_data(ticker,start,end)
		close = Data.get_close(ticker,start,end)
		ema_short = close.ewm(span=span, adjust=False).mean()

		# Taking the difference between the prices and the EMA timeseries
		trading_positions_raw = close - ema_short
		trading_positions_raw.tail()

		# Taking the sign of the difference to determine whether the price or the EMA is greater and then multiplying by 1/3
		trading_positions = trading_positions_raw.apply(np.sign) * 1/2
		trading_positions.tail()

		# Lagging our trading signals by one day.
		trading_positions_final = trading_positions.shift(1)
		fig, (ax,ax2) = plt.subplots(2,1,figsize=(13,8))

		ax.plot(close.loc[start:end,].index, close.loc[start:end], label=ticker)
		ax.plot(ema_short.loc[start:end,].index, ema_short.loc[start:end], label ='Span '+str(span)+'-days EMA')
		ax.legend(loc='best')
		ax.set_ylabel('Price in $')
		ax.xaxis.set_major_formatter(my_year_month_fmt)

		ax2.plot(trading_positions_final.loc[start:end,].index, trading_positions_final.loc[start:end],label='Trading position')
		ax2.set_ylabel('Trading position-'+str(span))
		ax2.xaxis.set_major_formatter(my_year_month_fmt)

		return fig

	@staticmethod
	def trend_strategy(ticker:str,threshold:int,start:datetime.date,end:datetime.date):
		panel_data = Data.stock_data(ticker,start,end)
		"""
		This trend strategy is based on both a two-month (i.e., 42 tradingdays) and a one-year 
		(i.e., 252 trading days) trend (i.e., the moving average of the index level for the respective period

		The rule to generate trading signals is the following:
			Buy signal (go long):
				the 42d trend is for the first time SD points above the 252d trend.
			Wait (park in cash):
				the 42d trend is within a range of +/– SD points around the 252d trend.
			Sell signal (go short):
				the 42d trend is for the first time SD points below the 252d trend
		"""

		stock_sd= pd.DataFrame()
		stock_sd['Close'] = panel_data["Adj Close"]
		stock_sd['42d'] = np.round(stock_sd['Close'].rolling(42).mean(), 2)
		stock_sd['252d'] = np.round(stock_sd["Close"].rolling(252).mean(), 2)

		stock_sd["42-252"] = stock_sd["42d"]-stock_sd["252d"]

		SD = threshold
		stock_sd['Regime'] = np.where(stock_sd['42-252'] > SD, 1, 0)
		stock_sd['Regime'] = np.where(stock_sd['42-252'] < -SD, -1, stock_sd['Regime'])

		#performance of regime strategy
		stock_sd['Market'] = np.log(stock_sd['Close'] / stock_sd['Close'].shift(1))
		stock_sd['Strategy'] = stock_sd['Regime'].shift(1) * stock_sd['Market']

		fig, ax = plt.subplots(3,1,figsize=(13,8))
		ax[0].plot(stock_sd["Close"],label="Close")
		ax[0].plot(stock_sd["42d"],label="42d")
		ax[0].plot(stock_sd["252d"],label="252d")
		ax[0].xaxis.set_major_formatter(my_year_month_fmt)
		ax[0].legend(loc="best")

		ax[1].plot(stock_sd["Regime"],linewidth=1.5)
		ax[1].axis(ymin=-1.1,ymax=1.1)
		ax[1].xaxis.set_major_formatter(my_year_month_fmt)

		ax[2].plot(stock_sd['Market'].cumsum().apply(np.exp),label="Market",color="blue")
		ax[2].plot(stock_sd['Strategy'].cumsum().apply(np.exp),label = "Strategy",color="green")
		ax[2].legend(loc="best")
		ax[2].xaxis.set_major_formatter(my_year_month_fmt)

		return fig

	@staticmethod
	def cycle_indicator_plot(ticker:str,start:datetime.date,end:datetime.date):
		panel_data = Data.stock_data(ticker,start,end)
		close = Data.get_close(ticker,start,end)

		sine, leadsine = talib.HT_SINE(panel_data["Adj Close"])
		integer = talib.HT_TRENDMODE(panel_data["Adj Close"])

		fig, (ax_stocktrend,ax_trendmode,ax_ht_sine) = plt.subplots(3,1,figsize = (13,10))
		ax_stocktrend.plot(close.loc[start:end,].index, close.loc[start:end], label=ticker)
		ax_stocktrend.legend(loc="best")
		ax_stocktrend.set_ylabel('Price in $')

		ax_trendmode.plot(integer,color = "#63abdb")
		ax_trendmode.set_ylabel("Hilbert Transform - Trend vs Cycle Mode")

		ax_ht_sine.plot(panel_data.index,sine, color = "#63abdb",label="Sine")
		ax_ht_sine.plot(panel_data.index, leadsine, color="#63abdb", dashes=[2, 2, 2, 2],label="Leadsine")

		ax_ht_sine.legend(loc="best")
		ax_ht_sine.set_ylabel('Hilbert Transform - SineWave')

		return fig