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
		fig, ax = plt.subplots(2,1,figsize=(16,9))
		close = Data.get_close(ticker,start,end)
		panel_data = Data.stock_data(ticker,start,end)
		
		##Close price plot
		ax[0].plot(close.index,close, label='Close')
		ax[0].set_xlabel('Date')
		ax[0].set_ylabel('Adjusted closing price ($)')
		ax[0].axhline(close.mean(),linestyle='--',color='red',label='Mean Price')
		ax[0].legend()
		
		#Volume traded
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
		plt.ion()
		return fig

	@staticmethod
	def macd_plot(ticker:str,start:datetime.date,end:datetime.date):
		"""
		Plots the Moving Average Convergence/Divergence (MACD) crossover
		to determine when to buy and sell stock
		"""
		close = Data.get_close(ticker,start,end)

		####Calculate the short and long exponential moving average
		shortEMA = close.ewm(span=12,adjust=False).mean()
		longEMA = close.ewm(span=26,adjust=False).mean()

		#calculate MACD line
		MACD = shortEMA - longEMA

		#calculate the signal line
		signal = MACD.ewm(span=9,adjust=False).mean()

		# panel_data = Data.stock_data(ticker,start,end)
		# close = Data.get_close(ticker,start,end)
		ema_short = close.ewm(span=12, adjust=False).mean()

		# Taking the difference between the prices and the EMA timeseries
		trading_positions_raw = close - ema_short
		trading_positions_raw.tail()

		# Taking the sign of the difference to determine whether the price or the EMA is greater and then multiplying by 1/3
		trading_positions = trading_positions_raw.apply(np.sign) * 1
		trading_positions.tail()

		# Lagging our trading signals by one day.
		trading_positions_final = trading_positions.shift(1)

		#plot chart
		fig, ax = plt.subplots(2,1,figsize=(13,8))

		ax[0].plot(close.index, MACD,label=f'MACD')
		ax[0].plot(close.index, signal,label =f'Signal Line')
		ax[0].legend(loc='best')
		ax[0].xaxis.set_major_formatter(my_year_month_fmt)

		ax[1].plot(trading_positions_final.loc[start:end,].index, trading_positions_final.loc[start:end],label='Trading position')
		ax[1].set_ylabel('Trading position')
		ax[1].xaxis.set_major_formatter(my_year_month_fmt)
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


	@staticmethod
	def pie_chart(df):

		#plt.style.use('seaborn-poster')
		###Visualization of Portfolio
		#df = df.index
		slices = df.fillna('Unknown').groupby(by=["Sector"])['Current Price'].count()
		labels = sorted(list(df['Sector'].fillna('Unknown').astype('str').unique()))

		explode = [0,0,0,0.1,0,0,0]
		# fig, ax = plt.subplots()
		fig = plt.figure()
		ax = fig.add_subplot(121)
		ax.pie(slices, labels=labels,autopct='%1.1f%%',textprops={'fontweight':'light','fontsize': 6,'fontfamily':'Times New Roman'},
			labeldistance=1.1,wedgeprops={'linewidth':.25,'edgecolor':'k'})
		return fig
		

	@staticmethod
	def stacked_bar(df,position):

		if position == 'Open':

			fig, ax = plt.subplots(figsize = (13,10.25))
			df = df.sort_values('Dollar Return', ascending=False)
			labels = df.index
			total_gain_loss = df['Dollar Return']
			cost = df['Total Cost']
			colors = ['#34a853','#fbbc04','#ea4335']

			ax.bar(labels,total_gain_loss,bottom=cost,color=colors[0],align='center',
					label='Total Gain/Loss')
			ax.bar(labels,cost,align='center',color=colors[2], label='Cost')
			ax.set_xlabel('Ticker')
			ax.set_ylabel('Currency USD ($)')
			ax.set_xticklabels(labels, rotation=45, ha='right')
			ax.legend(loc='best')
			return fig

		elif position == 'Closed':

			fig, ax = plt.subplots(figsize = (13,9.5))
			df = df.sort_values('Dollar Return', ascending=False)
			labels = df.index
			total_gain_loss = df['Dollar Return']
			colors = ['#34a853','#fbbc04','#ea4335']

			ax.bar(labels,total_gain_loss,color=colors[0],align='center',
					label='Total Gain/Loss')
			ax.set_xlabel('Ticker')
			ax.set_ylabel('Currency USD ($)')
			ax.set_xticklabels(labels, rotation=45, ha='right')
			ax.legend(loc='best')
			return fig


