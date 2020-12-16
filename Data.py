#create data class
import pandas as pd
from pandas_datareader import data 
import datetime

class Data:

	@staticmethod
	def stock_data(ticker:str,start:datetime.date,end:datetime.date):
		stock_data = data.DataReader(ticker,'yahoo',start,end)
		return stock_data
		
	@staticmethod	
	def get_adj_close(ticker:str,start:datetime.date,end:datetime.date):
		data = Data.stock_data(ticker,start,end)
		adj_close = data["Adj Close"]

		all_weekdays = pd.date_range(start=start,end=end, freq='B')

		adj_close = adj_close.reindex(all_weekdays)

		adj_close = adj_close.fillna(method='ffill')

		return adj_close

	@staticmethod
	def get_close(ticker:str,start:datetime.date,end:datetime.date):
		data = Data.stock_data(ticker,start,end)
		close = data["Close"]

		all_weekdays = pd.date_range(start=start,end=end, freq='B')

		close = close.reindex(all_weekdays)

		close = close.fillna(method='ffill')

		return close

	@staticmethod
	def get_open(ticker:str,start:datetime.date,end:datetime.date):
		data = Data.stock_data(ticker,start,end)
		open_ = data["Open"]

		all_weekdays = pd.date_range(start=start,end=end, freq='B')

		open_ = open_.reindex(all_weekdays)

		open_ = open_.fillna(method='ffill')

		return open_

	@staticmethod	
	def get_high(ticker:str,start:datetime.date, end:datetime.date):
		data = Data.stock_data(ticker,start,end)
		high = data["High"]

		all_weekdays = pd.date_range(start=start,end=end, freq='B')

		high = high.reindex(all_weekdays)

		high = high.fillna(method='ffill')

		return high

	@staticmethod	
	def get_low(ticker:str,start:datetime.date, end:datetime.date):
		data = Data.stock_data(ticker,start,end)
		low = data["Low"]
		
		all_weekdays = pd.date_range(start=start,end=end, freq='B')

		low = low.reindex(all_weekdays)

		low = low.fillna(method='ffill')

		return low