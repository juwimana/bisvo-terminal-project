"""
This is a class that returns the stock ticker live data from yahoo
finance
"""
####Imports
from pandas_datareader import data 


def Real_Time():
	def __init__(self, ticker:str):
		self.ticker = ticker

	def get_data(self):
		"""Gets Real Time data from yahoo finance"""
		data = data.get_quote_yahoo(ticker)

		return data

	def regular_percent_change(self):
		
