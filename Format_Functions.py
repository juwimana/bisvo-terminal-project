#create Format_Functions class with all format functions
import datetime
import numpy as np
class Format_Functions:

	@staticmethod
	def date_format(string:str):
		try:
			return datetime.datetime.strptime(string,"%Y").year
		except:
			pass

	def float_format(string:str):
		try:
			return round(float(string),4)
		except:
			pass

	def format_ticker(data,ticker:str):
		try:
			exchange = data[data["Ticker"]==ticker]["Exchange Abbv"].unique()
			if exchange == "XLON":
				return ".".join([ticker,"L"])
			elif exchange == "XTSE":
				return  ".".join([ticker,"TO"])
			elif exchange == "XTSX":
				return ".".join([ticker,"V"])
			elif exchange == "XCNQ":
				return ".".join([ticker,"CN"])
			elif exchange == "XNYS":
				return ticker
			elif exchange == "XNAS":
				return ticker
		except:
			return np.nan