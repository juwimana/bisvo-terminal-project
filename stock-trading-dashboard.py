#import stock data from github
from Data import *
from Plot import *
from Format_Functions import *
import streamlit as st
import datetime
import wiki
import yfinance as yf
import pandas as pd 
import numpy as np
st.set_page_config(page_title="Bisvo Terminal", page_icon="💸")

st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">', unsafe_allow_html=True)



@st.cache(allow_output_mutation=True)
def load_data():
	data_nyse = Data.import_data('nyse_tickers.csv')
	data_nasdaq = Data.import_data('nasdaq_tickers.csv')
	data_canada = Data.import_data('canada_tickers.csv')
	data = pd.concat([data_nyse,data_nasdaq,data_canada],axis=1)
	data = data.rename(columns={'Ticker':'NYSE','Tickers':"NASDAQ", 'Indices':'Canada Stocks'})
	raw_data = data

	return data

#left, center, right = st.beta_columns([2.5,5,2.5])
st.markdown(f"<div class='shadow p-3 mb-5 bg-white rounded' style='text-allign:center'>\
				<h2 style='text-align:center; font-weight:50;'><b>Bisvo Terminal</b></h2></div>",
			 unsafe_allow_html=True)
#center.title("Bisvo Terminal")

data = load_data()
start = st.date_input("Start Date",value=datetime.date(datetime.datetime.today().year - 1,1,1))
end = st.date_input("End Date", value=datetime.date.today())

stock_exchange = st.multiselect("Select Stock Exchange", options=['NYSE','NASDAQ','Canada Stocks'],
								default='NYSE')

tickers = data[stock_exchange].values.T.ravel().tolist()
data = pd.DataFrame({'Ticker':tickers}).dropna()
ticker = st.selectbox("Ticker",sorted(data['Ticker']))
stock = yf.Ticker(ticker)
name = stock.info['longName']

with st.spinner("Loading ...."):
	st.markdown(f"<div class='shadow p-3 mb-5 bg-white rounded'>\
				<h4 style='text-align:center;font-weight:400;'>{name} Analysis</h4></div>",
			 unsafe_allow_html=True)
	#st.header(f"{name} Analysis")

	word_cloud_container = st.beta_expander(f"Wikipedia Word Cloud", expanded=False)
	historical_data_container = st.beta_expander(f"Historical Data Plot", expanded=False)
	rolling_container = st.beta_expander("Rolling Plot", expanded=False)
	momentum_container = st.beta_expander("Momentum Plot", expanded=False)
	bollinger_container = st.beta_expander("Bollinger Bands", expanded=False)
	ema_container = st.beta_expander("Exponential Moving Average (EMA) Plot", expanded=False)
	trend_strategy_container = st.beta_expander("Trend Strategy", expanded=False)
	cycle_indicator_container = st.beta_expander("Cycle Indicator", expanded=False)

	try:
		word_cloud_container.subheader(f"Wikipedia {name} Word Cloud")
		word_cloud_container.pyplot(wiki.create_wordcloud(wiki.get_wiki(name)))
	except:
		word_cloud_container.write("Page Unavailable")

	historical_data_container.subheader(f"{name} Historical Data")
	historical_data_container.pyplot(Plot.historical_line_plot(ticker,start,end))

	rolling_container.subheader(f"Rolling Plot")
	window = rolling_container.number_input("Window", min_value=0,value=42,step=1)
	rolling_container.pyplot(Plot.rolling_plot(ticker,window,start,end))

	momentum_container.subheader(f"{name} Momentum Plot")
	momentum_container.pyplot(Plot.momentum_plot(ticker,start,end))

	bollinger_container.subheader(f"{name} Bollinger Bands")
	bollinger_container.pyplot(Plot.bollinger_bands(ticker,start,end))

	ema_container.subheader(f"{name} EMA Plot")
	span = ema_container.number_input("Span",min_value=0,value=42,step=1)
	ema_container.pyplot(Plot.ema_plot(ticker,span,start,end))

	trend_strategy_container.subheader(f"{name} Trend Strategy")
	threshold = trend_strategy_container.number_input("Threshold",min_value=0,value=42,step=1)
	trend_strategy_container.pyplot(Plot.trend_strategy(ticker,threshold,start,end))

	cycle_indicator_container.subheader(f"{name} Cycle Indicator")
	cycle_indicator_container.pyplot(Plot.cycle_indicator_plot(ticker,start,end))




