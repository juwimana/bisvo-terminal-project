"""
This is an application that users can use to analyze historical financial
data of a stock ticker and make informed decisions before trading a stock ticker.

"""

######Imports
import streamlit as st
import datetime
import wiki
import yfinance as yf
import pandas as pd 
import numpy as np
import time
import math
import requests
import os
from pandas_datareader import data
from scipy.stats import percentileofscore as score
from statistics import mean 

######Import classes I designed as modules
from Data import *
from Plot import *
from Format_Functions import *
from help import *
#####Main application function
def main():

	"""
	This is the main body of the Bisvo Terminal app which is run when the script
	is called from the terminal. It executes all the required computations and 
	contains all the app layout 

	"""

	#####Configure the page
	st.set_page_config(page_title="Bisvo Terminal", page_icon="💸", layout='wide', initial_sidebar_state='expanded')
	st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">', unsafe_allow_html=True)

	##### Load the data
	data = load_data()
	token = 'Tpk_621626e278f34ee9b6a58f1441efca27'
	path = r"C:\Users\jeanm\Downloads\Python\Streamlit\Trading-Stocks-Dashbord\data"
	markdown = r"C:\Users\jeanm\Downloads\Python\Streamlit\Trading-Stocks-Dashbord\markdown"
	#####Bisvo Terminal App Layout
	###Main page title: HTML Formatting Implemented
	##Title columns
	# title_container = st.beta_columns([6,4])
	html_main_page_title = "<div class='shadow p-3 mb-5 bg-white rounded' style='text-allign:center'><h2 style='text-align:left; font-weight:550;'><b>Bisvo Terminal</b></h2></div>"
	st.markdown(html_main_page_title,unsafe_allow_html=True)
	# title_container[1].write(get_date_time())
		
	#####Main Page Content

	####Main page columns
	#main_columns = st.beta_columns([7,3])

	####Sidebar containers
	display_options = st.sidebar.beta_container()
	st.sidebar.write("---")
	plot_options = st.sidebar.beta_container()

	#####Add content to Sidebar
	display_options.title("Navigation")
	help_btn = display_options.button("Help")
	display_options.header("View Dashboard")

	###View container
	view_option = display_options.beta_container()
	##view_option columns
	view_columns = view_option.beta_columns(2)
	##view checkbox
	dashboard = display_options.radio("",("My Portfolio","Real Time","Historical","Watchlist",\
										"Algorithmic Trading Strategies"), index=0)

	####Add Help Content
	if help_btn:
		###Header
		st.markdown("## Help\n *source: [Investopedia](https://www.investopedia.com/); \
					[Cambridge Dictionary](https://dictionary.cambridge.org/)*")

		###define help containers
		term_container = st.beta_expander("Term Definitions", expanded=False)
		plot_container = st.beta_expander("Plots Definition", expanded=False)

		###Add content to containers
		term_container.markdown(Help.term_definition())
		plot_container.markdown(Help.plot_definition())
		
	else:
		####Add content to Bisvo Terminal Portifolio Dahsboard
		if dashboard == "My Portfolio":
			st.markdown("## Bisvo Portfolio")
			
			####Select Portfolio to View
			#st.write("Select Portfolio")
			portfolio = st.radio('Portfolio',('Open Positions','Closed Positions'),index=0)
			# open_position = portfolio_columns[0].button("Open Positions")
			# closed_position = portfolio_columns[1].button("Closed Positions")

			if portfolio == 'Open Positions':
				#Ticker filter columns
				ticker_filter_columns = st.beta_columns(2)
				stock_market = ticker_filter_columns[0].multiselect("Select Market",\
								 							options=['US Stocks','Canada Stocks'],\
															default=['US Stocks','Canada Stocks'])
				
				#get tickers based on selected stock exchange
				tickers = pd.read_csv('data/open-position.csv')['Ticker']
				ticker = ticker_filter_columns[1].selectbox("Ticker",tickers)
				ticker_filter_colms = st.beta_columns(3)
				price = ticker_filter_colms[0].number_input('Stock Price', min_value=0.0000, max_value=100_000.0000)
				shares = ticker_filter_colms[1].number_input('Number of shares',min_value=0,value=0,max_value = 10_000)
				dividend = ticker_filter_colms[2].number_input('Dividend (click sell to submit)',min_value=0.0000,max_value=10_000.00)
				trade_columns = st.beta_columns([2.5,2.5,5])
				buy = trade_columns[0].button('Buy')
				sell = trade_columns[1].button('Sell')

				##Add non-existing ticker to dataframe	
				add_ticker_to_df_clms = st.beta_columns(2)
				add_ticker_to_df_text = add_ticker_to_df_clms[0].text_input("Add Ticker to Portfolio", max_chars= 10)

				add_ticker_to_df_btn_clms = st.beta_columns([2.5,2.5,5])			
				add_ticker_to_df_btn = add_ticker_to_df_btn_clms[0].button("Add Ticker")

				open_df = pd.read_csv('data/open-position.csv')
				portfolio_df = open_df.set_index('Ticker')

				if buy:
					trade = 'Buy'
					portfolio_df = portfolio_dataframe(open_df,ticker,price,shares,token,dividend,trade,path)
				elif sell:
					trade = 'Sell'
					portfolio_df = portfolio_dataframe(open_df,ticker,price,shares,token,dividend,trade,path)

				st.dataframe(portfolio_df)
				position = 'Open'
				#### Stacked Bar Cost vs. Dollar Return
				viz_columns = st.beta_columns(2)
				viz_columns[0].markdown('### Cost vs. Dollar Return')
				viz_columns[0].pyplot(Plot.stacked_bar(portfolio_df,position))
				viz_columns[1].markdown('### Diversification')
				viz_columns[1].pyplot(Plot.pie_chart(portfolio_df))

				if add_ticker_to_df_btn:
					update_portfolio(add_ticker_to_df_text.upper(),path)


				st.markdown("## Analysis of Open Portfolio")
				#filter dislay
				open_df = pd.read_csv('data/open-position.csv')
				filter_display(open_df)

			if portfolio == 'Closed Positions':
				closed_df = pd.read_csv('data/closed-position.csv')
				closed_df.set_index('Ticker', inplace=True)
				st.dataframe(closed_df, height=425)

				position = 'Closed'

				#### Stacked Bar Cost vs. Dollar Return
				viz_columns = st.beta_columns(2)
				viz_columns[0].markdown('### Cost vs. Dollar Return')
				viz_columns[0].pyplot(Plot.stacked_bar(closed_df,position))
				viz_columns[1].markdown('### Diversification')
				viz_columns[1].pyplot(Plot.pie_chart(closed_df))

				st.markdown("## Analysis of Closed Portfolio")
				#filter dislay
				portfolio = pd.read_csv('data/closed-position.csv')
				filter_display(portfolio)

		####Add content to Bisvo Terminal Historical Dashboard
		elif dashboard == "Historical":
			###main_columns[0] page containers
			filter_container = st.beta_expander(f"Data Filter",expanded =True)
		
			###main_columns[1] (coming soon)

			###Main page content
			##Filters
			#Date Filter columns
			date_filter_columns = filter_container.beta_columns(2)
			start = date_filter_columns[0].date_input("Start Date",value=datetime.date(datetime.datetime.today().year - 1,1,1))
			end = date_filter_columns[1].date_input("End Date", value=datetime.date.today())

			#Ticker filter columns
			ticker_filter_columns = filter_container.beta_columns(2)
			stock_market = ticker_filter_columns[0].multiselect("Select Market",\
							 							options=['US Stocks','Canada Stocks'],\
														default=['US Stocks','Canada Stocks'])
			
			#get tickers based on selected stock exchange
			tickers = get_tickers(data,stock_market)
			ticker = ticker_filter_columns[1].selectbox("Ticker",tickers)

			#get name of selected ticker
			name = stock_real_time_data(ticker)["longName"].values[0]


			##Button columns
			btn_columns = filter_container.beta_columns([5,2.5,2.5])
			#Add Watchlist Button
			add_to_watchlist_button = btn_columns[1].button("Add to Watchlist")
			add_to_portfolio_button = btn_columns[2].button("Add to My Portfolio")

			##Add To Watchlist event
			if add_to_watchlist_button:
				update_watchlist(ticker,path)

			##Add to my portfolio
			if add_to_portfolio_button:
				update_portfolio(ticker,path)

			##Add non-existing ticker to dataframe	
			add_ticker_to_df_clms = filter_container.beta_columns(2)
			add_ticker_to_df_text = add_ticker_to_df_clms[0].text_input("Can't Find Your Ticker?\
										 Add Ticker to DataFrame!", max_chars= 10)
			result_ = add_ticker_to_df_clms[1].empty()

			add_ticker_to_df_btn_clms = filter_container.beta_columns([2.5,2.5,5])			
			add_ticker_to_df_btn = add_ticker_to_df_btn_clms[0].button("Add Ticker")
			

			
			if add_ticker_to_df_btn:
				result = update_ticker_df(add_ticker_to_df_text.upper(),path)
				if result == "s":
					result_.success(f'{add_ticker_to_df_text.upper()} successfully added to DataFrame!')
				elif result == "f":
					result_.error(f"Cannot add {add_ticker_to_df_text.upper()} to DataFrame")
			
			#####Display the filtered data
			###Display Title
			st.markdown(f"<div class='shadow p-3 mb-5 bg-white rounded'><h4 style='text-align:center;font-weight:600;'>{name} Analysis</h4></div>",	unsafe_allow_html=True)
			
			###main_columns[0] page containers
			#hqm_container = st.beta_expander(f"Quantitative Momentum Strategy", expanded=False)
			#rv_container = st.beta_expander(f"Quantitative Value Strategy", expanded=False)
			word_cloud_container = st.beta_expander(f"Wikipedia Word Cloud", expanded=False)
			historical_data_container = st.beta_expander(f"Historical Data Plot", expanded=False)
			momentum_container = st.beta_expander("Momentum Plot", expanded=False)
			bollinger_container = st.beta_expander("Bollinger Bands", expanded=False)
			macd_container = st.beta_expander("Moving Average Convergence Divergence (MACD)", expanded=False)
			trend_strategy_container = st.beta_expander("Trend Strategy", expanded=False)
			cycle_indicator_container = st.beta_expander("Cycle Indicator", expanded=False)

			###Data loading announcement
			with st.spinner("Loading ...."):
				# portfolio_size = 10_000
				# tickers_df = pd.DataFrame({'Ticker':tickers})
				# feature,top,sort = display_momentum_strategy(tickers_df)
				# hqm_dataframe = high_quality_momentum(tickers_df,token,portfolio_size, feature,top,sort)
				# hqm_container.dataframe(hqm_dataframe,height=500)

				# feature,top,sort = display_value_strategy(tickers_df)
				# rv_dataframe = robust_quantitative_value(tickers_df,token,portfolio_size,feature,top,sort)
				# rv_container.dataframe(rv_dataframe,height=500)
			
				try:
					word_cloud_container.pyplot(wiki.create_wordcloud(wiki.get_wiki(name)))
				except:
					word_cloud_container.write("Page Unavailable")

				historical_data_container.pyplot(Plot.historical_line_plot(ticker,start,end))

				momentum_container.pyplot(Plot.momentum_plot(ticker,start,end))

				bollinger_container.pyplot(Plot.bollinger_bands(ticker,start,end))

				macd_container.pyplot(Plot.macd_plot(ticker,start,end))

				threshold = trend_strategy_container.number_input("Threshold",min_value=0,value=42,step=1)
				trend_strategy_container.pyplot(Plot.trend_strategy(ticker,threshold,start,end))

				cycle_indicator_container.pyplot(Plot.cycle_indicator_plot(ticker,start,end))
		####Add content to Bisvo Terminal Real Time Dashboard
		elif dashboard == "Real Time":
			st.markdown("#### WE ARE LIVE")
		####Add content to Bisvo Terminal Watchlist Dashboard
		elif dashboard == "Watchlist":
			PORTFOLIO_SIZE = 10_000
			watchlist = pd.read_csv("data/watchlist.csv")

			feature,top,sort = display_momentum_strategy(watchlist,markdown)
			momentum = high_quality_momentum(watchlist,token,PORTFOLIO_SIZE, feature,top,sort)
			st.dataframe(momentum)

			feature,top,sort = display_value_strategy(watchlist,markdown)
			value = robust_quantitative_value(watchlist,token,PORTFOLIO_SIZE,feature,top,sort)
			st.dataframe(value)

			#filter dislay
			filter_display(watchlist)

		elif dashboard == "Algorithmic Trading Strategies":
			st.markdown("# Algorithmic Trading Strategies")

			###Filter columns
			alg_filter_columns = st.beta_columns(2)
			collection = ['All','S&P 500']
			ticker_collection = alg_filter_columns[0].selectbox("Ticker Collection",collection)
			portfolio_size = alg_filter_columns[1].number_input("Enter the value of your portfolio:",\
	    				min_value= 0.0)
			
			stocks = pd.read_csv("data/sp_500_stocks - Copy.csv")
			feature,top,sort = display_momentum_strategy(stocks,markdown)
			hqm_dataframe = high_quality_momentum(stocks,token,portfolio_size, feature,top,sort)
			st.dataframe(hqm_dataframe,height=500)

			feature,top,sort = display_value_strategy(stocks,markdown)
			rv_dataframe = robust_quantitative_value(stocks,token,portfolio_size,feature,top,sort)
			st.dataframe(rv_dataframe,height=500)

### Cached functions
@st.cache(allow_output_mutation=True)
def load_data():
	data_us = pd.read_csv('data/us_tickers.csv')
	data_canada = pd.read_csv('data/canada_tickers.csv')
	data = pd.concat([data_us,data_canada],axis=1)
	data = data.rename(columns={'Tickers':'US Stocks', 'Indices':'Canada Stocks'})
	raw_data = data #keeping a copy of data as raw data
	return data	

@st.cache(allow_output_mutation=True)
def get_tickers(data,stock_market:str):
	"""For a given stock_market(s) the function returns the sorted listed tickers"""
	temp_data= data[stock_market].values.T.ravel().tolist()
	data_temp = pd.DataFrame({'Ticker':temp_data}).dropna()
	sorted_data = sorted(data_temp['Ticker'])
	return sorted_data

@st.cache(allow_output_mutation=True)
def stock_real_time_data(ticker:str):
	"""Gets Real Time Data of stock ticker from Yahoo Finance """
	stock_data = data.get_quote_yahoo(ticker)
	return stock_data

@st.cache(allow_output_mutation=True)
def load_watchlist(token):
	"""Gets watchlist DataFrame"""
	portfolio_size = 10_000
	watchlist = pd.read_csv("data/watchlist.csv")
	momentum = high_quality_momentum(watchlist,token,portfolio_size)
	value = robust_quantitative_value(watchlist,token,portfolio_size)
	return (momentum, value)

@st.cache(allow_output_mutation=True)
def high_quality_momentum(stocks,token,portfolio_size:float,feature,top,sort):
	"""Returns dataframe with the top 50 stocks ranked by HQM Score"""
	high_quality_momentum_columns = [
    'Ticker',
    'Price',
    'Number of Shares to Buy',
    'One-Year Price Return',
    'One-Year Return Percentile',
    'Six-Month Price Return',
    'Six-Month Return Percentile',
    'Three-Month Price Return',
    'Three-Month Return Percentile',
    'One-Month Price Return',
    'One-Month Return Percentile',
    'HQM Score']

	hqm_dataframe = pd.DataFrame(columns = high_quality_momentum_columns)


	#Function sourced from 
	# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
	def chunks(lst, n):
	    """Yield successive n-sized chunks from lst."""
	    for i in range(0, len(lst), n):
	        yield lst[i:i + n]   
	        
	symbol_groups = list(chunks(stocks['Ticker'], 100))
	symbol_strings = []
	for i in range(0, len(symbol_groups)):
	    symbol_strings.append(','.join(symbol_groups[i]))

	for symbol_string in symbol_strings:
	    batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch?symbols={symbol_string}&types=price,stats&token={token}'
	    data = requests.get(batch_api_call_url).json()
	    for symbol in symbol_string.split(','):
	    	try:
	    		try:
	    			price = data[symbol]['price']
	    		except TypeError:
	    			price = np.NaN
	    		try:
	    			year_1 = data[symbol]['stats']['year1ChangePercent']
	    		except TypeError:
	    			year_1 = np.NaN
	    		try:
	    			month_6 = data[symbol]['stats']['month6ChangePercent']
	    		except TypeError:
	    			month_6 = np.NaN
	    		try:
	    			month_3 = data[symbol]['stats']['month3ChangePercent']
	    		except TypeError:
	    			month_3 = np.NaN
	    		try:
	    			month_1 = data[symbol]['stats']['month1ChangePercent']
	    		except TypeError:
	    			month_1 = np.NaN
	            
	    		hqm_dataframe = hqm_dataframe.append(
		            pd.Series(
		            [
		                symbol,
		                price,
		                'N/A',
		                year_1,
		                'N/A',
		                month_6,
		                'N/A',
		                month_3,
		                'N/A',
		                month_1,
		                'N/A',
		                'N/A'
		            ],
		            index = high_quality_momentum_columns),
		            ignore_index=True
		            )
	    	except KeyError:
	    		pass

	time_periods = [
				    'One-Year',
				    'Six-Month',
				    'Three-Month',
				    'One-Month']

	for row in hqm_dataframe.index:
	    for time_period in time_periods:
	        change_col = f'{time_period} Price Return'
	        percentile_col = f'{time_period} Return Percentile'
	        hqm_dataframe.loc[row,percentile_col] =\
	        score(hqm_dataframe[change_col], hqm_dataframe.loc[row,change_col])

	for row in hqm_dataframe.index:
	    momemtum_percentiles = []
	    for time_period in time_periods:
	        momemtum_percentiles.append(hqm_dataframe.loc[row,f'{time_period} Return Percentile'])
	    hqm_dataframe.loc[row, 'HQM Score'] = mean(momemtum_percentiles)
	
	if sort == 'Descending':
		hqm_dataframe.sort_values(feature, ascending=False, inplace=True)
	else:
		hqm_dataframe.sort_values(feature, ascending=True, inplace=True)
	hqm_dataframe = hqm_dataframe[:top].set_index('Ticker')

	val = float(portfolio_size)
	position_size = val/len(hqm_dataframe.index)
	for ticker in hqm_dataframe.index:
	    hqm_dataframe.loc[ticker,"Number of Shares to Buy"] = \
	    math.floor(position_size/hqm_dataframe.loc[ticker,"Price"])

	### Styling DataFrame
	hqm_dataframe.style.format({'One-Year Price Return':'{:.4%}'})

	return hqm_dataframe

@st.cache(allow_output_mutation=True)
def robust_quantitative_value(stocks,token,portfolio_size:float,feature,top,sort):
	"""Returns dataframe with the top 50 stocks ranked by HQM Score"""

	# Function sourced from 
	# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
	def chunks(lst, n):
	    """Yield successive n-sized chunks from lst."""
	    for i in range(0, len(lst), n):
	        yield lst[i:i + n]   
	        
	symbol_groups = list(chunks(stocks['Ticker'], 100))
	symbol_strings = []
	for i in range(0, len(symbol_groups)):
	    symbol_strings.append(','.join(symbol_groups[i]))
	#     print(symbol_strings[i])

	rv_columns = ['Ticker', 
	              'Price', 
	              'Number of Shares to Buy', 
	              'Price-to-Earnings Ratio',
	              'PE Percentile',
	              'Price-to-Book Ratio',
	              'PB Percentile',
	              'Price-to-Sales Ratio',
	              'PS Percentile',
	              'EV/EBITDA',
	              'EV/EBITDA Percentile',
	              'EV/GP',
	              'EV/GP Percentile',
	              'RV Score']

	rv_dataframe = pd.DataFrame(columns=rv_columns)

	for symbol_string in symbol_strings:
	    batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch?symbols={symbol_string}&types=quote,advanced-stats&token={token}'
	    data = requests.get(batch_api_call_url).json()
	    for symbol in symbol_string.split(','):
	        try:
	            try:
	                latestPrice = data[symbol]['quote']['latestPrice']
	            except TypeError:
	                latestPrice = np.NaN
	            try:
	                peRatio = data[symbol]['quote']['peRatio']
	            except TypeError:
	                peRatio = np.NaN
	            try:
	                price_to_book = data[symbol]['advanced-stats']['priceToBook']
	            except TypeError:
	                price_to_book = np.NaN
	            try:
	                price_to_sales = data[symbol]['advanced-stats']['priceToSales']
	            except TypeError:
	                price_to_sales = np.NaN
	            try:
	                enterprise_value = data[symbol]['advanced-stats']['enterpriseValue']
	            except TypeError:
	                enterprise_value = np.NaN
	            try:
	                ebitda = data[symbol]['advanced-stats']['EBITDA']
	            except TypeError:
	                ebitda = np.NaN
	            try:
	                gross_profit = data[symbol]['advanced-stats']['grossProfit']
	            except TypeError:
	                gross_profit = np.NaN
	            try:
	                ev_to_ebitda = enterprise_value/ebitda
	            except TypeError:
	                ev_to_ebitda = np.NaN

	            try:
	                ev_to_gross_profit = enterprise_value/gross_profit
	            except TypeError:
	                ev_to_gross_profit = np.NaN
	            rv_dataframe = rv_dataframe.append(
	                pd.Series(
	                [
	                    symbol,
	                    latestPrice,
	                    'N/A',
	                    peRatio,
	                    'N/A',
	                    price_to_book,
	                    'N/A',
	                    price_to_sales,
	                    'N/A',
	                    ev_to_ebitda,
	                    'N/A',
	                    ev_to_gross_profit,
	                    'N/A',
	                    'N/A'
	                ],
	                index = rv_columns),
	                ignore_index = True
	                )
	        except KeyError:
	            pass

	for column in ['Price-to-Earnings Ratio','Price-to-Book Ratio','Price-to-Sales Ratio','EV/EBITDA','EV/GP']:
		rv_dataframe[column].fillna(rv_dataframe[column].mean(), inplace=True)
	
	# if ticker_collection == 'All':
	# 	rv_dataframe.dropna(axis='index', inplace =True)
	# else:
	# 	pass

	metrics = {
    'Price-to-Earnings Ratio':'PE Percentile',
    'Price-to-Book Ratio':'PB Percentile',
    'Price-to-Sales Ratio':'PS Percentile',
    'EV/EBITDA':'EV/EBITDA Percentile',
    'EV/GP':'EV/GP Percentile'}

	for metric in metrics.keys():
	    for row in rv_dataframe.index:
	        rv_dataframe.loc[row,metrics[metric]] = score(rv_dataframe[metric],rv_dataframe.loc[row,metric])


	for row in rv_dataframe.index:
	    value_percentiles = []
	    for metric in metrics.keys():
	        value_percentiles.append(rv_dataframe.loc[row,metrics[metric]])
	    rv_dataframe.loc[row, 'RV Score'] = mean(value_percentiles)
	
	if sort == "Ascending":
		rv_dataframe.sort_values(feature, ascending=True,inplace =True)
	else:
		rv_dataframe.sort_values(feature, ascending=False,inplace =True)

	rv_dataframe = rv_dataframe[:top].set_index('Ticker')

	val = float(portfolio_size)
	position_size = val/len(rv_dataframe.index)
	for ticker in rv_dataframe.index:
	    rv_dataframe.loc[ticker,"Number of Shares to Buy"] = \
	    math.floor(position_size/rv_dataframe.loc[ticker,"Price"])

	return rv_dataframe

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def portfolio_dataframe(stocks, ticker, price, shares,token,dividend,trade,path):
	"""Returns portfolio dataframe with ticker metrics 
			- Average Price Per Share
			- Total Cost
			- Current Value
			- Dollar Return
			- Percent Return
			- Number of shares
			-  
	"""

	# Function sourced from 
	# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
	def chunks(lst, n):
	    """Yield successive n-sized chunks from lst."""
	    for i in range(0, len(lst), n):
	        yield lst[i:i + n]   
	        
	symbol_groups = list(chunks(stocks['Ticker'], 100))
	symbol_strings = []
	for i in range(0, len(symbol_groups)):
	    symbol_strings.append(','.join(symbol_groups[i]))
	    stocks = stocks.set_index('Ticker')
	
	for symbol_string in symbol_strings:
		batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch?symbols={symbol_string}&types=quote,advanced-stats&token={token}'
		data = requests.get(batch_api_call_url).json()
		for symbol in stocks.index:
			stock = yf.Ticker(symbol)
			try:
				latestPrice = data[symbol]['quote']['latestPrice']
			except KeyError:
				try:
					latestPrice = stock.info['regularMarketPrice']
				except KeyError:
					latestPrice = np.NaN
			try:
				sector = stock.info['sector']
			except KeyError:
				sector = np.NaN

			stocks.loc[symbol,'Sector'] = sector
			stocks.loc[symbol,'Current Price'] = latestPrice

		if price > 0 and shares > 0 and trade == 'Buy':
			avg_price = stocks.loc[ticker,"Average Purchase Price Per Share"]
			number_of_shares = stocks.loc[ticker,"Total Number of Shares"]
			stocks.loc[ticker,"Average Purchase Price Per Share"] = ((avg_price * number_of_shares) + 
		                                          					(price * shares))/ (number_of_shares + shares)
			stocks.loc[ticker,"Total Number of Shares"] += shares
			stocks.loc[ticker,"Total Cost"] += price * shares
			stocks.loc[ticker,"Current Value"] = stocks.loc[ticker,"Total Number of Shares"] * stocks.loc[ticker,"Current Price"]
			stocks.loc[ticker,"Dollar Return"] = stocks.loc[ticker,"Current Value"] - stocks.loc[ticker,"Total Cost"] + stocks.loc[ticker,'Dividend']
			if stocks.loc[ticker,"Total Cost"] > 0:
				stocks.loc[ticker,"Percent Return"] = (stocks.loc[ticker,"Dollar Return"] + stocks.loc[ticker,"Dividend"]) / stocks.loc[ticker,"Total Cost"]
			else: 
				stocks.loc[ticker,"Total Cost"] = 0

		elif price > 0 and shares > 0 and trade == 'Sell':
			avg_price = stocks.loc[ticker,"Average Purchase Price Per Share"]
			number_of_shares = stocks.loc[ticker,"Total Number of Shares"]
			outstanding_shares_value = (avg_price * number_of_shares) - (avg_price * shares)
			gain_loss = (price *shares) - (avg_price * shares)
			outstanding_shares =  number_of_shares - shares
			cost_ = stocks.loc[ticker,"Total Cost"]
			if outstanding_shares == 0:
				stocks.loc[ticker,"Average Purchase Price Per Share"] = 0
			else:
				stocks.loc[ticker,"Average Purchase Price Per Share"] = outstanding_shares_value / outstanding_shares
			stocks.loc[ticker,"Total Number of Shares"] = outstanding_shares
			stocks.loc[ticker,"Total Cost"] = stocks.loc[ticker,"Average Purchase Price Per Share"] * stocks.loc[ticker,"Total Number of Shares"] 
			
			if stocks.loc[ticker,"Total Cost"] > 0:
				stocks.loc[ticker,"Current Value"] = stocks.loc[ticker,"Total Number of Shares"] * stocks.loc[ticker,"Current Price"]
				stocks.loc[ticker,"Dollar Return"] = stocks.loc[ticker,"Current Value"] - stocks.loc[ticker,"Total Cost"]
				stocks.loc[ticker,"Percent Return"] = (stocks.loc[ticker,"Dollar Return"] + stocks.loc[ticker,"Dividend"]) / stocks.loc[ticker,"Total Cost"]
			else:
				stocks.loc[ticker,"Current Value"] = 0
				stocks.loc[ticker,"Dollar Return"] = gain_loss
				stocks.loc[ticker,"Percent Return"] = (gain_loss + stocks.loc[ticker,'Dividend'])/cost_
	
	stocks.loc[ticker,'Dividend'] += dividend 
	stocks.sort_values(by=['Percent Return','Dollar Return','Current Value'], ascending=False, inplace = True)
	
	open_position = stocks[stocks['Current Value'] != 0]
	open_position.to_csv('data/open-position.csv')
	
	if (trade == 'Buy' or trade == 'Sell') and len(stocks[stocks['Current Value']==0].index) > 0:
		closed_df = pd.read_csv('data/closed-position.csv')

		stocks__ = stocks.reset_index()
		closed_position = closed_df.append(stocks__[stocks__['Current Value'] == 0][['Ticker','Current Price','Dollar Return',
																 'Percent Return', 'Sector']], ignore_index=True)
		closed_position.set_index('Ticker', inplace=True)
		closed_position.to_csv('data/closed-position.csv')

	return open_position

###Other functions
def get_date_time():
	"""Gets the current time of a specific time zone"""
	seconds = time.time()
	localtime = time.ctime(seconds)
	return localtime


def update_watchlist(ticker:str,path):
	"""Gets the watchlist dataframe and adds a new ticker to the watchlist"""
	watchlist = pd.read_csv('data/watchlist.csv')
	if ticker in watchlist.Ticker:
		pass
	else:
		ticker_df = pd.DataFrame({"Ticker":[ticker]})
		watchlist = watchlist.append(ticker_df, ignore_index=True)
		watchlist = watchlist.drop_duplicates()
		watchlist.to_csv('data/watchlist.csv',index=False)

def update_portfolio(ticker:str,path):
	"""Gets the portfolio dataframe and adds a new ticker to the portfolio"""
	portfolio_df = pd.read_csv('data/open-position.csv')
	columns = [
	    'Ticker',
	    'Current Price',
	    'Average Purchase Price Per Share',
	    'Total Number of Shares',
	    'Total Cost',
	    'Current Value',
	    'Dividend',
	    'Dollar Return',
	    'Percent Return',
	    'Sector',
	]
	if ticker in portfolio_df['Ticker']:
		pass
	else:
		portfolio_df = portfolio_df.append(
                pd.Series(
                [
                    ticker,
                    0.0,
                    0.0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    'N/A'
                ],
                index = columns),
                ignore_index = True
                )
	portfolio_df.to_csv('data/open-position.csv',index=False)

def update_ticker_df(ticker:str,path):
	"""Updates the ticker dataframe by adding ticker to the correct exchange dataframe"""
	#find the ticker exchange
	stock_info = data.get_quote_yahoo(ticker)
	stock_market = stock_info.market.values[0]

	if stock_market == "ca_market":
		ca_stocks = pd.read_csv('data/canada_tickers.csv')
		ticker_df = pd.DataFrame({"Indices":[ticker]})
		canada_tickers = ca_stocks.append(ticker_df, ignore_index=True)
		canada_tickers = canada_tickers.drop_duplicates()
		canada_tickers.to_csv('data/canada_tickers.csv',index=False)
		return "s"
	elif stock_market == "us_market":
		us_stocks = pd.read_csv('data/us_tickers.csv')
		ticker_df = pd.DataFrame({"Tickers":[ticker]})
		us_tickers = us_stocks.append(ticker_df, ignore_index=True)
		us_tickers = us_tickers.drop_duplicates()
		us_tickers.to_csv('data/us_tickers.csv',index=False)
		return "s"
	else:
		return "f"

def display_value_strategy(df,markdown):
	"""Returns value strategy content and display format"""

	st.markdown('## Quantitative Value Strategy')
	value_container = st.beta_expander('Definition',expanded=False)
	string_value = open(os.path.join(markdown,r'quantitative_value_strategy.txt'), 'r').read()
	value_container.markdown(string_value)
	st.subheader("Filter")
	value_filter_columns = st.beta_columns(3)
	feature = value_filter_columns[0].selectbox("Sort by", ['Price','Price-to-Earnings Ratio','PE Percentile','Price-to-Book Ratio',
	       										'PB Percentile','Price-to-Sales Ratio','PS Percentile','EV/EBITDA','EV/EBITDA Percentile',
	       										'EV/GP','EV/GP Percentile','RV Score'], index=11, key='value')
	top = value_filter_columns[1].number_input("Top",min_value=5,max_value =len(df.index), step=1, key='value')
	sort = value_filter_columns[2].radio("", ("Ascending", "Descending"),index=0, key='value')  

	return (feature,top,sort)

def display_momentum_strategy(df,markdown):
	"""Returns momentum strategy content and display format"""

	st.markdown('## Quantitative Momentum Strategy')
	momentum_container = st.beta_expander('Definition',expanded=False)
	string_momentum = open(os.path.join(markdown,r'quantitative_momentum_strategy.txt'), 'r').read()
	momentum_container.markdown(string_momentum)
	st.subheader("Filter")
	momentum_filter_columns = st.beta_columns(3)
	feature = momentum_filter_columns[0].selectbox("Sort by", ['Price','One-Year Price Return','One-Year Return Percentile','Six-Month Price Return',
    													  'Six-Month Return Percentile','Three-Month Price Return','Three-Month Return Percentile',
    													   'One-Month Price Return','One-Month Return Percentile','HQM Score'], index=9, key='momentum')
	top = momentum_filter_columns[1].number_input("Top",min_value=5,max_value=len(df.index),step=1, key='momentum')
	sort =momentum_filter_columns[2].radio("", ("Ascending", "Descending"),index=0, key='momentum')

	return (feature,top,sort)

def filter_display(df):
	#Filters
	#Date Filter columns
	date_filter_columns = st.beta_columns(2)
	start = date_filter_columns[0].date_input("Start Date",value=datetime.date(datetime.datetime.today().year - 1,1,1))
	end = date_filter_columns[1].date_input("End Date", value=datetime.date.today())

	#Ticker filter columns
	ticker_filter_columns = st.beta_columns(2)
			
	#get tickers
	tickers = df['Ticker']
	ticker = ticker_filter_columns[0].selectbox("Ticker",tickers, key='ticker')

	#get name of selected ticker
	name = stock_real_time_data(ticker)["longName"].values[0]

	###Display Title
	st.markdown(f"<div class='shadow p-3 mb-5 bg-white rounded'><h4 style='text-align:center;font-weight:600;'>{name} Analysis</h4></div>",	unsafe_allow_html=True)

	#Visualization
	historical_data_container = st.beta_expander(f"Historical Data Plot", expanded=False)
	macd_container = st.beta_expander("Moving Average Convergence Divergence (MACD)", expanded=False)
	cycle_indicator_container = st.beta_expander("Cycle Indicator", expanded=False)

	historical_data_container.pyplot(Plot.historical_line_plot(ticker,start,end))
	macd_container.pyplot(Plot.macd_plot(ticker,start,end))
	cycle_indicator_container.pyplot(Plot.cycle_indicator_plot(ticker,start,end))


if __name__ == "__main__":
    main()

