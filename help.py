"""
This is a help class the definies and explains terms, instructions and 
interpretation of app content
"""

#####Imports
import streamlit as st 
import os

class Help:

	def term_definition():
		markdown = r"C:\Users\jeanm\Downloads\Python\Streamlit\Trading-Stocks-Dashbord\markdown"
		text = open(os.path.join(markdown,r"term_definition.txt"),'r').read()
		return text

	def plot_definition():
		markdown = r"C:\Users\jeanm\Downloads\Python\Streamlit\Trading-Stocks-Dashbord\markdown"
		text = open(os.path.join(markdown,r"plot_definition.txt"),'r').read()
		return text

