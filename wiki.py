#####Imports
import wikipedia
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


def get_wiki(query):
	title = wikipedia.search(query)[0]
	page = wikipedia.page(title)

	return page.content

def create_wordcloud(text):

	stopwords = set(STOPWORDS)
	wc = WordCloud(background_color = "white", width = 600, height = 400, \
				   max_words =200, stopwords=stopwords).generate(text)

	# plot the WordCloud image
	fig, ax = plt.subplots(facecolor=None)

	ax.imshow(wc, interpolation="bilinear")
	ax.axis("off")

	return fig

