import streamlit as st 
import pandas as pd 
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests

# Configuration de la page
st.set_page_config(
    page_title="Sentiment Analysis of Tweets 🐦",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Titre et description
st.title("Sentiment Analysis of Tweets 🐦 about US Airlines")
st.sidebar.title("Sentiment Analysis of Tweets about US Airlines")
st.sidebar.markdown("This application is a Streamlit dashboard to analyze the sentiment of Tweets 🐦")

# Charger les données
@st.cache_data
def load_data():
    data = pd.read_csv("Tweets.csv")
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

data = load_data()

# Animations Lottie
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_twitter = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_2tlylbqv.json")
lottie_airplane = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_4jw5z2c6.json")

st_lottie(lottie_twitter, height=200, key="twitter")
st_lottie(lottie_airplane, height=200, key="airplane")

# Visualisations
st.sidebar.subheader("Show Random Tweet")
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
st.sidebar.markdown(data.query('airline_sentiment == @random_tweet')[['text']].sample(n=1).iat[0, 0])

st.sidebar.markdown("### Number of tweets by sentiment")
select = st.sidebar.selectbox('Visualization type', ['Histogram', 'Pie Chart'], key='1')
sentiment_count = data["airline_sentiment"].value_counts()
sentiment_count = pd.DataFrame({'Sentiment': sentiment_count.index, 'Tweets': sentiment_count.values})

if not st.sidebar.checkbox("Hide", True):
    st.markdown('### Number of tweets by Sentiment')
    if select == 'Histogram':
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500, template='plotly_dark')
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment', color='Tweets', template='plotly_dark')
        st.plotly_chart(fig)    

# Word Cloud avec Plotly
def plotly_wordcloud(words_freq):
    words = list(words_freq.keys())
    frequencies = list(words_freq.values())
    
    fig = go.Figure(go.Scatter(
        x=[random.random() for _ in words],
        y=[random.random() for _ in words],
        mode='text',
        text=words,
        textfont=dict(size=[freq * 10 for freq in frequencies], color=[f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})" for _ in words]),
    ))
    
    fig.update_layout(
        title="Word Cloud",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    
    return fig

st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment ?', ('positive', 'neutral', 'negative'))
if not st.sidebar.checkbox('Close', True, key='3'):
    st.header("Word Cloud for %s sentiment" % (word_sentiment))
    df = data[data['airline_sentiment'] == word_sentiment]
    words = ' '.join(df['text'])
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    words_freq = {word: processed_words.count(word) for word in processed_words.split()}
    st.plotly_chart(plotly_wordcloud(words_freq))

# Section "À propos"
st.sidebar.markdown("### À propos")
st.sidebar.markdown("Cette application analyse les sentiments des tweets concernant les compagnies aériennes américaines.")
st.sidebar.markdown("Développé avec  par Ibrahima Gabar Diop")
