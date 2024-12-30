import streamlit as st 
import pandas as pd 
import numpy as np
import plotly.express as px
import random
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests

st.set_page_config(
    page_title="Sentiment Analysis of Tweets 🐦",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Sentiment Analysis of Tweets 🐦 about US Airlines")
st.sidebar.title("Sentiment Analysis of Tweets about US Airlines")
st.sidebar.markdown("This application is a Streamlit dashboard to analyze the sentiment of Tweets 🐦")

data_url = "Tweets.csv"

# Dictionnaire pour les coordonnées de localisation
location_coords = {
    'New York': [40.712776, -74.005974],
    'Los Angeles': [34.052235, -118.243683],
    'Chicago': [41.878113, -87.629799],
    'Houston': [29.760427, -95.369804],
    'Phoenix': [33.448376, -112.074036],
    'Philadelphia': [39.952583, -75.165222],
    'San Antonio': [29.424122, -98.493629],
    'San Diego': [32.715736, -117.161087],
    'Dallas': [32.776665, -96.796989],
    'San Jose': [37.338207, -121.886330],
    'Toronto': [43.651070, -79.347015],
    'Vancouver': [49.282729, -123.120738],
    'Montreal': [45.501690, -73.567253],
    'Ottawa': [45.421530, -75.697193],
    'Calgary': [51.044733, -114.071883],
    'Edmonton': [53.546124, -113.493823],
    'Winnipeg': [49.895077, -97.138451],
    'Quebec City': [46.813878, -71.207981],
    'Halifax': [44.648618, -63.585948],
    'St. John\'s': [47.561510, -52.712576]
}

@st.cache_data(persist=True)
def load_data():
    data = pd.read_csv(data_url)
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    
    # Suppose the 'tweet_coord' column contains coordinates in the form '[lat, lon]'
    data['tweet_coord'] = data['tweet_coord'].apply(lambda x: eval(x) if pd.notnull(x) else [None, None])
    data[['latitude', 'longitude']] = pd.DataFrame(data['tweet_coord'].tolist(), index=data.index)
    
    # Remplir les valeurs manquantes de latitude et longitude
    for index, row in data.iterrows():
        if pd.isnull(row['latitude']) or pd.isnull(row['longitude']):
            # Attribuer aléatoirement une coordonnée à partir des villes définies
            random_city = random.choice(list(location_coords.values()))
            data.at[index, 'latitude'] = random_city[0]
            data.at[index, 'longitude'] = random_city[1]
    
    return data

data = load_data()

# Fonction pour charger les animations Lottie
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Animation Lottie
lottie_twitter = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_2tlylbqv.json")

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

st.sidebar.subheader("When and where users tweeting from ?")  

hour = st.sidebar.slider('Hour of day', 0, 23)  
modified_data = data[data['tweet_created'].dt.hour == hour]
if not st.sidebar.checkbox("Close", True, key='2'):
    st.markdown('### Tweets locations based on the time of day')
    st.markdown("%i Tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour + 1) % 24))
    # Filtrer les données pour afficher uniquement les lignes avec des coordonnées valides
    filtered_data = modified_data.dropna(subset=['latitude', 'longitude'])
    # Afficher la carte si des données valides existent
    if not filtered_data.empty:
        st.map(filtered_data[['latitude', 'longitude']])
    if st.sidebar.checkbox("Show raw data", False):
        st.write(modified_data)

st.sidebar.subheader('Breakdown Airline Tweets by Sentiment')
choice = st.sidebar.multiselect('Pick Airlines', ('US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America'), key='0')

if len(choice) > 0:
    choice_data = data[data.airline.isin(choice)]
    fig_choice = px.histogram(choice_data, x='airline', y='airline_sentiment', histfunc='count', color='airline_sentiment', labels={'airline_sentiment': 'tweets'}, height=600, width=800, template='plotly_dark')
    st.plotly_chart(fig_choice)

st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment ?', ('positive', 'neutral', 'negative'))
if not st.sidebar.checkbox('Close', True, key='3'):
    st.header("Word Cloud for %s sentiment" % (word_sentiment))
    df = data[data['airline_sentiment'] == word_sentiment]
    words = ' '.join(df['text'])
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=640, width=800).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
