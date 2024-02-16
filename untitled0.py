import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    processed_text = ' '.join([word for word in text.lower().split() if word not in stop_words])
    return processed_text

def analyze_sentiment(text):
    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=text)
    compound = round((1 + dd['compound'])/2, 2)
    return dd['pos'], dd['neg'], dd['neu'], compound

st.title('Sentiment Analysis')

# Custom CSS styles
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        text-align: center;
        margin-bottom: 20px;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

movie_title = st.text_input('Enter a movie title:')
text = preprocess_text(movie_title)

if st.button('Analyze', key='analyze_button'):
    pos, neg, neu, compound = analyze_sentiment(text)
    st.markdown(f"**Movie Title:** {movie_title}")
    st.markdown(f"**Positive Sentiment:** {pos}")
    st.markdown(f"**Negative Sentiment:** {neg}")
    st.markdown(f"**Neutral Sentiment:** {neu}")
    st.markdown(f"**Compound Sentiment:** {compound}")
