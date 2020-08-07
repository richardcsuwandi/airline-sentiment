import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def main():
    st.title("Sentiment Analysis on Tweets about US Airlines ✈️")
    st.sidebar.title("Sentiment Analysis on Tweets about US Airlines ✈️")
    st.sidebar.subheader("By [Richard Cornelius Suwandi](https://github.com/richardcsuwandi)")
    st.sidebar.markdown("[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/richardcsuwandi/airline-sentiment)")
    st.subheader("A data-driven web app that visualizes various sentiment and exploratory analysis on tweets about US airlines.")
    st.sidebar.markdown("A data-driven web app that visualizes various sentiment and exploratory analysis on tweets about US airlines.")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv("tweets.csv")
        data["tweet_created"] = pd.to_datetime(data["tweet_created"])
        return data

    data = load_data()

    # Show random tweet
    st.sidebar.subheader("Show Random Tweet")
    random_tweet = st.sidebar.radio("Sentiment", ("positive", "neutral", "negative"))
    if not st.sidebar.checkbox("Hide", True, key='0'):
        st.subheader(f"Random {random_tweet.capitalize()} Tweet")
        st.header(data.query("airline_sentiment == @random_tweet")[["text"]].sample(n=1).iat[0, 0])

    # Number of tweets by sentiment
    st.sidebar.subheader("Number of Tweets by Sentiment")
    select = st.sidebar.selectbox("Visualization Type", ["Bar Plot", "Pie Chart"])
    sentiment_count = data["airline_sentiment"].value_counts()
    sentiment_count = pd.DataFrame({"Sentiment":sentiment_count.index, "Tweets":sentiment_count.values})
    if not st.sidebar.checkbox("Hide", True, key='1'):
        st.subheader("Number of Tweets by Sentiment")
        if select == "Bar Plot":
            fig = px.bar(sentiment_count, x="Sentiment", y="Tweets", color="Tweets")
            st.plotly_chart(fig)
        if select == "Pie Chart":
            fig = px.pie(sentiment_count, values="Tweets", names="Sentiment")
            st.plotly_chart(fig)

    # Tweet locations based on time of day
    st.sidebar.subheader("Tweet Locations Based on Time of Day")
    hour = st.sidebar.slider("Hour to Look at", 0, 23)
    selected_data = data[data["tweet_created"].dt.hour == hour]
    if not st.sidebar.checkbox("Hide", True, key="2"):
        st.subheader("Tweet Locations Based on Time of Day")
        st.markdown(f"{len(selected_data)} tweets between {hour}:00 and {(hour + 1) % 24}:00")
        st.map(selected_data)

    # Number of tweets for each airline
    st.sidebar.subheader("Number of Tweets for Each Airline")
    each_airline = st.sidebar.selectbox("Visualization Type", ["Bar Plot", "Pie Chart"], key="3")
    airline_sentiment_count = data.groupby("airline")["airline_sentiment"].count().sort_values(ascending=False)
    airline_sentiment_count = pd.DataFrame({"Airline":airline_sentiment_count.index, "Tweets":airline_sentiment_count.values.flatten()})
    if not st.sidebar.checkbox("Hide", True, key="4"):
        if each_airline == "Bar Plot":
            st.subheader("Number of Tweets for Each Airline")
            fig = px.bar(airline_sentiment_count, x="Airline", y="Tweets", color="Tweets")
            st.plotly_chart(fig)
        if each_airline == "Pie Chart":
            st.subheader("Number of Tweets for Each Airline")
            fig = px.pie(airline_sentiment_count, values="Tweets", names="Airline")
            st.plotly_chart(fig)

    # Breakdown airline tweets by sentiment
    st.sidebar.subheader("Breakdown Airline Tweets by Sentiment")
    choice = st.sidebar.multiselect("Pick Airline(s)", tuple(pd.unique(data["airline"])))
    if not st.sidebar.checkbox("Hide", True, key="5"):
        if len(choice) > 0:
            chosen_data = data[data["airline"].isin(choice)]
            fig = px.histogram(chosen_data, x="airline", y="airline_sentiment",
                                histfunc="count", color="airline_sentiment",
                                facet_col="airline_sentiment", labels={"airline_sentiment": "sentiment"})
            st.plotly_chart(fig)

    # Word cloud
    st.sidebar.subheader("Word Cloud")
    word_sentiment = st.sidebar.radio("Which Sentiment to Display?", tuple(pd.unique(data["airline_sentiment"])))
    if not st.sidebar.checkbox("Hide", True, key="6"):
        st.subheader(f"Word Cloud for {word_sentiment.capitalize()} Sentiment")
        df = data[data["airline_sentiment"]==word_sentiment]
        words = " ".join(df["text"])
        processed_words = " ".join([word for word in words.split() if "http" not in word and not word.startswith("@") and word != "RT"])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640).generate(processed_words)
        plt.imshow(wordcloud)
        plt.xticks([])
        plt.yticks([])
        st.pyplot()

if __name__ == "__main__":
    main()