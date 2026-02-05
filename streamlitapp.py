import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Flipkart Review Sentiment Analyzer")

review = st.text_area("Enter your review")

if st.button("Analyze"):
    vec = vectorizer.transform([review])
    pred = model.predict(vec)
    st.success("Positive" if pred[0] == 1 else "Negative")
