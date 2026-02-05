import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")

st.title("Flipkart Review Sentiment Analyzer")

review = st.text_area("Enter your review")

if st.button("Analyze"):
    if review.strip():
        try:
            pred = model.predict([review])
            result = "Positive" if pred[0] == 1 else "Negative"
            st.success(result)
        except AttributeError:
            vectorizer = joblib.load("vectorizer.pkl")
            vec = vectorizer.transform([review])
            pred = model.predict(vec)
            result = "Positive" if pred[0] == 1 else "Negative"
            st.success(result)
    else:
        st.warning("Please enter a review")
