import streamlit as st
import joblib
import re

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

st.title("Flipkart Review Sentiment Analyzer")

review = st.text_area("Enter your review")

if st.button("Analyze"):
    if review.strip():
        cleaned_review = clean_text(review)
        vec = vectorizer.transform([cleaned_review])
        prediction = model.predict(vec)
        
        if prediction[0] == 1:
            st.success("Positive")
        else:
            st.error("Negative")
    else:
        st.warning("Please enter a review")
