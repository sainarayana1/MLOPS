import streamlit as st
import joblib


model = joblib.load("sentiment_model.pkl")

st.title("Flipkart Review Sentiment Analyzer")

review = st.text_area("Enter your review")

if st.button("Analyze"):
    if review.strip():
        
        prediction = model.predict([review])
        
       
        if prediction[0] == 1:
            st.success("Positive")
        else:
            st.error("Negative")
    else:
        st.warning("Please enter a review")
