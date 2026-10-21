import streamlit as st
import joblib

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("🏨 Hotel Review Sentiment Analysis")
st.write("Enter a hotel review to predict its sentiment.")

review = st.text_area("Write your hotel review:")

if st.button("Predict Sentiment"):
    if review.strip():
        review_tfidf = tfidf.transform([review])
        prediction = model.predict(review_tfidf)[0]

        st.subheader(f"Sentiment: {prediction}")
    else:
        st.warning("Please enter a review.")