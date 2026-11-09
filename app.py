import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page settings
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Title
st.title("📰 Fake News Detection")
st.write("Enter a news headline or article below and check whether it is predicted as **Fake** or **Real**.")

# Input box
news = st.text_area(
    "Enter News Text",
    height=220,
    placeholder="Paste a news headline or full article here..."
)

# Prediction button
if st.button("Detect News"):
    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        vector = vectorizer.transform([news])
        prediction = model.predict(vector)[0]

        if prediction == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")