import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Streamlit UI
st.set_page_config(page_title="Spam Classifier", layout="centered")
st.title("📩 SMS Spam Classifier")
st.write("Enter a message below to check if it's **spam** or **not**.")

user_input = st.text_area("Message", height=150)

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    label = "🚨 SPAM" if prediction == 1 else "✅ NOT SPAM"
    st.subheader(f"Result: {label}")
