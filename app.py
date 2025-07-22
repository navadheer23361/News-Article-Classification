import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (needed only once)
nltk.download('stopwords')

# Clean text function
def clean_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered)

# Load trained model and vectorizer
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector")
st.title("📰 Fake News Detector")
st.markdown("Enter any news article text below:")

user_input = st.text_area("Paste news article here")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = tfidf.transform([cleaned]).toarray()
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.success("✅ This appears to be **REAL** news.")
        else:
            st.error("❌ This appears to be **FAKE** news.")
