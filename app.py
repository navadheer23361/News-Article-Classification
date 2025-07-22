import streamlit as st
import pickle
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (only first time)
nltk.download('stopwords')

# Text preprocessing
def clean_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered)

# Load model and vectorizer
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# UI
st.title("📰 Fake News Detector")
user_input = st.text_area("Paste a news article:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess and predict
        cleaned = clean_text(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]

        # Display prediction
        if prediction == 1:
            st.success("✅ This appears to be **REAL** news.")
        else:
            st.error("❌ This appears to be **FAKE** news.")

        # Display confidence score
        st.write(f"🧠 Model confidence:")
        st.write(f"- **REAL**: {proba[1]*100:.2f}%")
        st.write(f"- **FAKE**: {proba[0]*100:.2f}%")

        # Optional: Top influencing words (Logistic Regression)
        feature_names = tfidf.get_feature_names_out()
        coef = model.coef_[0]
        input_vec = vectorized.toarray()[0]

        top_indices = np.argsort(input_vec * coef)[-5:][::-1]
        top_words = [(feature_names[i], coef[i]) for i in top_indices if input_vec[i] > 0]

        st.markdown("🔍 **Top keywords influencing prediction:**")
        for word, weight in top_words:
            st.write(f"- {word} (weight: {weight:.3f})")
