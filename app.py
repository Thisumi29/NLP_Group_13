import streamlit as st
import joblib
import pickle
import re
import numpy as np
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================
# Setup — download NLTK resources (only runs once, then cached)
# ============================================
@st.cache_resource
def setup_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()

stop_words, lemmatizer = setup_nltk()

# ============================================
# Load both models (cached so they only load once)
# ============================================
PROJECT_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = PROJECT_DIR / "saved_models"

MAX_LEN = 300  # must match bilstm_model.py training setting

@st.cache_resource
def load_rf_model():
    model = joblib.load(SAVED_MODELS_DIR / "random_forest_model.pkl")
    vectorizer = joblib.load(SAVED_MODELS_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer

@st.cache_resource
def load_bilstm_model():
    model = load_model(SAVED_MODELS_DIR / "bilstm_model.keras")
    with open(SAVED_MODELS_DIR / "bilstm_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

rf_model, rf_vectorizer = load_rf_model()
bilstm_model, bilstm_tokenizer = load_bilstm_model()

# ============================================
# Preprocessing — matches src/data_preprocessing.py exactly
# ============================================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text: str) -> str:
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]
    return " ".join(tokens)

def full_pipeline(raw_text: str) -> str:
    cleaned = clean_text(raw_text)
    processed = preprocess_text(cleaned)
    return processed

# ============================================
# Prediction functions
# ============================================
def predict_random_forest(processed_text: str):
    features = rf_vectorizer.transform([processed_text])
    prediction = rf_model.predict(features)[0]
    confidence = rf_model.predict_proba(features)[0].max()
    return prediction, confidence

def predict_bilstm(processed_text: str):
    sequence = bilstm_tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
    probability = bilstm_model.predict(padded, verbose=0)[0][0]
    prediction = int(probability >= 0.5)
    confidence = probability if prediction == 1 else 1 - probability
    return prediction, confidence

# ============================================
# Streamlit UI
# ============================================
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.write(
    "Paste a news article below and choose which of Oshini's best-performing "
    "models to use for the prediction."
)

user_input = st.text_area("Paste a news article here:", height=250)

if st.button("Check Article"):
    if not user_input.strip():
        st.warning("Please paste some article text first.")
    else:
        processed_text = full_pipeline(user_input)

        col1, col2 = st.columns(2)

        rf_pred, rf_conf = predict_random_forest(processed_text)
        rf_label = "🟢 Real News" if rf_pred == 1 else "🔴 Fake News"
        with col1:
            st.subheader("Random Forest (ML)")
            st.write(f"**Prediction:** {rf_label}")
            st.write(f"**Confidence:** {rf_conf:.2%}")

        bilstm_pred, bilstm_conf = predict_bilstm(processed_text)
        bilstm_label = "🟢 Real News" if bilstm_pred == 1 else "🔴 Fake News"
        with col2:
            st.subheader("Bi-LSTM (DL)")
            st.write(f"**Prediction:** {bilstm_label}")
            st.write(f"**Confidence:** {bilstm_conf:.2%}")

        st.divider()
        if rf_pred == bilstm_pred:
            final_label = "🟢 Real News" if rf_pred == 1 else "🔴 Fake News"
            st.subheader(f"✅ Both models agree: {final_label}")
        else:
            st.subheader("⚠️ Models disagree — review manually")

        with st.expander("See processed text (what the models actually saw)"):
            st.write(processed_text)

st.divider()
st.warning(
    "⚠️ **Known limitation:** Testing showed the Random Forest model is sensitive to "
    "source-related keywords (e.g., 'Reuters') rather than purely evaluating "
    "article content. This is a documented bias in the training dataset. "
    "Predictions should be treated as advisory only."
)
st.caption(
    "Models: Random Forest & Bi-LSTM (Member 3 — Oshini) | "
    "Trained on Fake and Real News Dataset (Kaggle) | "
    "This tool is advisory only — always verify news through trusted sources."
)