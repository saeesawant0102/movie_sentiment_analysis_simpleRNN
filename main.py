import streamlit as st
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="CineBot AI",
    page_icon="🎬",
    layout="centered"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">

<style>

/* App Background */
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: #f8fafc;
    font-family: 'Share Tech Mono', monospace;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    max-width: 800px;
}

/* Main title */
.main-title {
    text-align: center;
    font-family: 'Orbitron', sans-serif;
    color: #38bdf8;
    font-size: 3.2rem;
    letter-spacing: 4px;
    text-shadow: 4px 4px 0px #facc15;
    margin-bottom: 0.5rem;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 2rem;
    font-size: 1rem;
}

/* Text area label */
.stTextArea label {
    color: #f8fafc !important;
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 1px;
}

/* Text area */
.stTextArea textarea {
    background-color: #1e293b;
    color: #ffffff;
    border: 4px solid #38bdf8;
    border-radius: 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 16px;
    min-height: 180px;
    box-shadow: 8px 8px 0px #f59e0b;
}

/* Button */
.stButton > button {
    width: 100%;
    background-color: #facc15;
    color: #111827;
    border: 4px solid #111827;
    border-radius: 0;
    padding: 1rem;
    font-family: 'Orbitron', sans-serif;
    font-weight: bold;
    letter-spacing: 2px;
    box-shadow: 8px 8px 0px #fb923c;
    transition: all 0.1s ease;
}

/* Hover */
.stButton > button:hover {
    transform: translate(2px, 2px);
    box-shadow: 6px 6px 0px #fb923c;
}

/* Click effect */
.stButton > button:active {
    transform: translate(8px, 8px);
    box-shadow: none;
}

/* Result card */
.result-box {
    background-color: #1e293b;
    border: 4px solid #38bdf8;
    padding: 25px;
    margin-top: 25px;
    box-shadow: 10px 10px 0px #facc15;
}

.result-title {
    font-family: 'Orbitron', sans-serif;
    color: #38bdf8;
    font-size: 2rem;
    margin-bottom: 15px;
}

.result-score {
    font-family: 'Share Tech Mono', monospace;
    color: #fb923c;
    font-size: 1.2rem;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background-color: #38bdf8;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL AND WORD INDEX
# --------------------------------------------------

@st.cache_resource
def load_resources():
    model = load_model("simple_rnn_imdb.h5")
    word_index = imdb.get_word_index()
    return model, word_index

model, word_index = load_resources()

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

def preprocess_text(text):
    words = text.lower().split()

    encoded_review = [
        word_index.get(word, 2) + 3
        for word in words
    ]

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review

# --------------------------------------------------
# UI
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎬 CINEBOT AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">⚡ Feed a movie review into the sentiment engine.</div>',
    unsafe_allow_html=True
)

user_input = st.text_area(
    "MOVIE REVIEW",
    placeholder="Type your review here..."
)

if st.button("ANALYZE SENTIMENT 🤖"):

    if user_input.strip():

        preprocessed_input = preprocess_text(user_input)

        prediction = model.predict(
            preprocessed_input,
            verbose=0
        )

        score = float(prediction[0][0])

        if score > 0.5:
            sentiment = "😊 Positive"
            confidence = score
        else:
            sentiment = "😞 Negative"
            confidence = 1 - score

        result_html = f"""<div class="result-box">
<div class="result-title">{sentiment}</div>
<div class="result-score">Confidence Score: {confidence * 100:.2f}%</div>
</div>"""

        st.markdown(result_html, unsafe_allow_html=True)

        st.progress(confidence)

    else:
        st.warning("⚠️ Please enter a movie review.")