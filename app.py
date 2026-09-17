

import streamlit as st
import joblib
from pathlib import Path


@st.cache_resource
def load_models():
    """Load the trained models relative to this application file."""
    model_dir = Path(__file__).resolve().parent
    return (
        joblib.load(model_dir / "vectorization.pkl"),
        joblib.load(model_dir / "lr_model.pkl"),
        joblib.load(model_dir / "rf_model.pkl"),
    )


vectorization, lr_model, rf_model = load_models()

# Page configuration
st.set_page_config(
    page_title="Fake News Detection",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #111827;
    }
    [data-testid="stHeader"] {
        background: rgba(17, 24, 39, 0.92);
    }
    .block-container {
        max-width: 900px;
        padding: 4rem 1.5rem 5rem;
    }
    .hero {
        padding: 1.8rem 2rem;
        border: 1px solid #2d3a50;
        border-radius: 12px;
        background: #1b2535;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
    }
    .eyebrow {
        color: #9aa9bf;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .hero h1 {
        color: #f3f4f6;
        font-size: clamp(1.8rem, 4vw, 2.6rem);
        letter-spacing: -0.025em;
        line-height: 1.15;
        margin: 0;
    }
    .hero p {
        color: #b5c0d0;
        font-size: 0.98rem;
        margin: 0.75rem 0 0;
    }
    .section-label {
        color: #e5e7eb;
        font-size: 1.05rem;
        font-weight: 700;
        margin: 1.8rem 0 0.55rem;
    }
    .hint {
        color: #9aa9bf;
        font-size: 0.86rem;
        margin-bottom: 0.8rem;
    }
    [data-testid="stTextArea"] textarea {
        background: #182235;
        border: 1px solid #3a4961;
        border-radius: 8px;
        color: #f3f4f6;
        font-size: 0.98rem;
        line-height: 1.6;
        padding: 1rem;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #6b9cff;
        box-shadow: 0 0 0 1px #6b9cff;
    }
    [data-testid="stTextArea"] textarea::placeholder {
        color: #aab6c8 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #aab6c8;
    }
    .stButton > button {
        background: #386edb;
        border: 0;
        border-radius: 8px;
        color: white;
        font-size: 0.95rem;
        font-weight: 700;
        min-height: 3rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton > button:hover {
        background: #2f5fc5;
        box-shadow: 0 5px 12px rgba(37, 99, 235, 0.2);
        color: white;
    }
    [data-testid="stAlert"] {
        border-radius: 14px;
    }
    [data-testid="stMetric"] {
        background: #182235;
        border: 1px solid #2d3a50;
        border-radius: 10px;
        padding: 0.9rem 1rem;
    }
    [data-testid="stMetricLabel"] {
        color: #9aa9bf;
    }
    [data-testid="stMetricValue"] {
        color: #f3f4f6;
    }
    .footer-note {
        color: #8492a8;
        font-size: 0.78rem;
        margin-top: 2rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">News classification</div>
        <h1>Fake News Detection</h1>
        <p>Review an article using predictions from two trained machine learning models.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Paste an article to begin</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hint">For the best result, include the full headline and article text.</div>',
    unsafe_allow_html=True,
)

news_text = st.text_area(
    "News article",
    height=250,
    placeholder="Paste the headline and article content here...",
    label_visibility="collapsed",
)

# Prediction button
if st.button("Detect article", use_container_width=True):

    if news_text.strip() == "":
        st.warning("Please enter a news article.")

    else:
        # Convert text into TF-IDF features
        news_vector = vectorization.transform([news_text])

        # Get probabilities
        lr_prob = lr_model.predict_proba(news_vector)[0]
        rf_prob = rf_model.predict_proba(news_vector)[0]

        # Ensemble probability
        final_prob = (lr_prob + rf_prob) / 2

        fake_probability = final_prob[0] * 100
        real_probability = final_prob[1] * 100

        # Final classification
        if max(fake_probability, real_probability) < 60:
            result = "UNCERTAIN"
        elif fake_probability > real_probability:
            result = "FAKE NEWS"
        else:
            result = "REAL NEWS"

        # Model predictions
        lr_prediction = lr_model.predict(news_vector)[0]
        rf_prediction = rf_model.predict(news_vector)[0]

        lr_label = "FAKE" if lr_prediction == 0 else "REAL"
        rf_label = "FAKE" if rf_prediction == 0 else "REAL"

        # Display result
        st.markdown('<div class="section-label">Detection result</div>', unsafe_allow_html=True)

        if result == "FAKE NEWS":
            st.error("FAKE NEWS")
        elif result == "REAL NEWS":
            st.success("REAL NEWS")
        else:
            st.warning("UNCERTAIN")

        # Probabilities
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Fake Probability",
                f"{fake_probability:.2f}%"
            )

        with col2:
            st.metric(
                "Real Probability",
                f"{real_probability:.2f}%"
            )

        # Model agreement
        st.markdown('<div class="section-label">Model agreement</div>', unsafe_allow_html=True)
        model_col1, model_col2 = st.columns(2)
        with model_col1:
            st.markdown(f"**Logistic Regression**  \n`{lr_label}`")
        with model_col2:
            st.markdown(f"**Random Forest**  \n`{rf_label}`")

        st.progress(
            int(real_probability),
            text=f"Real News Probability: {real_probability:.2f}%"
        )

st.markdown(
    '<div class="footer-note">Predictions reflect patterns learned from the training data '
    "and are not independent fact-checks.</div>",
    unsafe_allow_html=True,
)