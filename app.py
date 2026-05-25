import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

# --- GLOBAL STYLES: Vintage Maritime Archive ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f5f1e8;
    background-image: radial-gradient(#e8e0d0 1px, transparent 1px);
    background-size: 24px 24px;
    color: #1a2f4a;
}

.hero-title {
    font-family: 'Cinzel', serif;
    font-size: 2.4rem;
    font-weight: 700;
    text-align: center;
    color: #1a2f4a;
    letter-spacing: 0.04em;
    margin-bottom: 0.3rem;
    text-shadow: 0 1px 2px rgba(26,47,74,0.08);
}
.hero-subtitle {
    font-size: 0.95rem;
    text-align: center;
    color: #8a7d6b;
    margin-bottom: 2rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 500;
}

.divider-ornament {
    display: flex;
    align-items: center;
    text-align: center;
    margin: 1.5rem 0;
    color: #c75b39;
    font-size: 1.2rem;
}
.divider-ornament::before,
.divider-ornament::after {
    content: '';
    flex: 1;
    border-bottom: 1px solid #d4c5b0;
}
.divider-ornament::before { margin-right: 1em; }
.divider-ornament::after  { margin-left:  1em; }

.about-card {
    background: #ffffff;
    border: 1px solid #e0d8c8;
    border-left: 5px solid #c75b39;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 28px;
    color: #3d4f65;
    line-height: 1.7;
    box-shadow: 0 2px 12px rgba(26,47,74,0.06);
}
.about-card h3 {
    color: #1a2f4a;
    font-family: 'Cinzel', serif;
    margin-bottom: 10px;
    font-size: 1.15rem;
}

h2, h3 {
    color: #1a2f4a !important;
    font-family: 'Cinzel', serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}

.stSelectbox label, .stSlider label,
.stNumberInput label, .stRadio label {
    color: #5a6d7f !important;
    font-weight: 500;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background: #ffffff;
    border: 1px solid #e0d8c8;
    border-radius: 10px;
    padding: 6px;
    box-shadow: 0 2px 10px rgba(26,47,74,0.04);
}

.stButton > button {
    background: #1a2f4a;
    color: #f5f1e8 !important;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border: 2px solid #d4af37;
    border-radius: 6px;
    padding: 14px 0;
    width: 100%;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 3px 14px rgba(26,47,74,0.15);
}
.stButton > button:hover {
    background: #c75b39;
    border-color: #c75b39;
    color: #ffffff !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(199,91,57,0.25);
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e0d8c8;
    border-top: 4px solid #d4af37;
    border-radius: 8px;
    padding: 16px 18px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(26,47,74,0.05);
}
[data-testid="stMetricLabel"] {
    color: #8a7d6b !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}
[data-testid="stMetricValue"] {
    color: #1a2f4a !important;
    font-family: 'Cinzel', serif !important;
    font-size: 1.45rem !important;
    font-weight: 700;
}

.result-box {
    border-radius: 8px;
    padding: 18px 24px;
    text-align: center;
    font-size: 1.25rem;
    font-weight: 700;
    font-family: 'Cinzel', serif;
    margin: 14px 0;
    letter-spacing: 0.04em;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
}
.result-survived     { background: #ffffff; border: 2px solid #2d6a4f; color: #2d6a4f; }
.result-not-survived { background: #ffffff; border: 2px solid #9d0208; color: #9d0208; }

hr {
    border-color: #d4c5b0 !important;
    border-width: 1px !important;
}
[data-testid="stCaptionContainer"] p {
    color: #9e9485 !important;
    font-size: 0.78rem;
}

/* Radio option labels — force dark text so they're visible on parchment */
[data-testid="stRadio"] div[role="radiogroup"] label {
    color: #1a2f4a !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label p {
    color: #1a2f4a !important;
}
/* Also fix any span text inside radio options */
[data-testid="stRadio"] div[role="radiogroup"] label span {
    color: #1a2f4a !important;
}
</style>
""", unsafe_allow_html=True)

# --- SECTION 1: Hero Header ---
st.markdown("<div class='hero-title'>RMS Titanic</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Survival Prediction Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='divider-ornament'>❖</div>", unsafe_allow_html=True)

# --- SECTION 2: About ---
st.markdown("""
<div class='about-card'>
    <h3>Passenger Manifest Notes</h3>
    <p>This instrument predicts the survival probability of a passenger aboard the RMS Titanic
    using a trained Artificial Neural Network. Adjust the passenger details below to observe how
    class, age, fare, and embarkation port influenced the model's assessment of survival likelihood
    during the maiden voyage of April 1912.</p>
</div>
""", unsafe_allow_html=True)

# --- SECTION 3: Load Model ---
model_path = './titanic_model.h5'

@st.cache_resource
def load_model():
    try:
        # compile=False avoids Keras 3 metric-deserialization errors (e.g. 'keras.metrics.mse')
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(
            f"Unable to load model from '{model_path}'. "
            f"Ensure the file exists alongside app.py. Details: {e}"
        )
        return None

model = load_model()

if model is None:
    st.stop()

# --- SECTION 4: Passenger Input Form ---
st.header("Passenger Details")

with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        pclass   = st.selectbox("Passenger Class", [1, 2, 3], index=2,
                                help="1st = Upper · 2nd = Middle · 3rd = Lower")
        age      = st.slider("Age", 0, 80, 22)
        sibsp    = st.number_input("Siblings / Spouses Aboard", min_value=0, max_value=8, value=0)
        parch    = st.number_input("Parents / Children Aboard", min_value=0, max_value=6, value=0)
    with col2:
        fare     = st.number_input("Fare (£)", min_value=0.0, max_value=512.0, value=7.25, step=0.01)
        sex      = st.radio("Sex", ["male", "female"], index=0)
        embarked = st.radio("Port of Embarkation", ["C", "Q", "S"], index=2,
                            help="C = Cherbourg · Q = Queenstown · S = Southampton")

# --- SECTION 5: Feature detection & adaptive preprocessing ---
def get_expected_features(model):
    try:
        return model.layers[0].get_weights()[0].shape[0]
    except Exception:
        pass
    try:
        return model.input_shape[-1]
    except Exception:
        return 8

N_FEATURES = get_expected_features(model)

FEATURE_SETS = {
    3: ['Pclass', 'Sex_male', 'Age'],
    4: ['Pclass', 'Sex_male', 'Age', 'Fare'],
    5: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
    6: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male'],
    7: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_S'],
    8: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S'],
}

def preprocess_input(pclass, age, sibsp, parch, fare, sex, embarked):
    full = {
        'Pclass':     pclass,
        'Age':        age,
        'SibSp':      sibsp,
        'Parch':      parch,
        'Fare':       fare,
        'Sex_male':   1 if sex == 'male' else 0,
        'Embarked_Q': 1 if embarked == 'Q' else 0,
        'Embarked_S': 1 if embarked == 'S' else 0,
    }
    cols = FEATURE_SETS.get(N_FEATURES, list(full.keys())[:N_FEATURES])
    return np.array([[full[c] for c in cols]], dtype='float32')

# --- SECTION 6: Prediction ---
st.markdown("<div class='divider-ornament'>⚓</div>", unsafe_allow_html=True)
st.header("Forecast")

if st.button("Compute Survival Probability", type="primary", use_container_width=True):
    processed_input = preprocess_input(pclass, age, sibsp, parch, fare, sex, embarked)
    prediction_proba = float(model.predict(processed_input)[0][0])

    survival_pct     = prediction_proba * 100
    non_survival_pct = (1 - prediction_proba) * 100
    survived         = prediction_proba > 0.5
    confidence       = abs(prediction_proba - 0.5) * 200

    result_class = 'result-survived' if survived else 'result-not-survived'
    result_label = 'Survived — Probability Favorable' if survived else 'Did Not Survive — Probability Unfavorable'
    st.markdown(f"<div class='result-box {result_class}'>{result_label}</div>",
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Forecasted Outcome",  "Survived" if survived else "Not Survived")
    with c2:
        st.metric("Survival Likelihood", f"{survival_pct:.1f}%")
    with c3:
        st.metric("Model Confidence",    f"{confidence:.1f}%")

    st.markdown("<h4 style='text-align:center; margin-top:24px; color:#1a2f4a; font-family:Cinzel,serif;'>Probability Distribution</h4>",
                unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#f5f1e8')
    ax.set_facecolor('#f5f1e8')

    bars = ax.bar(
        ['Survived', 'Not Survived'],
        [survival_pct, non_survival_pct],
        color=['#2d6a4f', '#9d0208'],
        width=0.4,
        edgecolor='#1a2f4a',
        linewidth=0.8,
    )
    ax.bar_label(bars, fmt='%.1f%%', padding=5,
                 color='#1a2f4a', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.set_ylabel("Probability (%)", color='#5a6d7f', fontsize=10)
    ax.set_title("Survival Assessment", color='#1a2f4a',
                 fontsize=13, fontweight='bold', pad=14)
    ax.tick_params(colors='#5a6d7f', labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#d4c5b0')
        spine.set_linewidth(0.8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)