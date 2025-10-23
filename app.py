# app.py
import streamlit as st
from PIL import Image
import numpy as np
import os
import joblib
import tensorflow as tf
import random
from io import BytesIO

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Mental Health & Brain MRI Demo", layout="wide",
                   initial_sidebar_state="expanded")

MODEL_DIR = "models"
SAMPLE_DIR = os.path.join(MODEL_DIR, "sample_images")  # folder for sample MRI images
MENTAL_MODEL_PATH = os.path.join(MODEL_DIR, "mental_model.joblib")
BRAIN_MODEL_PATH = os.path.join(MODEL_DIR, "brain_model.h5")

# ---------- STYLES (gradient + chat bubbles) ----------
st.markdown(
    """
    <style>
    .gradient-header {
      background: linear-gradient(90deg,#7b2ff7,#f107a3);
      padding: 28px;
      border-radius: 12px;
      color: white;
      box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }
    .module-card {
      background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(250,250,250,0.9));
      border-radius: 12px;
      padding: 18px;
      box-shadow: 0 6px 16px rgba(12,12,12,0.06);
    }
    .chat-bubble {
      margin: 8px 0;
      padding: 12px 16px;
      border-radius: 12px;
      max-width: 80%;
      line-height: 1.4;
    }
    .chat-user {
      background: linear-gradient(90deg,#f6d365,#fda085);
      margin-left: auto;
      color: #222;
    }
    .chat-bot {
      background: linear-gradient(90deg,#c3ec52,#0ba29d);
      color: #042;
    }
    .result-card {
      padding: 16px;
      border-radius: 12px;
      color: white;
    }
    .small-muted { color: #777; font-size:12px; }
    </style>
    """, unsafe_allow_html=True
)

# ---------- Helpers: load models ----------
@st.cache_resource
def load_mental_model(path=MENTAL_MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Failed to load mental model: {e}")
    return None

@st.cache_resource
def load_brain_model(path=BRAIN_MODEL_PATH):
    if os.path.exists(path):
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            st.error(f"Failed to load brain model: {e}")
    return None

mental_model = load_mental_model()
brain_model = load_brain_model()

# ---------- Sidebar ----------
with st.sidebar:
    st.title("AI Health Demo")
    st.write("Modules:")
    page = st.radio("", ["Home", "Mental Health", "Brain MRI"])
    st.markdown("---")
    st.write("⚠️ **Disclaimer:** Demo — not a medical device. Consult professionals for advice.")
    st.markdown("**Models:**")
    st.write(f"- Mental model: {'✅' if mental_model else '❌ (missing)'}")
    st.write(f"- Brain model: {'✅' if brain_model else '❌ (missing)'}")
    st.markdown("---")
    st.write("Created for learning. Keep personal data private.")

# ---------- Home ----------
if page == "Home":
    st.markdown('<div class="gradient-header"><h1>AI Mental Health & Brain MRI Demo</h1>'
                '<p style="opacity:0.95">Two modules — mental health assessment (chat-like + questionnaire) '
                'and brain MRI image check.</p></div>', unsafe_allow_html=True)
    st.write("")
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown('<div class="module-card">', unsafe_allow_html=True)
        st.subheader("🧠 Mental Health Check")
        st.write("Conversational assessment + 10-question PHQ-style questionnaire. You can also type free-form text for the model.")
        if st.button("Go to Mental Health"):
            page = "Mental Health"
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="module-card">', unsafe_allow_html=True)
        st.subheader("🩻 Brain MRI Check")
        st.write("Upload MRI image → preview → model predicts tumor / no tumor with confidence.")
        if st.button("Go to Brain MRI"):
            page = "Brain MRI"
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Mental Health Module ----------
elif page == "Mental Health":
    st.markdown('<div class="gradient-header"><h2>🧠 Mental Health — Chat-style Assessment</h2>'
                '<p class="small-muted">A friendly questionnaire + optional text analysis. Demo only.</p></div>', unsafe_allow_html=True)
    st.write("**How it works:** Sequence of PHQ-style questions. Combined risk estimate using questionnaire and text model.")

    if "mh_chat" not in st.session_state:
        st.session_state.mh_chat = []
        st.session_state.q_index = 0
        st.session_state.answers = []

    questions = [
        "Little interest or pleasure in doing things?",
        "Feeling down, depressed, or hopeless?",
        "Trouble falling or staying asleep, or sleeping too much?",
        "Feeling tired or having little energy?",
        "Poor appetite or overeating?",
        "Feeling bad about yourself — or that you are a failure?",
        "Trouble concentrating on things, such as reading or watching TV?",
        "Moving or speaking so slowly that other people could have noticed? Or the opposite — being fidgety?",
        "Thoughts that you would be better off dead or hurting yourself?",
        "Have you experienced sudden panic attacks or intense fear recently?"
    ]
    options = ["Not at all", "Several days", "More than half the days", "Nearly every day"]
    score_map = {"Not at all":0, "Several days":1, "More than half the days":2, "Nearly every day":3}

    # display previous chat
    for msg in st.session_state.mh_chat:
        if msg["sender"]=="bot":
            st.markdown(f'<div class="chat-bubble chat-bot">{msg["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble chat-user" style="text-align:right">{msg["text"]}</div>', unsafe_allow_html=True)

    if st.session_state.q_index >= len(questions):
        st.success("✅ Questionnaire completed.")
        total = sum(st.session_state.answers)
        if total <= 4: severity, color = "Minimal", "#16a34a"
        elif total <= 9: severity, color = "Mild", "#f59e0b"
        elif total <= 14: severity, color = "Moderate", "#f97316"
        else: severity, color = "Severe", "#ef4444"
        st.markdown(f'<div class="result-card" style="background:{color}"><h3>Questionnaire Score: {total} / 30</h3>'
                    f'<p style="opacity:0.95">Severity: <b>{severity}</b></p></div>', unsafe_allow_html=True)

        user_text = st.text_area("Free text (optional)", key="mh_freetext", height=120)

        if st.button("Get combined prediction"):
            text_conf = None
            if mental_model:
                try:
                    proba = mental_model.predict_proba([user_text.strip() or " "])[0]
                    text_conf = float(max(proba))
                except Exception as e:
                    st.warning("Mental model exists but failed to predict: " + str(e))
            q_norm = total / 30.0
            combined = 0.6*text_conf + 0.4*q_norm if text_conf else q_norm
            st.write("Combined risk score (0 low — 1 high):", f"{combined:.2f}")
            if combined > 0.6: st.error("⚠️ Elevated risk indicated. Recommend contacting a professional.")
            elif combined > 0.35: st.warning("⚠️ Mild-to-moderate risk. Consider follow-up.")
            else: st.success("🙂 Low risk (demo). Consult professional if concerned.")

        if st.button("Restart questionnaire"):
            st.session_state.mh_chat = []
            st.session_state.q_index = 0
            st.session_state.answers = []
            st.experimental_rerun()
    else:
        q = questions[st.session_state.q_index]
        if not st.session_state.mh_chat or st.session_state.mh_chat[-1]["text"]!=q:
            st.session_state.mh_chat.append({"sender":"bot","text":q})
        choice = st.radio("Select an answer:", options, key=f"q{st.session_state.q_index}")
        coln1, coln2 = st.columns([1,1])
        with coln1:
            if st.button("Submit answer"):
                st.session_state.mh_chat.append({"sender":"user","text":choice})
                st.session_state.answers.append(score_map[choice])
                st.session_state.q_index += 1
                st.experimental_rerun()
        with coln2:
            if st.button("Skip question"):
                st.session_state.mh_chat.append({"sender":"user","text":"Skipped"})
                st.session_state.answers.append(0)
                st.session_state.q_index += 1
                st.experimental_rerun()

# ---------- Brain MRI Module ----------
elif page == "Brain MRI":
    st.markdown('<div class="gradient-header"><h2>🩻 Brain MRI — Image Check</h2>'
                '<p class="small-muted">Upload an MRI (jpg/png). Demo model predicts tumor / no tumor with confidence.</p></div>', unsafe_allow_html=True)
    st.write("**Note:** Demo, not diagnostic. Always consult a radiologist.")

    uploaded_file = st.file_uploader("Upload MRI image (jpg, png)", type=["jpg","jpeg","png"])
    sample_col1, sample_col2 = st.columns([1,3])

    with sample_col1:
        if st.button("Use sample image (demo)"):
            try:
                sample_files = os.listdir(SAMPLE_DIR)
                if not sample_files:
                    st.warning("No sample images found in SAMPLE_DIR.")
                else:
                    sample_file = random.choice(sample_files)
                    uploaded_file = os.path.join(SAMPLE_DIR, sample_file)
            except Exception as e:
                st.error(f"Failed to load sample image: {e}")

    with sample_col2:
        st.markdown("Tip: crop to region of interest for best results (demo).")

    if uploaded_file:
        try:
            img = Image.open(uploaded_file)
            if img.width != img.height:
                st.warning(f"Image is not square ({img.width}x{img.height}). Resizing to 224x224.")
            img_resized = img.convert("RGB").resize((224,224))
            st.image(img_resized, caption="Uploaded image preview (resized)", use_column_width=False, width=420)

            if st.button("Run model prediction"):
                if not brain_model:
                    st.error(f"Brain model not found at `{BRAIN_MODEL_PATH}`.")
                else:
                    arr = np.expand_dims(np.array(img_resized)/255.0, axis=0)
                    pred_prob = float(brain_model.predict(arr)[0][0])
                    label = "Tumor (likely)" if pred_prob >= 0.5 else "No Tumor (likely)"
                    pct = pred_prob*100
                    if pred_prob >= 0.5:
                        st.markdown(f'<div class="result-card" style="background:#db2777"><h3>{label}</h3>'
                                    f'<p>Confidence: {pct:.1f}%</p></div>', unsafe_allow_html=True)
                        st.warning("⚠️ Demo model indicates tumor presence. Not diagnostic.")
                    else:
                        st.markdown(f'<div class="result-card" style="background:#059669"><h3>{label}</h3>'
                                    f'<p>Confidence: {100-pct:.1f}%</p></div>', unsafe_allow_html=True)
                        st.success("No tumor predicted (demo).")
        except Exception as e:
            st.error("Failed to process image: " + str(e))
    else:
        st.info("Upload an MRI image to run prediction.")

# ---------- Footer ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<small class='small-muted'>Built for learning/demo. Keep data private. Add model files to `models/`.</small>", unsafe_allow_html=True)
