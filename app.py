import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Set page config
st.set_page_config(page_title="Sciatica Detection", page_icon="🩺", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
        .header { font-size: 40px; font-weight: bold; text-align: center; color: #2E3B4E; }
        .subheader { font-size: 20px; text-align: center; color: #444; }
        .stButton>button { background-color: #4CAF50; color: white; font-size: 18px; padding: 10px; }
        .section-title { font-size: 25px; font-weight: bold; color: #2E3B4E; }
        .disclaimer { font-size: 16px; color: red; font-style: italic; }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<h1><p class="header">Sciatica Detection System 🩺</p></h1>', unsafe_allow_html=True)

# Sciatica Information
st.markdown("""#### What is Sciatica?
Sciatica is a neuropathic syndrome characterized by radicular pain extending from the lumbosacral region to the gluteal and lower extremities, resulting from compression of the lumbosacral nerve root, specifically the sciatic nerve.
Compression of the sciatic nerve root is primarily attributed to intervertebral disc degeneration or nucleus pulposus herniation, with the nerve originating from L4 to S1 and coursing distally to the foot and ankle.
""")


# Causes and Impact
st.markdown("""#### Causes of Sciatica
- Herniated or slipped disc
- Injury or trauma affecting the lower spine
- Prolonged sitting or poor posture

#### Impact of Sciatica
- Sharp pain in the lower back and legs
- Numbness or tingling sensation
- Muscle weakness in affected areas
- Difficulty in movement or standing for long periods
""")

st.markdown("---")

# Sciatica Prediction Interface
st.markdown('<h2><p class="section-title">Check for Sciatica Symptoms </p></h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    below_knee_pain = st.checkbox("🦵 Below knee pain (Yes=1)")
    neural_tension = st.checkbox("⚡ Neural tension (Yes=1)")
    slump_positive = st.checkbox("🛌 Slump positive (Yes=1)")
    crossed_slr = st.checkbox("↔️ Crossed SLR positive (Yes=1)")

with col2:
    slr_positive = st.checkbox("✅ SLR positive (Yes=1)")
    reflex_deficit = st.checkbox("🌀 Reflex deficit (Yes=1)")
    neuro_deficit = st.checkbox("🧠 Neurological deficit (Yes=1)")
    sensory_changes = st.checkbox("🔍 Subjective sensory changes (Yes=1)")

# Convert to input format
input_data = np.array([
    int(below_knee_pain), int(neural_tension), int(slump_positive),
    int(crossed_slr), int(slr_positive), int(reflex_deficit),
    int(neuro_deficit), int(sensory_changes)
]).reshape(1, -1)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Sciatica"):
    prediction = model.predict(input_data)
    result = "✅ Sciatica Detected" if prediction[0] == 1 else "❌ No Sciatica"
    st.success(f"### **{result}**")

st.markdown("---")

# Disclaimer
st.markdown('<p class="disclaimer">⚠️ Disclaimer: This model is trained on a small dataset and may not be fully accurate. Please consult a doctor for a proper diagnosis.</p>', unsafe_allow_html=True)

# Dataset Reference
st.markdown("### Reference Dataset")
if st.button("🔗 View Dataset"):
    st.markdown("[Click Here](https://www.kaggle.com/datasets/medewaramey/scitica-prediction-dataset)", unsafe_allow_html=True)
