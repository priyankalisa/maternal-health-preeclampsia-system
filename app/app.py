"""
Maternal Health & Preeclampsia Assessment System
==================================================
A two-phase assessment tool:
- Phase 1: Maternal Health risk assessment
- Phase 2: Preeclampsia risk assessment (only if Phase 1 shows high risk)
"""

import streamlit as st
import pandas as pd
import json
import joblib
from pathlib import Path
from chatbot import get_chatbot_response

# ============================================================================
# CONFIGURATION
# ============================================================================
# Path to the JSON config file (doctor advice messages and thresholds)
CONFIG_PATH = Path(__file__).parent / "doctor_advice.json"
# Path to the trained ML models
MODELS_PATH = Path(__file__).parent / "models"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_config():
    """Load the doctor advice config from JSON file."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_range(prob, thresholds):
    """
    Determine the risk range based on probability and thresholds.
    Returns: 'low', 'moderate', or 'high'
    """
    if prob < thresholds["low"]:
        return "low"
    elif prob < thresholds["moderate"]:
        return "moderate"
    else:
        return "high"

def get_advice(model_name, prob):
    """
    Get the appropriate doctor advice based on model type and probability.
    Args:
        model_name: 'maternal_health' or 'preeclampsia'
        prob: probability percentage
    Returns: dict with advice content
    """
    config = load_config()
    thresholds = config["thresholds"][model_name]
    range_key = get_range(prob, thresholds)
    advice = config[model_name].get(range_key, {})
    return {**advice, "probability": prob, "range": range_key}

def should_proceed_to_phase2(prob):
    """
    Check if we should proceed to Phase 2 (preeclampsia assessment).
    Only proceeds if maternal health risk is high.
    """
    config = load_config()
    return prob >= config["thresholds"]["maternal_health"]["high"]

def load_model(name):
    """Load a trained model from the models folder."""
    return joblib.load(MODELS_PATH / f"{name}.pkl")

# ============================================================================
# STREAMLIT APP SETUP
# ============================================================================

st.set_page_config(
    page_title="Maternal Health Assessment",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state to track which phase we're in
if "phase" not in st.session_state:
    st.session_state.phase = 1
    st.session_state.maternal_result = None
    st.session_state.preeclampsia_result = None

# ============================================================================
# MAIN APP UI
# ============================================================================

st.title("🏥 Maternal & Preeclampsia Assessment System")

# Quick navigation buttons
st.markdown("---")
col_quick1, col_quick2, col_quick3, col_quick4 = st.columns(4)

with col_quick1:
    st.link_button("🏠 Home", url="https://guileless-ganache-2ac578.netlify.app/", use_container_width=True)

with col_quick2:
    if st.button("📋 Step 1: Maternal Health", use_container_width=True):
        st.session_state.phase = 1
        st.session_state.preeclampsia_result = None
        st.rerun()

with col_quick3:
    if st.button("🫀 Step 2: Preeclampsia", use_container_width=True, disabled=not st.session_state.maternal_result):
        if st.session_state.maternal_result and should_proceed_to_phase2(st.session_state.maternal_result["probability"]):
            st.session_state.phase = 2
            st.rerun()

with col_quick4:
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.phase = 1
        st.session_state.maternal_result = None
        st.session_state.preeclampsia_result = None
        st.rerun()

st.markdown("---")

# Show current step with navigation buttons
col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])

with col_nav1:
    if st.button("⬅️ Back", disabled=st.session_state.phase == 1):
        st.session_state.phase = 1
        st.session_state.preeclampsia_result = None
        st.rerun()

with col_nav2:
    st.markdown("### " + (
        "📍 Step 1: Maternal Health Assessment"
        if st.session_state.phase == 1
        else "📍 Step 2: Preeclampsia Assessment"
    ))

with col_nav3:
    if st.button("Next ➡️", disabled=st.session_state.phase == 2 or not st.session_state.maternal_result):
        if st.session_state.maternal_result and should_proceed_to_phase2(st.session_state.maternal_result["probability"]):
            st.session_state.phase = 2
            st.rerun()

# ============================================================================
# PHASE 1: MATERNAL HEALTH ASSESSMENT
# ============================================================================

if st.session_state.phase == 1:
    with st.expander("📋 Enter Maternal Health Data", expanded=True):
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic info
            age = st.number_input("Age", 15, 60, 25)
            gravida = st.number_input("Gravida (pregnancies)", 0, 20, 1)
            titi_tika = st.number_input("TiTi Tika", 0, 10, 0)
            gestation = st.number_input("Gestation (weeks)", 1, 42, 20)
            weight = st.number_input("Weight (kg)", 30.0, 200.0, 60.0)
            height = st.number_input("Height (cm)", 100.0, 220.0, 160.0)
        
        with col2:
            # Medical conditions
            anemia = st.selectbox("Anemia", ["None", "Minimal", "Medium"])
            jaundice = st.selectbox("Jaundice", ["None", "Minimal", "Medium"])
            fetal_pos = st.selectbox("Fetal Position", ["Normal", "Abnormal"])
            fetal_hb = st.number_input("Fetal Heart Beat", 80, 200, 140)
            albumin = st.selectbox("Albumin", ["None", "Minimal", "Medium","Higher"])
            blood_sugar = st.selectbox("Blood Sugar", ["Yes", "No"])

        col3, col4 = st.columns(2)
        
        with col3:
            vdrl = st.selectbox("VDRL", ["Negative", "Positive"])
            hrsag = st.selectbox("HRsAG", ["Negative", "Positive"])
        
        with col4:
            sys_bp = st.number_input("Systolic BP (mmHg)", 60, 200, 120)
            dia_bp = st.number_input("Diastolic BP (mmHg)", 40, 120, 80)

        # Create DataFrame for the model
        maternal_data = pd.DataFrame({
            'Age': [age], 'Gravida': [gravida], 'TiTi Tika': [titi_tika],
            'Gestation period': [gestation], 'Weight': [weight], 'Height': [height],
            'Anemia': [anemia], 'Jaundice': [jaundice], 'Fetal position': [fetal_pos],
            'Fetal heart beat': [fetal_hb], 'Albumin': [albumin], 'Blood sugar': [blood_sugar],
            'VDRL': [vdrl], 'HRsAG': [hrsag], 'Systolic_BP': [sys_bp], 'Diastolic_BP': [dia_bp]
        })

    # Run assessment button
    if st.button("🔍 Assess Maternal Health", type="primary"):
        with st.spinner("Analyzing maternal health risk..."):
            model = load_model("maternal_health_model")
            prob = model.predict_proba(maternal_data)[0][1] * 100
            st.session_state.maternal_result = {"probability": prob}

    # Show results if we have them
    if st.session_state.maternal_result:
        advice = get_advice("maternal_health", st.session_state.maternal_result["probability"])
        
        # Color code: green=low, orange=moderate, red=high
        color = "green" if advice["range"] == "low" else "orange" if advice["range"] == "moderate" else "red"
        
        st.markdown(f"## {advice.get('title', 'Result')}")
        st.markdown(f"### Probability: <span style='color:{color}'>{advice['probability']:.1f}%</span>", unsafe_allow_html=True)
        
        if "message" in advice:
            st.markdown(f"**{advice['message']}**")
        
        if "recommendations" in advice:
            st.markdown("### 📋 Recommendations")
            for r in advice["recommendations"]:
                st.markdown(f"- {r}")
        
        # If high risk, show button to proceed to Phase 2
        if should_proceed_to_phase2(advice["probability"]):
            st.warning("⚠️ High risk factors detected!")
            if st.button("➡️ Continue to Preeclampsia Assessment"):
                st.session_state.phase = 2
                st.rerun()

# ============================================================================
# PHASE 2: PREECLAMPSIA ASSESSMENT
# ============================================================================

elif st.session_state.phase == 2:
    with st.expander("🫀 Enter Preeclampsia Data", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", 15, 60, 25)
            sys_bp = st.number_input("Systolic BP (mmHg)", 60, 200, 120)
            dia_bp = st.number_input("Diastolic BP (mmHg)", 40, 120, 80)
            bs = st.number_input("Blood Sugar (mmol/L)", 3.0, 20.0, 7.0)
            temp = st.number_input("Body Temperature (°F)", 95.0, 105.0, 98.6)
        
        with col2:
            bmi = st.number_input("BMI", 15.0, 50.0, 22.0)
            prev_comp = st.selectbox("Previous Complications", [0, 1])
            pre_diab = st.selectbox("Preexisting Diabetes", [0, 1])
            gest_diab = st.selectbox("Gestational Diabetes", [0, 1])
            mental = st.selectbox("Mental Health Issues", [0, 1])
            hr = st.number_input("Heart Rate (bpm)", 50, 150, 72)

        preeclampsia_data = pd.DataFrame({
            'Age': [age], 'Systolic BP': [sys_bp], 'Diastolic': [dia_bp],
            'BS': [bs], 'Body Temp': [temp], 'BMI': [bmi],
            'Previous Complications': [prev_comp], 'Preexisting Diabetes': [pre_diab],
            'Gestational Diabetes': [gest_diab], 'Mental Health': [mental], 'Heart Rate': [hr]
        })

    if st.button("🔍 Assess Preeclampsia", type="primary"):
        with st.spinner("Analyzing preeclampsia risk..."):
            model = load_model("preeclampsia_model")
            prob = model.predict_proba(preeclampsia_data)[0][1] * 100
            st.session_state.preeclampsia_result = {"probability": prob}

    if st.session_state.preeclampsia_result:
        advice = get_advice("preeclampsia", st.session_state.preeclampsia_result["probability"])
        color = "green" if advice["range"] == "low" else "orange" if advice["range"] == "moderate" else "red"
        
        st.markdown(f"## {advice.get('title', 'Result')}")
        st.markdown(f"### Probability: <span style='color:{color}'>{advice['probability']:.1f}%</span>", unsafe_allow_html=True)
        
        if "message" in advice:
            st.markdown(f"**{advice['message']}**")
        
        if "explanation" in advice:
            st.markdown("### 📖 Explanation")
            st.markdown(advice["explanation"])
        
        if "do" in advice:
            st.markdown("### ✅ DO")
            for d in advice["do"]:
                st.markdown(f"- {d}")
        
        if "dont" in advice:
            st.markdown("### ❌ DON'T")
            for d in advice["dont"]:
                st.markdown(f"- {d}")
        
        if "immediate_actions" in advice:
            st.markdown("### 🚨 IMMEDIATE ACTIONS")
            for i, a in enumerate(advice["immediate_actions"], 1):
                st.markdown(f"{i}. {a}")
               
import streamlit as st
from chatbot import get_chatbot_response, check_medical_emergency

# ============================================================================
# MUST BE FIRST STREAMLIT COMMAND
# ============================================================================
st.set_page_config(page_title="Maternal Health AI", page_icon="🤰", layout="wide")

# ============================================================================
# SIDEBAR (Dashboard)
# ============================================================================
with st.sidebar:
    st.title("🤰 Health Dashboard")

    st.markdown("### ⚠️ Emergency Symptoms")
    st.write("""
    - Severe headache  
    - Blurred vision  
    - High BP  
    - Chest pain  
    - Reduced fetal movement  
    """)

    st.markdown("---")

    st.markdown("### 💡 Pregnancy Tips")
    st.write("""
    - Stay hydrated  
    - Regular BP monitoring  
    - Balanced diet  
    - Regular checkups  
    """)

    st.markdown("---")
    st.info("⚠️ This AI is not a medical diagnosis tool.")

# ============================================================================
# MAIN TITLE
# ============================================================================
st.markdown("<h1 style='text-align:center;'>🤖 Maternal Health AI Assistant</h1>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# CHAT STATE
# ============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []

# ============================================================================
# DISPLAY CHAT HISTORY
# ============================================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================================
# CHAT INPUT
# ============================================================================
user_question = st.chat_input("Ask about pregnancy, BP, preeclampsia...")

if user_question:

    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.markdown(user_question)

    # Emergency check
    is_emergency = check_medical_emergency(user_question)

    if is_emergency:
        st.error("⚠️ Emergency detected! Please consult a doctor immediately.")

    # AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing health query... 🤔"):
            try:
                answer = get_chatbot_response(user_question, st.session_state.messages)
            except Exception:
                answer = "Sorry, something went wrong. Please try again."
                st.error(answer)

            st.markdown(answer)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})