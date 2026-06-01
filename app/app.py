"""
Maternal Health & Preeclampsia Assessment System
==================================================
A two-phase assessment tool:
- Phase 1: Maternal Health risk assessment
- Phase 2: Preeclampsia risk assessment (only if Phase 1 shows high risk)
"""

import sys
import os

# Ensure the app directory is in the path for Render deployment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import json
import joblib
import plotly.graph_objects as go
from pathlib import Path

from chatbot import get_chatbot_response, check_medical_emergency

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG_PATH = Path(__file__).parent / "doctor_advice.json"
MODELS_PATH = Path(__file__).parent / "models"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_range(prob, thresholds):
    if prob < thresholds["low"]:
        return "low"
    elif prob < thresholds["moderate"]:
        return "moderate"
    else:
        return "high"

def get_advice(model_name, prob):
    config = load_config()
    thresholds = config["thresholds"][model_name]
    range_key = get_range(prob, thresholds)
    advice = config[model_name].get(range_key, {})
    return {**advice, "probability": prob, "range": range_key}

def should_proceed_to_phase2(prob):
    config = load_config()
    return prob >= config["thresholds"]["maternal_health"]["high"]

def load_model(name):
    return joblib.load(MODELS_PATH / f"{name}.pkl")

def make_gauge(title, prob):
    """Create a circular gauge (speedometer) chart for a given probability."""
    if prob < 40:
        color = "#2ecc71"
        risk_label = "Low Risk"
    elif prob < 70:
        color = "#f39c12"
        risk_label = "Moderate Risk"
    else:
        color = "#e74c3c"
        risk_label = "High Risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob,
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        title={"text": f"<b>{title}</b><br><span style='font-size:14px;color:{color}'>{risk_label}</span>", "font": {"size": 18}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "gray",
                "tickvals": [0, 20, 40, 60, 80, 100],
                "ticktext": ["0", "20", "40", "60", "80", "100"]
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "lightgray",
            "steps": [
                {"range": [0, 40],   "color": "#d5f5e3"},
                {"range": [40, 70],  "color": "#fdebd0"},
                {"range": [70, 100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8,
                "value": prob
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin={"t": 80, "b": 20, "l": 30, "r": 30},
        paper_bgcolor="white",
        font={"family": "Arial"}
    )
    return fig

# ============================================================================
# STREAMLIT APP SETUP
# ============================================================================

st.set_page_config(
    page_title="Maternal Health Assessment",
    page_icon="🏥",
    layout="wide"
)

# ============================================================================
# SESSION STATE INIT
# ============================================================================

MENU_OPTIONS = [
    "📊 Dashboard",
    "👩‍⚕️ Maternal Check",
    "🫀 Preeclampsia Check",
    "💬 AI Assistant",
    "📈 Analytics"
]

if "menu_index" not in st.session_state:
    st.session_state.menu_index = 0

if "maternal_result" not in st.session_state:
    st.session_state.maternal_result = None

if "preeclampsia_result" not in st.session_state:
    st.session_state.preeclampsia_result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================================
# SIDEBAR MENU — driven by session state index so buttons can change it
# ============================================================================

menu = st.sidebar.selectbox(
    "🏥 Hospital System",
    MENU_OPTIONS,
    index=st.session_state.menu_index,
    key="sidebar_menu"
)
st.session_state.menu_index = MENU_OPTIONS.index(menu)

# ============================================================================
# MAIN TITLE
# ============================================================================

st.title("🏥 Maternal & Preeclampsia Assessment System")
st.markdown("---")

# ============================================================================
# QUICK NAV BUTTONS
# ============================================================================

col_quick1, col_quick2, col_quick3, col_quick4 = st.columns(4)

with col_quick1:
    st.link_button("🏠 Home", url="https://priyankalisa.github.io/maternal-health-preeclampsia-system/", use_container_width=True)

with col_quick2:
    if st.button("📋 Step 1: Maternal Health", use_container_width=True):
        st.session_state.menu_index = MENU_OPTIONS.index("👩‍⚕️ Maternal Check")
        st.session_state.preeclampsia_result = None
        st.rerun()

with col_quick3:
    phase2_disabled = not (
        st.session_state.maternal_result and
        should_proceed_to_phase2(st.session_state.maternal_result["probability"])
    )
    if st.button("🫀 Step 2: Preeclampsia", use_container_width=True, disabled=phase2_disabled):
        st.session_state.menu_index = MENU_OPTIONS.index("🫀 Preeclampsia Check")
        st.rerun()

with col_quick4:
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.menu_index = 0
        st.session_state.maternal_result = None
        st.session_state.preeclampsia_result = None
        st.rerun()

st.markdown("---")

# ============================================================================
# BACK / NEXT NAV BAR
# ============================================================================

col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])

with col_nav1:
    back_disabled = st.session_state.menu_index == 0
    if st.button("⬅️ Back", disabled=back_disabled):
        st.session_state.menu_index = max(0, st.session_state.menu_index - 1)
        st.rerun()

with col_nav2:
    st.markdown(f"### 📍 {MENU_OPTIONS[st.session_state.menu_index]}")

with col_nav3:
    next_disabled = st.session_state.menu_index >= len(MENU_OPTIONS) - 1
    if st.button("Next ➡️", disabled=next_disabled):
        next_index = st.session_state.menu_index + 1
        if MENU_OPTIONS[next_index] == "🫀 Preeclampsia Check":
            if st.session_state.maternal_result and should_proceed_to_phase2(st.session_state.maternal_result["probability"]):
                st.session_state.menu_index = next_index
                st.rerun()
            else:
                st.warning("⚠️ Preeclampsia check is only available after a High Risk maternal result.")
        else:
            st.session_state.menu_index = next_index
            st.rerun()

st.markdown("---")

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

if menu == "📊 Dashboard":
    st.title("🏥 Hospital Overview Dashboard")

    maternal_prob = st.session_state.maternal_result["probability"] if st.session_state.maternal_result else None
    preeclampsia_prob = st.session_state.preeclampsia_result["probability"] if st.session_state.preeclampsia_result else None

    maternal_done = maternal_prob is not None
    preeclampsia_done = preeclampsia_prob is not None
    high_risk = maternal_done and should_proceed_to_phase2(maternal_prob)

    col1, col2, col3 = st.columns(3)
    col1.metric("Assessments Done", int(maternal_done) + int(preeclampsia_done))
    col2.metric("High Risk", "Yes ⚠️" if high_risk else "No ✅")
    col3.metric("AI Status", "Active ✅")

    st.markdown("---")

    if maternal_done:
        color = "green" if maternal_prob < 40 else "orange" if maternal_prob < 70 else "red"
        st.markdown(f"**🤰 Maternal Health Risk:** <span style='color:{color}'>{maternal_prob:.1f}%</span>", unsafe_allow_html=True)
    else:
        st.info("🤰 Maternal Health assessment not done yet.")

    if preeclampsia_done:
        color = "green" if preeclampsia_prob < 40 else "orange" if preeclampsia_prob < 70 else "red"
        st.markdown(f"**🫀 Preeclampsia Risk:** <span style='color:{color}'>{preeclampsia_prob:.1f}%</span>", unsafe_allow_html=True)
    else:
        st.info("🫀 Preeclampsia assessment not done yet.")

# ============================================================================
# PAGE: MATERNAL CHECK
# ============================================================================

elif menu == "👩‍⚕️ Maternal Check":
    with st.expander("📋 Enter Maternal Health Data", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 15, 60, 25)
            gravida = st.number_input("Gravida (pregnancies)", 0, 20, 1)
            titi_tika = st.number_input("TiTi Tika", 0, 10, 0)
            gestation = st.number_input("Gestation (weeks)", 1, 42, 20)
            weight = st.number_input("Weight (kg)", 30.0, 200.0, 60.0)
            height = st.number_input("Height (cm)", 100.0, 220.0, 160.0)

        with col2:
            anemia = st.selectbox("Anemia", ["None", "Minimal", "Medium"])
            jaundice = st.selectbox("Jaundice", ["None", "Minimal", "Medium"])
            fetal_pos = st.selectbox("Fetal Position", ["Normal", "Abnormal"])
            fetal_hb = st.number_input("Fetal Heart Beat", 80, 200, 140)
            albumin = st.selectbox("Albumin", ["None", "Minimal", "Medium", "Higher"])
            blood_sugar = st.selectbox("Blood Sugar", ["Yes", "No"])

        col3, col4 = st.columns(2)

        with col3:
            vdrl = st.selectbox("VDRL", ["Negative", "Positive"])
            hrsag = st.selectbox("HRsAG", ["Negative", "Positive"])

        with col4:
            sys_bp = st.number_input("Systolic BP (mmHg)", 60, 200, 120)
            dia_bp = st.number_input("Diastolic BP (mmHg)", 40, 120, 80)

        maternal_data = pd.DataFrame({
            'Age': [age], 'Gravida': [gravida], 'TiTi Tika': [titi_tika],
            'Gestation period': [gestation], 'Weight': [weight], 'Height': [height],
            'Anemia': [anemia], 'Jaundice': [jaundice], 'Fetal position': [fetal_pos],
            'Fetal heart beat': [fetal_hb], 'Albumin': [albumin], 'Blood sugar': [blood_sugar],
            'VDRL': [vdrl], 'HRsAG': [hrsag], 'Systolic_BP': [sys_bp], 'Diastolic_BP': [dia_bp]
        })

    if st.button("🔍 Assess Maternal Health", type="primary"):
        with st.spinner("Analyzing maternal health risk..."):
            model = load_model("maternal_health_model")
            prob = model.predict_proba(maternal_data)[0][1] * 100
            st.session_state.maternal_result = {"probability": prob}

    if st.session_state.maternal_result:
        advice = get_advice("maternal_health", st.session_state.maternal_result["probability"])
        color = "green" if advice["range"] == "low" else "orange" if advice["range"] == "moderate" else "red"

        st.markdown(f"## {advice.get('title', 'Result')}")
        st.markdown(f"### Probability: <span style='color:{color}'>{advice['probability']:.1f}%</span>", unsafe_allow_html=True)

        if "message" in advice:
            st.markdown(f"**{advice['message']}**")

        if "recommendations" in advice:
            st.markdown("### 📋 Recommendations")
            for r in advice["recommendations"]:
                st.markdown(f"- {r}")

        if should_proceed_to_phase2(advice["probability"]):
            st.warning("⚠️ High risk factors detected!")
            if st.button("➡️ Continue to Preeclampsia Assessment"):
                st.session_state.menu_index = MENU_OPTIONS.index("🫀 Preeclampsia Check")
                st.rerun()

# ============================================================================
# PAGE: PREECLAMPSIA CHECK
# ============================================================================

elif menu == "🫀 Preeclampsia Check":
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

# ============================================================================
# PAGE: AI ASSISTANT
# ============================================================================

elif menu == "💬 AI Assistant":
    st.subheader("💬 Ask Maternal Health Assistant")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about pregnancy, BP, preeclampsia, nutrition...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        if check_medical_emergency(user_input):
            emergency_reply = "⚠️ Emergency detected. Please contact a doctor immediately."
            st.session_state.chat_history.append({"role": "assistant", "content": emergency_reply})
            with st.chat_message("assistant"):
                st.error(emergency_reply)
        else:
            try:
                response = get_chatbot_response(user_input, st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            except Exception:
                error_msg = "⚠️ AI service error. Please try again."
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)

# ============================================================================
# PAGE: ANALYTICS
# ============================================================================

elif menu == "📈 Analytics":
    st.title("📊 Risk Analytics")

    maternal_prob = st.session_state.maternal_result["probability"] if st.session_state.maternal_result else None
    preeclampsia_prob = st.session_state.preeclampsia_result["probability"] if st.session_state.preeclampsia_result else None

    if maternal_prob is None and preeclampsia_prob is None:
        st.warning("No assessment data available. Please complete at least one assessment first.")
    else:
        # ── Summary Table ──────────────────────────────────────────────────
        rows = []
        if maternal_prob is not None:
            risk = "Low" if maternal_prob < 40 else "Moderate" if maternal_prob < 70 else "High"
            rows.append({"Assessment": "Maternal Health", "Probability (%)": round(maternal_prob, 1), "Risk Level": risk})
        if preeclampsia_prob is not None:
            risk = "Low" if preeclampsia_prob < 40 else "Moderate" if preeclampsia_prob < 70 else "High"
            rows.append({"Assessment": "Preeclampsia", "Probability (%)": round(preeclampsia_prob, 1), "Risk Level": risk})

        df = pd.DataFrame(rows)

        st.markdown("### 📋 Assessment Summary")
        st.dataframe(df, use_container_width=True)

        # ── Circular Gauges ────────────────────────────────────────────────
        st.markdown("### 🎯 Risk Probability Gauges")

        if maternal_prob is not None and preeclampsia_prob is not None:
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.plotly_chart(
                    make_gauge("🤰 Maternal Health Risk", maternal_prob),
                    use_container_width=True
                )
            with col_g2:
                st.plotly_chart(
                    make_gauge("🫀 Preeclampsia Risk", preeclampsia_prob),
                    use_container_width=True
                )
        elif maternal_prob is not None:
            col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
            with col_g2:
                st.plotly_chart(
                    make_gauge("🤰 Maternal Health Risk", maternal_prob),
                    use_container_width=True
                )
        elif preeclampsia_prob is not None:
            col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
            with col_g2:
                st.plotly_chart(
                    make_gauge("🫀 Preeclampsia Risk", preeclampsia_prob),
                    use_container_width=True
                )

        # ── Risk Level Breakdown ───────────────────────────────────────────
        st.markdown("### 🎯 Risk Level Breakdown")
        risk_counts = df["Risk Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        st.dataframe(risk_counts, use_container_width=True)
