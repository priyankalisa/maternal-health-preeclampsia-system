"""
Maternal Health & Preeclampsia Assessment System
==================================================
A two-phase assessment tool:
- Phase 1: Maternal Health risk assessment
- Phase 2: Preeclampsia risk assessment (only if Phase 1 shows high risk)
"""

import sys
import os
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
# IMPROVEMENT #1 — Helper: risk pill HTML (icon + label)
# ============================================================================

def risk_pill(range_key):
    """Return a styled pill badge with icon for a given risk range."""
    if range_key == "low":
        return "<span style='display:inline-flex;align-items:center;gap:5px;background:#EAF3DE;color:#27500A;border:0.5px solid #C0DD97;padding:4px 12px;border-radius:99px;font-size:12px;font-weight:500;'>✅ Low risk</span>"
    elif range_key == "moderate":
        return "<span style='display:inline-flex;align-items:center;gap:5px;background:#FAEEDA;color:#633806;border:0.5px solid #FAC775;padding:4px 12px;border-radius:99px;font-size:12px;font-weight:500;'>⚠️ Moderate risk</span>"
    else:
        return "<span style='display:inline-flex;align-items:center;gap:5px;background:#FCEBEB;color:#791F1F;border:0.5px solid #F7C1C1;padding:4px 12px;border-radius:99px;font-size:12px;font-weight:500;'>🚨 High risk</span>"

# ============================================================================
# IMPROVEMENT #3 — Helper: section header HTML (pill style)
# ============================================================================

def section_header(icon, label):
    """Return a teal pill-style section header."""
    return f"""<div style='background:#E1F5EE;border-radius:8px;padding:8px 14px;
        display:inline-flex;align-items:center;gap:8px;margin-bottom:12px;'>
        <span style='font-size:16px;'>{icon}</span>
        <span style='font-size:14px;font-weight:500;color:#085041;'>{label}</span>
    </div>"""

# ============================================================================
# STREAMLIT APP SETUP
# ============================================================================

st.set_page_config(
    page_title="Maternal Health Assessment",
    page_icon="🏥",
    layout="wide"
)

# ============================================================================
# GLOBAL TEAL THEME
# ============================================================================

st.markdown("""
<style>
/* Page background */
.stApp { background-color: #F4FCF8; }

/* Sidebar background + right border */
section[data-testid="stSidebar"] {
    background-color: #E1F5EE;
    border-right: 2px solid #0F6E56;
}
section[data-testid="stSidebar"] * { color: #085041 !important; }

/* Section headers (####) left-border accent */
h4 {
    border-left: 4px solid #0F6E56;
    padding-left: 10px;
    border-radius: 0;
    color: #04342C !important;
}

/* h3 titles */
h3 { color: #085041 !important; }

/* Input fields: teal border */
input[type="number"], input[type="text"], select, textarea {
    border: 1px solid #9FE1CB !important;
    border-radius: 6px !important;
}
input[type="number"]:focus, input[type="text"]:focus, select:focus {
    border-color: #0F6E56 !important;
    box-shadow: 0 0 0 2px #9FE1CB !important;
}

/* Primary buttons — full width + taller */
div.stButton > button[kind="primary"] {
    background-color: #0F6E56 !important;
    border-color: #0F6E56 !important;
    color: #E1F5EE !important;
    width: 100% !important;
    padding: 0.65rem 1rem !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #085041 !important;
    border-color: #085041 !important;
}

/* Horizontal rules */
hr { border-color: #9FE1CB !important; }

/* Streamlit progress bar */
div[data-testid="stProgressBar"] > div > div { background-color: #0F6E56 !important; }

/* Radio buttons in sidebar */
div[data-testid="stRadio"] label { color: #085041 !important; }
</style>
""", unsafe_allow_html=True)

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
# SIDEBAR MENU
# ============================================================================

st.sidebar.markdown("### 🏥 Hospital System")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation",
    MENU_OPTIONS,
    index=st.session_state.menu_index,
    key="sidebar_menu",
    label_visibility="collapsed"
)
st.session_state.menu_index = MENU_OPTIONS.index(menu)

maternal_done  = st.session_state.maternal_result is not None
preclamp_done  = st.session_state.preeclampsia_result is not None
steps_done     = int(maternal_done) + int(preclamp_done)
steps_total    = 2
progress_pct   = int((steps_done / steps_total) * 100)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Assessment progress** — {steps_done} of {steps_total} done")
st.sidebar.progress(progress_pct)
if maternal_done:
    st.sidebar.markdown("✅ Maternal health done")
else:
    st.sidebar.markdown("⬜ Maternal health pending")
if preclamp_done:
    st.sidebar.markdown("✅ Preeclampsia done")
else:
    st.sidebar.markdown("⬜ Preeclampsia pending")

# ============================================================================
# MAIN HEADER BANNER
# ============================================================================

maternal_done_hdr  = st.session_state.maternal_result is not None
preclamp_done_hdr  = st.session_state.preeclampsia_result is not None
steps_done_hdr     = int(maternal_done_hdr) + int(preclamp_done_hdr)

import datetime
today_str = datetime.date.today().strftime("%B %Y")

st.markdown(f"""
<div style='background:var(--color-background-secondary);border-radius:var(--border-radius-lg);
     padding:1.25rem 1.75rem;display:flex;align-items:center;justify-content:space-between;
     margin-bottom:1rem;'>
  <div>
    <p style='font-size:11px;color:var(--color-text-tertiary);margin:0 0 4px;
       text-transform:uppercase;letter-spacing:0.07em;'>Maternal health system</p>
    <p style='font-size:20px;font-weight:500;margin:0;color:var(--color-text-primary);'>
      🏥 Maternal &amp; Preeclampsia Assessment
    </p>
  </div>
  <div style='display:flex;gap:8px;align-items:center;flex-wrap:wrap;'>
    <div style='background:#EAF3DE;border-radius:var(--border-radius-md);padding:5px 12px;
        font-size:12px;color:#3B6D11;font-weight:500;'>
      🤖 AI active
    </div>
    <div style='background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);
        border-radius:var(--border-radius-md);padding:5px 12px;font-size:12px;color:var(--color-text-secondary);'>
      📅 {today_str}
    </div>
    <div style='background:#E6F1FB;border-radius:var(--border-radius-md);padding:5px 12px;
        font-size:12px;color:#185FA5;font-weight:500;'>
      ✅ {steps_done_hdr}/2 steps done
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# QUICK NAV BUTTONS
# ============================================================================

phase2_eligible = (
    st.session_state.maternal_result is not None and
    should_proceed_to_phase2(st.session_state.maternal_result["probability"])
)

if phase2_eligible:
    col_quick1, col_quick2, col_quick3, col_quick4 = st.columns(4)
else:
    col_quick1, col_quick2, col_quick4 = st.columns(3)
    col_quick3 = None

with col_quick1:
    st.link_button("🏠 Home", url="https://priyankalisa.github.io/maternal-health-preeclampsia-system/", use_container_width=True)

with col_quick2:
    if st.button("📋 Step 1: Maternal Health", use_container_width=True):
        st.session_state.menu_index = MENU_OPTIONS.index("👩‍⚕️ Maternal Check")
        st.session_state.preeclampsia_result = None
        st.rerun()

if col_quick3 is not None:
    with col_quick3:
        if st.button("🫀 Step 2: Preeclampsia", use_container_width=True):
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
# STEP PROGRESS BAR
# ============================================================================

step_icons = ["📊", "👩‍⚕️", "🫀", "💬", "📈"]
step_labels = ["Dashboard", "Maternal Check", "Preeclampsia", "AI Assistant", "Analytics"]
current_idx = st.session_state.menu_index

dots_html = ""
for i in range(len(MENU_OPTIONS)):
    if i < current_idx:
        color = "#3B6D11"
    elif i == current_idx:
        color = "#185FA5"
    else:
        color = "var(--color-border-tertiary)"
    dots_html += f"<div style='flex:1;height:5px;border-radius:3px;background:{color};'></div>"

labels_html = ""
for i in range(len(MENU_OPTIONS)):
    lcolor  = "#185FA5" if i == current_idx else "var(--color-text-tertiary)"
    lweight = "500"     if i == current_idx else "400"
    labels_html += f"<span style='font-size:10px;color:{lcolor};font-weight:{lweight};'>{step_icons[i]} {step_labels[i]}</span>"

step_header = f"Step {current_idx + 1} of {len(MENU_OPTIONS)} &mdash; <b>{step_labels[current_idx]}</b>"

st.markdown(f"""
<div style='background:var(--color-background-secondary);border-radius:var(--border-radius-lg);
     padding:0.9rem 1.25rem;margin-bottom:1rem;'>
  <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;'>
    <p style='font-size:12px;color:var(--color-text-secondary);margin:0;'>Assessment steps</p>
    <p style='font-size:12px;color:var(--color-text-secondary);margin:0;'>{step_header}</p>
  </div>
  <div style='display:flex;gap:4px;margin-bottom:10px;'>{dots_html}</div>
  <div style='display:flex;justify-content:space-between;'>{labels_html}</div>
</div>
""", unsafe_allow_html=True)

col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])

with col_nav1:
    back_disabled = st.session_state.menu_index == 0
    if st.button("⬅️ Back", disabled=back_disabled):
        st.session_state.menu_index = max(0, st.session_state.menu_index - 1)
        st.rerun()

with col_nav2:
    st.markdown("")

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

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

if menu == "📊 Dashboard":
    st.title("🏥 Hospital Overview Dashboard")

    maternal_prob     = st.session_state.maternal_result["probability"] if st.session_state.maternal_result else None
    preeclampsia_prob = st.session_state.preeclampsia_result["probability"] if st.session_state.preeclampsia_result else None

    maternal_done     = maternal_prob is not None
    preeclampsia_done = preeclampsia_prob is not None
    high_risk         = maternal_done and should_proceed_to_phase2(maternal_prob)

    def _risk_colors(p):
        if p < 40:   return "#EAF3DE", "#3B6D11", "#27500A", "#C0DD97"
        elif p < 70: return "#FAEEDA", "#854F0B", "#412402", "#FAC775"
        else:        return "#FCEBEB", "#A32D2D", "#501313", "#F7C1C1"

    # ── Stat Cards ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div style='background:#E6F1FB;border-radius:10px;padding:1rem;'>
            <p style='font-size:12px;color:#185FA5;margin:0 0 6px;'>📋 Assessments done</p>
            <p style='font-size:28px;font-weight:600;margin:0;color:#042C53;'>{}</p>
        </div>""".format(int(maternal_done) + int(preeclampsia_done)), unsafe_allow_html=True)
    with col2:
        hr_bg = "#FCEBEB" if high_risk else "#EAF3DE"
        hr_tc = "#501313" if high_risk else "#173404"
        hr_sc = "#A32D2D" if high_risk else "#3B6D11"
        hr_val = "Yes ⚠️" if high_risk else "No ✅"
        st.markdown(f"""<div style='background:{hr_bg};border-radius:10px;padding:1rem;'>
            <p style='font-size:12px;color:{hr_sc};margin:0 0 6px;'>🚨 High risk</p>
            <p style='font-size:28px;font-weight:600;margin:0;color:{hr_tc};'>{hr_val}</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div style='background:#EAF3DE;border-radius:10px;padding:1rem;'>
            <p style='font-size:12px;color:#3B6D11;margin:0 0 6px;'>🤖 AI status</p>
            <p style='font-size:28px;font-weight:600;margin:0;color:#173404;'>Active</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        done_count = int(maternal_done) + int(preeclampsia_done)
        remaining  = 2 - done_count
        st.markdown(f"""<div style='background:#F1EFE8;border-radius:10px;padding:1rem;'>
            <p style='font-size:12px;color:#5F5E5A;margin:0 0 6px;'>📊 Steps remaining</p>
            <p style='font-size:28px;font-weight:600;margin:0;color:#2C2C2A;'>{remaining}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── IMPROVEMENT #6: Empty state OR Risk Summary Cards ───────────────────
    col_m, col_p = st.columns(2)

    with col_m:
        if maternal_done:
            bg, bc, tc, bar_bg = _risk_colors(maternal_prob)
            risk_label = "Low" if maternal_prob < 40 else "Moderate" if maternal_prob < 70 else "High"
            rng = "low" if maternal_prob < 40 else "moderate" if maternal_prob < 70 else "high"
            st.markdown(f"""<div style='background:{bg};border:1px solid {bc};border-radius:10px;padding:1.25rem;'>
                <p style='font-size:13px;color:{bc};margin:0 0 4px;'>🤰 Maternal Health Risk</p>
                <p style='font-size:32px;font-weight:600;margin:0 0 8px;color:{tc};'>{maternal_prob:.1f}%</p>
                <div style='height:6px;border-radius:3px;background:{bar_bg};margin-bottom:10px;'>
                    <div style='height:6px;border-radius:3px;background:{bc};width:{maternal_prob:.0f}%;'></div>
                </div>
                {risk_pill(rng)}
            </div>""", unsafe_allow_html=True)
        else:
            # IMPROVEMENT #6 — Empty state with icon + prompt
            st.markdown("""
            <div style='background:#E1F5EE;border:1px dashed #9FE1CB;border-radius:12px;
                 padding:2rem 1.5rem;text-align:center;'>
                <div style='font-size:36px;margin-bottom:10px;'>🤰</div>
                <p style='font-size:14px;font-weight:500;color:#085041;margin:0 0 6px;'>No maternal assessment yet</p>
                <p style='font-size:12px;color:#0F6E56;margin:0 0 14px;'>Complete Step 1 to see your results here</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("")
            if st.button("▶ Begin maternal assessment", use_container_width=True, type="primary"):
                st.session_state.menu_index = MENU_OPTIONS.index("👩‍⚕️ Maternal Check")
                st.rerun()

    with col_p:
        if preeclampsia_done:
            bg, bc, tc, bar_bg = _risk_colors(preeclampsia_prob)
            risk_label = "Low" if preeclampsia_prob < 40 else "Moderate" if preeclampsia_prob < 70 else "High"
            rng = "low" if preeclampsia_prob < 40 else "moderate" if preeclampsia_prob < 70 else "high"
            st.markdown(f"""<div style='background:{bg};border:1px solid {bc};border-radius:10px;padding:1.25rem;'>
                <p style='font-size:13px;color:{bc};margin:0 0 4px;'>🫀 Preeclampsia Risk</p>
                <p style='font-size:32px;font-weight:600;margin:0 0 8px;color:{tc};'>{preeclampsia_prob:.1f}%</p>
                <div style='height:6px;border-radius:3px;background:{bar_bg};margin-bottom:10px;'>
                    <div style='height:6px;border-radius:3px;background:{bc};width:{preeclampsia_prob:.0f}%;'></div>
                </div>
                {risk_pill(rng)}
            </div>""", unsafe_allow_html=True)
        else:
            # IMPROVEMENT #6 — Empty state
            pending_msg = "Complete Step 2 to see your results here" if phase2_eligible else "Available after a High Risk maternal result"
            st.markdown(f"""
            <div style='background:#F1EFE8;border:1px dashed #B4B2A9;border-radius:12px;
                 padding:2rem 1.5rem;text-align:center;'>
                <div style='font-size:36px;margin-bottom:10px;'>🫀</div>
                <p style='font-size:14px;font-weight:500;color:#444441;margin:0 0 6px;'>No preeclampsia assessment yet</p>
                <p style='font-size:12px;color:#5F5E5A;margin:0;'>{pending_msg}</p>
            </div>""", unsafe_allow_html=True)

# ============================================================================
# PAGE: MATERNAL CHECK
# ============================================================================

elif menu == "👩‍⚕️ Maternal Check":
    st.markdown("### 📋 Maternal Health Assessment Form")

    # IMPROVEMENT #3 — Pill-style section headers
    st.markdown(section_header("👤", "Basic Information"), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", 15, 60, 25)
    with c2:
        gravida = st.number_input("Gravida (pregnancies)", 0, 20, 1)
    with c3:
        titi_tika = st.number_input("TiTi Tika", 0, 10, 0)

    c4, c5, c6 = st.columns(3)
    with c4:
        gestation = st.number_input("Gestation (weeks)", 1, 42, 20)
    with c5:
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 60.0)
    with c6:
        height = st.number_input("Height (cm)", 100.0, 220.0, 160.0)

    st.markdown("---")

    st.markdown(section_header("🩺", "Clinical Indicators"), unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    with c7:
        anemia = st.selectbox("Anemia", ["None", "Minimal", "Medium"])
    with c8:
        jaundice = st.selectbox("Jaundice", ["None", "Minimal", "Medium"])
    with c9:
        albumin = st.selectbox("Albumin", ["None", "Minimal", "Medium", "Higher"])

    c10, c11, c12 = st.columns(3)
    with c10:
        fetal_pos = st.selectbox("Fetal Position", ["Normal", "Abnormal"])
    with c11:
        fetal_hb = st.number_input("Fetal Heart Beat (bpm)", 80, 200, 140)
    with c12:
        blood_sugar = st.selectbox("Blood Sugar", ["Yes", "No"])

    st.markdown("---")

    st.markdown(section_header("🧪", "Lab Tests & Blood Pressure"), unsafe_allow_html=True)
    c13, c14, c15, c16 = st.columns(4)
    with c13:
        vdrl = st.selectbox("VDRL", ["Negative", "Positive"])
    with c14:
        hrsag = st.selectbox("HRsAG", ["Negative", "Positive"])
    with c15:
        sys_bp = st.number_input("Systolic BP (mmHg)", 60, 200, 120)
    with c16:
        dia_bp = st.number_input("Diastolic BP (mmHg)", 40, 120, 80)

    st.markdown("")

    maternal_data = pd.DataFrame({
        'Age': [age], 'Gravida': [gravida], 'TiTi Tika': [titi_tika],
        'Gestation period': [gestation], 'Weight': [weight], 'Height': [height],
        'Anemia': [anemia], 'Jaundice': [jaundice], 'Fetal position': [fetal_pos],
        'Fetal heart beat': [fetal_hb], 'Albumin': [albumin], 'Blood sugar': [blood_sugar],
        'VDRL': [vdrl], 'HRsAG': [hrsag], 'Systolic_BP': [sys_bp], 'Diastolic_BP': [dia_bp]
    })

    # IMPROVEMENT #4 — Full-width primary button (handled by CSS above)
    if st.button("🔍 Assess Maternal Health", type="primary"):
        with st.spinner("Analyzing maternal health risk..."):
            model = load_model("maternal_health_model")
            prob  = model.predict_proba(maternal_data)[0][1] * 100
            st.session_state.maternal_result = {"probability": prob}

    if st.session_state.maternal_result:
        advice = get_advice("maternal_health", st.session_state.maternal_result["probability"])
        prob   = advice["probability"]

        if advice["range"] == "low":
            bg, bc, tc, bar_bg, icon = "#EAF3DE", "#3B6D11", "#173404", "#C0DD97", "✅"
        elif advice["range"] == "moderate":
            bg, bc, tc, bar_bg, icon = "#FAEEDA", "#854F0B", "#412402", "#FAC775", "⚠️"
        else:
            bg, bc, tc, bar_bg, icon = "#FCEBEB", "#A32D2D", "#501313", "#F7C1C1", "🚨"

        # IMPROVEMENT #1 — Risk pill in result card
        st.markdown(f"""<div style='background:{bg};border:1px solid {bc};border-radius:12px;padding:1.5rem;margin:1rem 0;'>
            <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;'>
                <div>
                    <p style='font-size:11px;color:{bc};margin:0 0 2px;text-transform:uppercase;letter-spacing:0.05em;'>Maternal Health Assessment</p>
                    <p style='font-size:17px;font-weight:600;margin:0;color:{tc};'>{icon} {advice.get("title", "Result")}</p>
                </div>
                <div style='text-align:right;'>
                    <p style='font-size:36px;font-weight:700;margin:0;color:{tc};'>{prob:.1f}%</p>
                    <p style='font-size:11px;color:{bc};margin:0;'>risk probability</p>
                </div>
            </div>
            <div style='height:6px;border-radius:3px;background:{bar_bg};margin-bottom:12px;'>
                <div style='height:6px;border-radius:3px;background:{bc};width:{prob:.0f}%;'></div>
            </div>
            <div style='display:flex;align-items:center;justify-content:space-between;'>
                <p style='font-size:13px;color:{tc};margin:0;flex:1;margin-right:12px;'>{advice.get("message", "")}</p>
                {risk_pill(advice["range"])}
            </div>
        </div>""", unsafe_allow_html=True)

        if "recommendations" in advice:
            st.markdown("##### 📋 Recommendations")
            for r in advice["recommendations"]:
                st.markdown(f"- {r}")

        if should_proceed_to_phase2(prob):
            st.warning("⚠️ High risk factors detected!")
            if st.button("➡️ Continue to Preeclampsia Assessment"):
                st.session_state.menu_index = MENU_OPTIONS.index("🫀 Preeclampsia Check")
                st.rerun()

# ============================================================================
# PAGE: PREECLAMPSIA CHECK
# ============================================================================

elif menu == "🫀 Preeclampsia Check":
    st.markdown("### 🫀 Preeclampsia Risk Assessment Form")

    # IMPROVEMENT #2 — Sticky Phase 1 context banner
    if st.session_state.maternal_result:
        m_prob = st.session_state.maternal_result["probability"]
        m_rng  = "low" if m_prob < 40 else "moderate" if m_prob < 70 else "high"
        if m_rng == "low":
            b_bg, b_bc, b_tc = "#EAF3DE", "#9FE1CB", "#085041"
        elif m_rng == "moderate":
            b_bg, b_bc, b_tc = "#FAEEDA", "#FAC775", "#633806"
        else:
            b_bg, b_bc, b_tc = "#FCEBEB", "#F7C1C1", "#791F1F"
        st.markdown(f"""
        <div style='background:{b_bg};border:1px solid {b_bc};border-radius:10px;
             padding:10px 16px;margin-bottom:18px;
             display:flex;align-items:center;justify-content:space-between;'>
            <span style='font-size:13px;color:{b_tc};'>
                <strong>Phase 1 result:</strong> Maternal health risk — {m_prob:.1f}%
            </span>
            {risk_pill(m_rng)}
        </div>""", unsafe_allow_html=True)

    # IMPROVEMENT #3 — Pill-style section headers
    st.markdown(section_header("👤", "Patient Profile"), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", 15, 60, 25)
    with c2:
        gravidity = st.number_input("Gravidity (pregnancies)", 0, 20, 1)
    with c3:
        gestational_age = st.number_input("Gestational Age (weeks)", 1, 42, 20)

    st.markdown("---")

    st.markdown(section_header("💉", "Vitals & Measurements"), unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        pre_preg_bmi = st.number_input("Pre-Pregnancy BMI", 10.0, 60.0, 22.0)
    with c5:
        sys_bp = st.number_input("Systolic BP (mmHg)", 60, 200, 120)
    with c6:
        dia_bp = st.number_input("Diastolic BP (mmHg)", 40, 120, 80)

    st.markdown("---")

    st.markdown(section_header("🧪", "Lab Results"), unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    with c7:
        hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 12.0)
    with c8:
        fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", 50.0, 300.0, 90.0)

    st.markdown("---")

    st.markdown(section_header("🚩", "Clinical Flags"), unsafe_allow_html=True)
    c9, c10, c11 = st.columns(3)
    with c9:
        proteinuria = st.selectbox("Proteinuria", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with c10:
        hiv_status = st.selectbox("HIV Status", [0, 1], format_func=lambda x: "Positive" if x == 1 else "Negative")
    with c11:
        anemia_status = st.selectbox("Anemia Status", ["none", "moderate", "severe"], format_func=lambda x: x.capitalize())

    st.markdown("")

    anemia_moderate = 1 if anemia_status == "moderate" else 0
    anemia_none     = 1 if anemia_status == "none"     else 0
    anemia_severe   = 1 if anemia_status == "severe"   else 0

    preeclampsia_data = pd.DataFrame({
        'Age':                    [age],
        'Gravidity':              [gravidity],
        'Gestational_Age_Weeks':  [gestational_age],
        'Pre_Pregnancy_BMI':      [pre_preg_bmi],
        'Systolic_BP':            [sys_bp],
        'Diastolic_BP':           [dia_bp],
        'Hemoglobin':             [hemoglobin],
        'Fasting_Glucose':        [fasting_glucose],
        'Proteinuria':            [proteinuria],
        'HIV_Status':             [hiv_status],
        'Anemia_Status_moderate': [anemia_moderate],
        'Anemia_Status_none':     [anemia_none],
        'Anemia_Status_severe':   [anemia_severe],
    })

    # IMPROVEMENT #4 — Full-width button (handled by CSS)
    if st.button("🔍 Assess Preeclampsia", type="primary"):
        with st.spinner("Analyzing preeclampsia risk..."):
            model = load_model("preeclampsia_model")
            prob  = model.predict_proba(preeclampsia_data)[0][1] * 100
            st.session_state.preeclampsia_result = {"probability": prob}

    if st.session_state.preeclampsia_result:
        advice = get_advice("preeclampsia", st.session_state.preeclampsia_result["probability"])
        prob   = advice["probability"]

        if advice["range"] == "low":
            bg, bc, tc, bar_bg, icon = "#EAF3DE", "#3B6D11", "#173404", "#C0DD97", "✅"
        elif advice["range"] == "moderate":
            bg, bc, tc, bar_bg, icon = "#FAEEDA", "#854F0B", "#412402", "#FAC775", "⚠️"
        else:
            bg, bc, tc, bar_bg, icon = "#FCEBEB", "#A32D2D", "#501313", "#F7C1C1", "🚨"

        # IMPROVEMENT #1 — Risk pill in result card
        st.markdown(f"""<div style='background:{bg};border:1px solid {bc};border-radius:12px;padding:1.5rem;margin:1rem 0;'>
            <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;'>
                <div>
                    <p style='font-size:11px;color:{bc};margin:0 0 2px;text-transform:uppercase;letter-spacing:0.05em;'>Preeclampsia Assessment</p>
                    <p style='font-size:17px;font-weight:600;margin:0;color:{tc};'>{icon} {advice.get("title", "Result")}</p>
                </div>
                <div style='text-align:right;'>
                    <p style='font-size:36px;font-weight:700;margin:0;color:{tc};'>{prob:.1f}%</p>
                    <p style='font-size:11px;color:{bc};margin:0;'>risk probability</p>
                </div>
            </div>
            <div style='height:6px;border-radius:3px;background:{bar_bg};margin-bottom:12px;'>
                <div style='height:6px;border-radius:3px;background:{bc};width:{prob:.0f}%;'></div>
            </div>
            <div style='display:flex;align-items:center;justify-content:space-between;'>
                <p style='font-size:13px;color:{tc};margin:0;flex:1;margin-right:12px;'>{advice.get("message", "")}</p>
                {risk_pill(advice["range"])}
            </div>
        </div>""", unsafe_allow_html=True)

        if "explanation" in advice:
            st.markdown("##### 📖 Explanation")
            st.markdown(advice["explanation"])

        if "do" in advice or "dont" in advice:
            do_col, dont_col = st.columns(2)
            if "do" in advice:
                do_items = "".join([f"<div style='display:flex;align-items:flex-start;gap:8px;margin-bottom:7px;'><div style='width:6px;height:6px;border-radius:50%;background:#3B6D11;margin-top:5px;flex-shrink:0;'></div><p style='font-size:13px;color:#27500A;margin:0;'>{d}</p></div>" for d in advice["do"]])
                with do_col:
                    st.markdown(f"""<div style='background:#EAF3DE;border:0.5px solid #3B6D11;border-radius:var(--border-radius-lg);padding:1.1rem;'>
                        <p style='font-size:13px;font-weight:500;color:#173404;margin:0 0 10px;'>✅ Do</p>
                        {do_items}
                    </div>""", unsafe_allow_html=True)
            if "dont" in advice:
                dont_items = "".join([f"<div style='display:flex;align-items:flex-start;gap:8px;margin-bottom:7px;'><div style='width:6px;height:6px;border-radius:50%;background:#A32D2D;margin-top:5px;flex-shrink:0;'></div><p style='font-size:13px;color:#791F1F;margin:0;'>{d}</p></div>" for d in advice["dont"]])
                with dont_col:
                    st.markdown(f"""<div style='background:#FCEBEB;border:0.5px solid #A32D2D;border-radius:var(--border-radius-lg);padding:1.1rem;'>
                        <p style='font-size:13px;font-weight:500;color:#501313;margin:0 0 10px;'>❌ Don't</p>
                        {dont_items}
                    </div>""", unsafe_allow_html=True)

        if "immediate_actions" in advice:
            st.markdown("##### 🚨 IMMEDIATE ACTIONS")
            for i, a in enumerate(advice["immediate_actions"], 1):
                st.markdown(f"{i}. {a}")

# ============================================================================
# PAGE: AI ASSISTANT
# ============================================================================

elif menu == "💬 AI Assistant":
    st.markdown("""
<div style='background:var(--color-background-secondary);border-radius:var(--border-radius-lg);
     padding:1.25rem 1.5rem;margin-bottom:1.25rem;display:flex;align-items:flex-start;gap:14px;'>
  <div style='width:42px;height:42px;border-radius:50%;background:#E6F1FB;
       display:flex;align-items:center;justify-content:center;flex-shrink:0;'>
    🤖
  </div>
  <div>
    <p style='font-size:15px;font-weight:500;margin:0 0 4px;color:var(--color-text-primary);'>Maternal health AI assistant</p>
    <p style='font-size:13px;color:var(--color-text-secondary);margin:0;'>
      Ask anything about pregnancy, blood pressure, preeclampsia symptoms, nutrition, or your assessment results.
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<p style='font-size:12px;color:var(--color-text-tertiary);text-transform:uppercase;letter-spacing:0.06em;margin:0 0 8px;'>Suggested questions</p>", unsafe_allow_html=True)

    suggested = [
        ("🫀", "What are early signs of preeclampsia?"),
        ("🥗", "What foods should I avoid during pregnancy?"),
        ("💧", "How do I manage high blood pressure?"),
        ("🚨", "When should I visit the doctor urgently?"),
    ]
    sq_cols = st.columns(2)
    for idx, (icon, question) in enumerate(suggested):
        with sq_cols[idx % 2]:
            if st.button(f"{icon} {question}", use_container_width=True, key=f"sq_{idx}"):
                st.session_state.chat_history.append({"role": "user", "content": question})
                try:
                    response = get_chatbot_response(question, st.session_state.chat_history)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception:
                    st.session_state.chat_history.append({"role": "assistant", "content": "⚠️ AI service error. Please try again."})
                st.rerun()

    st.markdown("---")

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

    maternal_prob     = st.session_state.maternal_result["probability"] if st.session_state.maternal_result else None
    preeclampsia_prob = st.session_state.preeclampsia_result["probability"] if st.session_state.preeclampsia_result else None

    if maternal_prob is None and preeclampsia_prob is None:
        st.warning("No assessment data available. Please complete at least one assessment first.")
    else:
        rows = []
        if maternal_prob is not None:
            risk = "Low" if maternal_prob < 40 else "Moderate" if maternal_prob < 70 else "High"
            rows.append({"Assessment": "Maternal Health", "Probability (%)": round(maternal_prob, 1), "Risk Level": risk})
        if preeclampsia_prob is not None:
            risk = "Low" if preeclampsia_prob < 40 else "Moderate" if preeclampsia_prob < 70 else "High"
            rows.append({"Assessment": "Preeclampsia", "Probability (%)": round(preeclampsia_prob, 1), "Risk Level": risk})

        df = pd.DataFrame(rows)

        # ── IMPROVEMENT #5 — Combined comparison card (shown when both done) ──
        if maternal_prob is not None and preeclampsia_prob is not None:
            st.markdown("### 🔗 Combined Assessment Summary")

            def _ac(p):
                if p < 40:   return "#EAF3DE", "#3B6D11", "#173404", "#C0DD97"
                elif p < 70: return "#FAEEDA", "#854F0B", "#412402", "#FAC775"
                else:        return "#FCEBEB", "#A32D2D", "#501313", "#F7C1C1"

            m_bg, m_bc, m_tc, m_bar = _ac(maternal_prob)
            p_bg, p_bc, p_tc, p_bar = _ac(preeclampsia_prob)
            m_rng = "low" if maternal_prob < 40 else "moderate" if maternal_prob < 70 else "high"
            p_rng = "low" if preeclampsia_prob < 40 else "moderate" if preeclampsia_prob < 70 else "high"

            # Derive combined interpretation
            if m_rng == "high" and p_rng == "high":
                interp = "⚠️ Both assessments indicate high risk. Immediate clinical review is strongly recommended."
                i_bg, i_bc = "#FCEBEB", "#A32D2D"
            elif m_rng == "high" or p_rng == "high":
                interp = "⚠️ One or more high-risk indicators detected. Close monitoring and follow-up advised."
                i_bg, i_bc = "#FAEEDA", "#854F0B"
            else:
                interp = "✅ Both assessments show low to moderate risk. Continue regular prenatal check-ups."
                i_bg, i_bc = "#EAF3DE", "#3B6D11"

            st.markdown(f"""
            <div style='background:var(--color-background-secondary);border:0.5px solid var(--color-border-tertiary);
                 border-radius:12px;padding:1.25rem;margin-bottom:1rem;'>
              <div style='display:flex;gap:12px;margin-bottom:14px;'>
                <div style='flex:1;background:{m_bg};border:1px solid {m_bc};border-radius:10px;padding:14px;text-align:center;'>
                  <p style='font-size:12px;color:{m_bc};margin:0 0 4px;'>🤰 Maternal health</p>
                  <p style='font-size:28px;font-weight:700;color:{m_tc};margin:0 0 8px;'>{maternal_prob:.1f}%</p>
                  {risk_pill(m_rng)}
                </div>
                <div style='display:flex;align-items:center;font-size:22px;color:var(--color-text-tertiary);'>→</div>
                <div style='flex:1;background:{p_bg};border:1px solid {p_bc};border-radius:10px;padding:14px;text-align:center;'>
                  <p style='font-size:12px;color:{p_bc};margin:0 0 4px;'>🫀 Preeclampsia</p>
                  <p style='font-size:28px;font-weight:700;color:{p_tc};margin:0 0 8px;'>{preeclampsia_prob:.1f}%</p>
                  {risk_pill(p_rng)}
                </div>
              </div>
              <div style='background:{i_bg};border:0.5px solid {i_bc};border-radius:8px;padding:10px 14px;'>
                <p style='font-size:13px;color:{i_bc};margin:0;'>{interp}</p>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Individual Summary Cards ─────────────────────────────────────────
        st.markdown("### 📋 Assessment Summary")

        def _ac(p):
            if p < 40:   return "#EAF3DE", "#3B6D11", "#173404", "#C0DD97", "Low"
            elif p < 70: return "#FAEEDA", "#854F0B", "#412402", "#FAC775", "Moderate"
            else:        return "#FCEBEB", "#A32D2D", "#501313", "#F7C1C1", "High"

        scols = st.columns(len(rows))
        for i, row in enumerate(rows):
            p = row["Probability (%)"]
            bg, bc, tc, bar_bg, rl = _ac(p)
            rng = "low" if p < 40 else "moderate" if p < 70 else "high"
            with scols[i]:
                st.markdown(f"""<div style='background:{bg};border:1px solid {bc};border-radius:10px;padding:1.1rem;'>
                    <p style='font-size:12px;color:{bc};margin:0 0 4px;'>{row["Assessment"]}</p>
                    <p style='font-size:30px;font-weight:700;margin:0 0 8px;color:{tc};'>{p}%</p>
                    <div style='height:5px;border-radius:3px;background:{bar_bg};margin-bottom:10px;'>
                        <div style='height:5px;border-radius:3px;background:{bc};width:{p}%;'></div>
                    </div>
                    {risk_pill(rng)}
                </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Circular Gauges ──────────────────────────────────────────────────
        st.markdown("### 🎯 Risk Probability Gauges")

        if maternal_prob is not None and preeclampsia_prob is not None:
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.plotly_chart(make_gauge("🤰 Maternal Health Risk", maternal_prob), use_container_width=True)
            with col_g2:
                st.plotly_chart(make_gauge("🫀 Preeclampsia Risk", preeclampsia_prob), use_container_width=True)
        elif maternal_prob is not None:
            col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
            with col_g2:
                st.plotly_chart(make_gauge("🤰 Maternal Health Risk", maternal_prob), use_container_width=True)
        elif preeclampsia_prob is not None:
            col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
            with col_g2:
                st.plotly_chart(make_gauge("🫀 Preeclampsia Risk", preeclampsia_prob), use_container_width=True)

        # ── Risk Level Breakdown ─────────────────────────────────────────────
        st.markdown("### 🎯 Risk Level Breakdown")
        risk_counts = df["Risk Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        rl_styles = {
            "High":     ("#FCEBEB", "#A32D2D", "#501313"),
            "Moderate": ("#FAEEDA", "#854F0B", "#412402"),
            "Low":      ("#EAF3DE", "#3B6D11", "#173404")
        }
        rl_cols = st.columns(len(risk_counts))
        for i, row in risk_counts.iterrows():
            rl = row["Risk Level"]
            bg, bc, tc = rl_styles.get(rl, ("#F1EFE8", "#5F5E5A", "#2C2C2A"))
            with rl_cols[i]:
                st.markdown(f"""<div style='background:{bg};border:1px solid {bc};border-radius:10px;padding:1rem;text-align:center;'>
                    <p style='font-size:12px;color:{bc};margin:0 0 4px;'>{rl} risk</p>
                    <p style='font-size:32px;font-weight:700;margin:0;color:{tc};'>{row["Count"]}</p>
                </div>""", unsafe_allow_html=True)
