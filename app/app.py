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
import datetime

from chatbot import get_chatbot_response, check_medical_emergency

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG_PATH = Path(__file__).parent / "doctor_advice.json"
MODELS_PATH = Path(__file__).parent / "models"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
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
    return prob >= config["thresholds"]["maternal_health"]["low"]

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
# IMPROVEMENT — Helper: generate PDF report HTML
# ============================================================================

def generate_report_html(maternal_result, preeclampsia_result, maternal_ts, preeclampsia_ts):
    """Return an HTML string for the assessment report (used for PDF/print)."""
    import datetime as _dt

    def _risk_label(p):
        if p < 40:   return "Low Risk",   "#27500A", "#EAF3DE", "✅"
        elif p < 70: return "Moderate Risk", "#633806", "#FAEEDA", "⚠️"
        else:        return "High Risk",  "#791F1F", "#FCEBEB", "🚨"

    today = _dt.date.today().strftime("%-d %B %Y")

    sections = []

    if maternal_result:
        p = maternal_result["probability"]
        rl, tc, bg, icon = _risk_label(p)
        ts_line = f"<p style='font-size:11px;color:#888;margin:4px 0 0;'>Assessed on {maternal_ts}</p>" if maternal_ts else ""
        sections.append(f"""
        <div style='background:{bg};border:1px solid #ccc;border-radius:10px;padding:16px;margin-bottom:16px;'>
          <p style='font-size:11px;color:#888;text-transform:uppercase;margin:0 0 2px;'>Maternal Health Assessment</p>
          <p style='font-size:20px;font-weight:700;color:{tc};margin:0;'>{icon} {rl} — {p:.1f}%</p>
          {ts_line}
        </div>""")

    if preeclampsia_result:
        p = preeclampsia_result["probability"]
        rl, tc, bg, icon = _risk_label(p)
        ts_line = f"<p style='font-size:11px;color:#888;margin:4px 0 0;'>Assessed on {preeclampsia_ts}</p>" if preeclampsia_ts else ""
        sections.append(f"""
        <div style='background:{bg};border:1px solid #ccc;border-radius:10px;padding:16px;margin-bottom:16px;'>
          <p style='font-size:11px;color:#888;text-transform:uppercase;margin:0 0 2px;'>Preeclampsia Assessment</p>
          <p style='font-size:20px;font-weight:700;color:{tc};margin:0;'>{icon} {rl} — {p:.1f}%</p>
          {ts_line}
        </div>""")

    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html>
<head><meta charset='utf-8'>
<title>Maternal Health Assessment Report</title>
<style>
  body {{ font-family: Arial, sans-serif; color: #1a1a1a; padding: 40px; max-width: 700px; margin: auto; }}
  h1 {{ font-size: 22px; color: #085041; border-bottom: 2px solid #0F6E56; padding-bottom: 10px; }}
  .meta {{ font-size: 12px; color: #666; margin-bottom: 24px; }}
  .footer {{ font-size: 11px; color: #aaa; margin-top: 40px; border-top: 1px solid #eee; padding-top: 12px; }}
</style>
</head>
<body>
  <h1>🏥 Maternal & Preeclampsia Assessment Report</h1>
  <p class='meta'>Report generated: {today} &nbsp;|&nbsp; Maternal Health System</p>
  {body}
  <p class='footer'>⚠️ This report is generated by an AI-assisted screening tool and is not a substitute for professional medical advice. Please share with your healthcare provider.</p>
</body>
</html>"""



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

/* Mobile: stack columns on screens narrower than 640px */
@media (max-width: 640px) {
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
}

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

if "maternal_timestamp" not in st.session_state:
    st.session_state.maternal_timestamp = None

if "preeclampsia_timestamp" not in st.session_state:
    st.session_state.preeclampsia_timestamp = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "prefill_age" not in st.session_state:
    st.session_state.prefill_age = 25

if "prefill_gestation" not in st.session_state:
    st.session_state.prefill_gestation = 20

if "_pending_rerun" not in st.session_state:
    st.session_state._pending_rerun = False

# ============================================================================
# SIDEBAR MENU
# ============================================================================

st.sidebar.markdown("### 🏥 Hospital System")
st.sidebar.markdown("---")

# Compute phase2 eligibility before building the sidebar
# (defined here early so we can inject CSS before rendering the radio)
_maternal_for_sidebar = st.session_state.maternal_result
_phase2_eligible_sidebar = (
    _maternal_for_sidebar is not None and
    should_proceed_to_phase2(_maternal_for_sidebar["probability"])
)

# Visually disable the Preeclampsia Check radio option when not eligible
if not _phase2_eligible_sidebar:
    st.sidebar.markdown("""
<style>
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-child(3) {
    opacity: 0.4 !important;
    pointer-events: none !important;
    cursor: not-allowed !important;
}
</style>
""", unsafe_allow_html=True)

# Build filtered options: always show all, but we will block navigation to
# Preeclampsia Check via the Next button guard. For the radio we keep all
# options visible; eligibility is enforced at the button/nav level.
menu = st.sidebar.radio(
    "Navigation",
    MENU_OPTIONS,
    index=st.session_state.menu_index,
    key="sidebar_menu",
    label_visibility="collapsed"
)

# If user tries to navigate to Preeclampsia Check when not eligible, redirect
if menu == "🫀 Preeclampsia Check" and not _phase2_eligible_sidebar:
    st.session_state.menu_index = MENU_OPTIONS.index("👩‍⚕️ Maternal Check")
    st.session_state._pending_rerun = True

if st.session_state._pending_rerun:
    st.session_state._pending_rerun = False
    st.rerun()

st.session_state.menu_index = MENU_OPTIONS.index(menu)

maternal_done  = st.session_state.maternal_result is not None
preclamp_done  = st.session_state.preeclampsia_result is not None
steps_done     = int(maternal_done) + int(preclamp_done)
steps_total    = 2
progress_pct   = int((steps_done / steps_total) * 100)

st.sidebar.markdown("---")
st.sidebar.markdown("**Assessment progress**")
if maternal_done:
    st.sidebar.markdown("✅ Maternal health done")
else:
    st.sidebar.markdown("⬜ Maternal health pending")
if preclamp_done:
    st.sidebar.markdown("✅ Preeclampsia done")
else:
    st.sidebar.markdown("⬜ Preeclampsia pending")

# ── PDF / Download Report ────────────────────────────────────────────────────
if maternal_done and preclamp_done:
    st.sidebar.markdown("---")
    _report_html = generate_report_html(
        st.session_state.maternal_result,
        st.session_state.preeclampsia_result,
        st.session_state.maternal_timestamp,
        st.session_state.preeclampsia_timestamp,
    )
    st.sidebar.download_button(
        label="📄 Download Report",
        data=_report_html.encode("utf-8"),
        file_name="maternal_health_report.html",
        mime="text/html",
        use_container_width=True,
    )
    st.sidebar.caption("Opens as a printable HTML page. Use browser's Print → Save as PDF.")

# ============================================================================
# MAIN HEADER BANNER
# ============================================================================

maternal_done_hdr  = st.session_state.maternal_result is not None
preclamp_done_hdr  = st.session_state.preeclampsia_result is not None
steps_done_hdr     = int(maternal_done_hdr) + int(preclamp_done_hdr)

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

col_quick1, col_quick2, col_quick3, col_quick4 = st.columns(4)

with col_quick1:
    st.link_button("🏠 Home", url="https://priyankalisa.github.io/maternal-health-preeclampsia-system/", use_container_width=True)

with col_quick2:
    if st.button("📋 Step 1: Maternal Health", use_container_width=True):
        st.session_state.menu_index = MENU_OPTIONS.index("👩‍⚕️ Maternal Check")
        st.rerun()

with col_quick3:
    if st.button("🫀 Step 2: Preeclampsia", use_container_width=True, disabled=not phase2_eligible):
        st.session_state.menu_index = MENU_OPTIONS.index("🫀 Preeclampsia Check")
        st.rerun()

with col_quick4:
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.menu_index = 0
        st.session_state.maternal_result = None
        st.session_state.preeclampsia_result = None
        st.session_state.maternal_timestamp = None
        st.session_state.preeclampsia_timestamp = None
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
                st.warning("⚠️ Preeclampsia check is only available after a Moderate or High Risk maternal result.")
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
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", 15, 60, 25)
    with c2:
        gravida = st.number_input("Gravida (pregnancies)", 0, 20, 1)
    c3a, c3b = st.columns(2)
    with c3a:
        titi_tika = st.number_input("TiTi Tika", 0, 10, 0)
    with c3b:
        gestation = st.number_input("Gestation (weeks)", 1, 42, 20)

    c4, c5 = st.columns(2)
    with c4:
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 60.0)
    with c5:
        height = st.number_input("Height (cm)", 100.0, 220.0, 160.0)

    st.markdown("---")

    st.markdown(section_header("🩺", "Clinical Indicators"), unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    with c7:
        anemia = st.selectbox("Anemia", ["None", "Minimal", "Medium"])
    with c8:
        jaundice = st.selectbox("Jaundice", ["None", "Minimal", "Medium"])
    c9a, c9b = st.columns(2)
    with c9a:
        albumin = st.selectbox("Albumin", ["None", "Minimal", "Medium", "Higher"])
    with c9b:
        fetal_pos = st.selectbox("Fetal Position", ["Normal", "Abnormal"])

    c10, c11 = st.columns(2)
    with c10:
        fetal_hb = st.number_input("Fetal Heart Beat (bpm)", 80, 200, 140)
    with c11:
        blood_sugar = st.selectbox("Blood Sugar", ["Yes", "No"])

    st.markdown("---")

    st.markdown(section_header("🧪", "Lab Tests & Blood Pressure"), unsafe_allow_html=True)
    c13, c14 = st.columns(2)
    with c13:
        vdrl = st.selectbox("VDRL", ["Negative", "Positive"])
    with c14:
        hrsag = st.selectbox("HRsAG", ["Negative", "Positive"])
    c15, c16 = st.columns(2)
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

    # ── Input validation warnings ────────────────────────────────────────────
    _val_warnings = []
    if sys_bp > 180:
        _val_warnings.append("⚠️ Systolic BP > 180 mmHg — this is critically high. Results will be flagged accordingly.")
    if dia_bp > 120:
        _val_warnings.append("⚠️ Diastolic BP > 120 mmHg — this is critically high.")
    if sys_bp > 140 or dia_bp > 90:
        _val_warnings.append("⚠️ Blood pressure above normal pregnancy range (>140/90). Ensure readings are accurate.")
    bmi_val = weight / ((height / 100) ** 2) if height > 0 else 0
    if bmi_val < 15 or bmi_val > 50:
        _val_warnings.append(f"⚠️ BMI of {bmi_val:.1f} is outside expected range (15–50). Please check weight/height values.")
    if fetal_hb < 100 or fetal_hb > 180:
        _val_warnings.append(f"⚠️ Fetal heart beat of {fetal_hb} bpm is outside the normal range (100–180 bpm).")
    if age < 18:
        _val_warnings.append("⚠️ Patient age is under 18 — this may indicate a high-risk pregnancy.")
    for _w in _val_warnings:
        st.warning(_w)

    # IMPROVEMENT #4 — Full-width primary button (handled by CSS above)
    if st.button("🔍 Assess Maternal Health", type="primary"):
        with st.spinner("Analyzing maternal health risk..."):
            model = load_model("maternal_health_model")
            prob  = model.predict_proba(maternal_data)[0][1] * 100
            st.session_state.maternal_result = {"probability": prob, "gestation": gestation}
            st.session_state.maternal_timestamp = datetime.datetime.now().strftime("%-d %B %Y at %-I:%M %p")
            # Store shared fields so Step 2 can pre-fill them without re-entry
            st.session_state.prefill_age = age
            st.session_state.prefill_gestation = gestation
            st.rerun()

    if st.session_state.maternal_result:
        advice   = get_advice("maternal_health", st.session_state.maternal_result["probability"])
        prob     = advice["probability"]
        # Use live slider value so trimester tabs update immediately if user adjusts gestation
        gest_wks = gestation
        trimester_key = "1" if gest_wks <= 13 else "2" if gest_wks <= 27 else "3"

        if advice["range"] == "low":
            bg, bc, tc, bar_bg, icon = "#EAF3DE", "#3B6D11", "#173404", "#C0DD97", "✅"
        elif advice["range"] == "moderate":
            bg, bc, tc, bar_bg, icon = "#FAEEDA", "#854F0B", "#412402", "#FAC775", "⚠️"
        else:
            bg, bc, tc, bar_bg, icon = "#FCEBEB", "#A32D2D", "#501313", "#F7C1C1", "🚨"

        # Urgency badge colours
        urgency = advice.get("urgency", "routine")
        if urgency == "routine":
            urg_bg, urg_tc = "#EAF3DE", "#27500A"
        elif urgency == "monitor":
            urg_bg, urg_tc = "#FAEEDA", "#633806"
        else:
            urg_bg, urg_tc = "#FCEBEB", "#791F1F"

        # ── Result header card ──────────────────────────────────────────────
        st.markdown(f"""<div style='background:{bg};border:1px solid {bc};border-radius:12px;padding:1.5rem;margin:1rem 0;'>
            <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;'>
                <div>
                    <p style='font-size:11px;color:{bc};margin:0 0 2px;text-transform:uppercase;letter-spacing:0.05em;'>Maternal Health Assessment</p>
                    <p style='font-size:17px;font-weight:600;margin:0;color:{tc};'>{icon} {advice.get("title", "Result")}</p>
                    <span style='display:inline-block;margin-top:6px;background:{urg_bg};color:{urg_tc};border-radius:99px;padding:3px 10px;font-size:11px;font-weight:500;'>{advice.get("urgency_label","")}</span>
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

        # ── Timestamp + Re-assess ───────────────────────────────────────────
        _ts_col, _ra_col = st.columns([3, 1])
        with _ts_col:
            if st.session_state.maternal_timestamp:
                st.markdown(f"<span style='display:inline-flex;align-items:center;gap:5px;background:#E6F1FB;color:#0C447C;border:0.5px solid #B5D4F4;padding:3px 10px;border-radius:99px;font-size:12px;'>📅 {st.session_state.maternal_timestamp}</span>", unsafe_allow_html=True)
        with _ra_col:
            if st.button("🔁 Re-assess", key="maternal_reassess", help="Clear this result and re-run with different inputs"):
                st.session_state.confirm_maternal_reassess = True
                st.rerun()
        if st.session_state.get("confirm_maternal_reassess"):
            st.warning("This will clear your maternal health result. Are you sure?")
            _mr1, _mr2 = st.columns(2)
            with _mr1:
                if st.button("Yes, clear", key="maternal_reassess_yes", type="primary", use_container_width=True):
                    st.session_state.maternal_result = None
                    st.session_state.maternal_timestamp = None
                    st.session_state.confirm_maternal_reassess = False
                    st.rerun()
            with _mr2:
                if st.button("Cancel", key="maternal_reassess_no", use_container_width=True):
                    st.session_state.confirm_maternal_reassess = False
                    st.rerun()

        # ── Recommendations ─────────────────────────────────────────────────
        if "recommendations" in advice:
            rec_cards = "".join([
                f"""<div style='display:flex;align-items:flex-start;gap:12px;background:white;
                    border:1px solid {bc}22;border-left:3px solid {bc};border-radius:8px;
                    padding:10px 14px;margin-bottom:8px;'>
                    <span style='font-size:16px;flex-shrink:0;margin-top:1px;'>{'🏥' if i==0 else '📌' if i==1 else '⚡' if i==2 else '🛏️' if i==3 else '🥗' if i==4 else '💊'}</span>
                    <p style='font-size:13px;color:{tc};margin:0;font-weight:450;line-height:1.5;'>{r}</p>
                </div>"""
                for i, r in enumerate(advice["recommendations"])
            ])
            st.markdown(f"""
            <div style='background:{bg};border:1px solid {bc}44;border-radius:12px;padding:1.25rem;margin:1rem 0;'>
                <div style='display:flex;align-items:center;gap:8px;margin-bottom:14px;'>
                    <div style='width:32px;height:32px;border-radius:8px;background:{bc};display:flex;align-items:center;justify-content:center;font-size:16px;'>📋</div>
                    <p style='font-size:14px;font-weight:600;color:{tc};margin:0;'>Recommendations</p>
                </div>
                {rec_cards}
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Warning Signs — only for moderate/high ──────────────────────────
        if "warning_signs" in advice and advice["range"] in ("moderate", "high"):
            ws_cards = "".join([
                f"""<div style='display:flex;align-items:flex-start;gap:10px;background:#fff0f0;
                    border-radius:8px;padding:9px 13px;margin-bottom:7px;border:1px solid #ffd0d0;'>
                    <div style='min-width:6px;height:6px;border-radius:50%;background:#A32D2D;margin-top:5px;flex-shrink:0;'></div>
                    <p style='font-size:13px;color:#6B0000;margin:0;font-weight:450;'>{w}</p>
                </div>"""
                for w in advice["warning_signs"]
            ])
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#FCEBEB,#fff5f5);border:1.5px solid #F7C1C1;
                border-radius:12px;padding:1.25rem;margin-bottom:1rem;'>
                <div style='display:flex;align-items:center;gap:10px;margin-bottom:14px;'>
                    <div style='width:36px;height:36px;border-radius:10px;background:#A32D2D;
                        display:flex;align-items:center;justify-content:center;font-size:18px;'>🚨</div>
                    <div>
                        <p style='font-size:14px;font-weight:700;color:#791F1F;margin:0;'>Seek Care Immediately</p>
                        <p style='font-size:11px;color:#A32D2D;margin:0;'>If you notice any of the following signs</p>
                    </div>
                </div>
                {ws_cards}
            </div>""", unsafe_allow_html=True)

        # ── Diet & Exercise ─────────────────────────────────────────────────
        if "diet" in advice or "exercise" in advice:
            diet_col, ex_col = st.columns(2)
            if "diet" in advice:
                with diet_col:
                    eat_items = "".join([
                        f"""<div style='display:flex;align-items:flex-start;gap:8px;padding:7px 10px;
                            background:#f0faf4;border-radius:6px;margin-bottom:5px;'>
                            <span style='color:#1a7a3c;font-size:13px;flex-shrink:0;margin-top:1px;'>✅</span>
                            <p style='font-size:12px;color:#0d4a24;margin:0;line-height:1.4;'>{item}</p>
                        </div>"""
                        for item in advice["diet"]["eat"]
                    ])
                    avoid_items = "".join([
                        f"""<div style='display:flex;align-items:flex-start;gap:8px;padding:7px 10px;
                            background:#fff4f4;border-radius:6px;margin-bottom:5px;'>
                            <span style='color:#c0392b;font-size:13px;flex-shrink:0;margin-top:1px;'>🚫</span>
                            <p style='font-size:12px;color:#6B0000;margin:0;line-height:1.4;'>{item}</p>
                        </div>"""
                        for item in advice["diet"]["avoid"]
                    ])
                    st.markdown(f"""
                    <div style='background:white;border:1px solid #d4edda;border-radius:12px;overflow:hidden;height:100%;'>
                        <div style='background:linear-gradient(135deg,#0F6E56,#1a9e7a);padding:12px 16px;display:flex;align-items:center;gap:10px;'>
                            <span style='font-size:22px;'>🥗</span>
                            <p style='font-size:14px;font-weight:600;color:white;margin:0;'>Diet Guidance</p>
                        </div>
                        <div style='padding:14px;'>
                            <p style='font-size:10px;font-weight:700;color:#1a7a3c;margin:0 0 8px;
                                text-transform:uppercase;letter-spacing:0.08em;'>✦ Recommended Foods</p>
                            {eat_items}
                            <p style='font-size:10px;font-weight:700;color:#c0392b;margin:12px 0 8px;
                                text-transform:uppercase;letter-spacing:0.08em;'>✦ Foods to Avoid</p>
                            {avoid_items}
                        </div>
                    </div>""", unsafe_allow_html=True)
            if "exercise" in advice:
                with ex_col:
                    rec_items = "".join([
                        f"""<div style='display:flex;align-items:flex-start;gap:8px;padding:7px 10px;
                            background:#f0faf4;border-radius:6px;margin-bottom:5px;'>
                            <span style='color:#1a7a3c;font-size:13px;flex-shrink:0;margin-top:1px;'>✅</span>
                            <p style='font-size:12px;color:#0d4a24;margin:0;line-height:1.4;'>{item}</p>
                        </div>"""
                        for item in advice["exercise"]["recommended"]
                    ])
                    avoid_items = "".join([
                        f"""<div style='display:flex;align-items:flex-start;gap:8px;padding:7px 10px;
                            background:#fff4f4;border-radius:6px;margin-bottom:5px;'>
                            <span style='color:#c0392b;font-size:13px;flex-shrink:0;margin-top:1px;'>🚫</span>
                            <p style='font-size:12px;color:#6B0000;margin:0;line-height:1.4;'>{item}</p>
                        </div>"""
                        for item in advice["exercise"]["avoid"]
                    ])
                    st.markdown(f"""
                    <div style='background:white;border:1px solid #d4edda;border-radius:12px;overflow:hidden;height:100%;'>
                        <div style='background:linear-gradient(135deg,#185FA5,#2980d9);padding:12px 16px;display:flex;align-items:center;gap:10px;'>
                            <span style='font-size:22px;'>🏃</span>
                            <p style='font-size:14px;font-weight:600;color:white;margin:0;'>Exercise Guidance</p>
                        </div>
                        <div style='padding:14px;'>
                            <p style='font-size:10px;font-weight:700;color:#1a7a3c;margin:0 0 8px;
                                text-transform:uppercase;letter-spacing:0.08em;'>✦ Recommended</p>
                            {rec_items}
                            <p style='font-size:10px;font-weight:700;color:#c0392b;margin:12px 0 8px;
                                text-transform:uppercase;letter-spacing:0.08em;'>✦ Avoid</p>
                            {avoid_items}
                        </div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Trimester-specific advice ───────────────────────────────────────
        if "trimester_advice" in advice:
            tri_data = advice["trimester_advice"][trimester_key]
            tri_tabs = st.tabs(["🌱 1st Trimester", "🌿 2nd Trimester", "🍀 3rd Trimester"])
            for t_idx, t_key in enumerate(["1", "2", "3"]):
                t = advice["trimester_advice"][t_key]
                active = t_key == trimester_key
                with tri_tabs[t_idx]:
                    focus_bg = bg if active else "#f8fffe"
                    focus_bc = bc if active else "#c8e6de"
                    focus_tc = tc if active else "#2c5f4a"
                    badge = f"<span style='background:{bc};color:white;border-radius:99px;padding:2px 10px;font-size:10px;font-weight:700;margin-left:8px;letter-spacing:0.04em;'>CURRENT</span>" if active else ""
                    tip_items = "".join([
                        f"""<div style='display:flex;align-items:flex-start;gap:10px;
                            background:{'white' if active else '#f9f9f9'};border:1px solid {focus_bc};
                            border-left:3px solid {bc if active else '#9FE1CB'};
                            border-radius:7px;padding:9px 12px;margin-bottom:7px;'>
                            <span style='font-size:14px;flex-shrink:0;color:{bc if active else "#9FE1CB"};margin-top:1px;'>{'◆' if active else '◇'}</span>
                            <p style='font-size:13px;color:{"#1a3a2e" if active else "#4a6b5c"};margin:0;line-height:1.5;'>{tip}</p>
                        </div>"""
                        for tip in t["tips"]
                    ])
                    st.markdown(f"""
                    <div style='background:{focus_bg};border:1.5px solid {focus_bc};border-radius:12px;padding:1.25rem;margin-top:8px;'>
                        <div style='display:flex;align-items:center;gap:10px;margin-bottom:12px;'>
                            <div style='width:38px;height:38px;border-radius:10px;background:{"linear-gradient(135deg,"+bc+","+bc+"aa)" if active else "#e8f5ef"};
                                display:flex;align-items:center;justify-content:center;font-size:20px;'>
                                {'🌱' if t_key=='1' else '🌿' if t_key=='2' else '🍀'}
                            </div>
                            <div>
                                <p style='font-size:14px;font-weight:600;color:{focus_tc};margin:0;'>{t["label"]}{badge}</p>
                                <p style='font-size:11px;color:#5f8a78;margin:0;'>Focus: {t["focus"]}</p>
                            </div>
                        </div>
                        {tip_items}
                    </div>""", unsafe_allow_html=True)

        if should_proceed_to_phase2(prob):
            risk_label = "High risk" if advice["range"] == "high" else "Moderate risk"
            st.warning(f"⚠️ {risk_label} detected — Preeclampsia screening is recommended.")
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

    # Pre-fill age and gestational age from Step 1 if available
    _prefill_age = st.session_state.prefill_age
    _prefill_gest = st.session_state.prefill_gestation

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", 15, 60, _prefill_age,
                              help="Pre-filled from Step 1 — edit if needed")
    with c2:
        gravidity = st.number_input("Gravidity (pregnancies)", 0, 20, 1)
    c1b, c2b = st.columns(2)
    with c1b:
        gestational_age = st.number_input("Gestational Age (weeks)", 1, 42, _prefill_gest,
                                          help="Pre-filled from Step 1 — edit if needed")

    st.markdown("---")

    st.markdown(section_header("💉", "Vitals & Measurements"), unsafe_allow_html=True)
    c4, c5 = st.columns(2)
    with c4:
        pre_preg_bmi = st.number_input("Pre-Pregnancy BMI", 10.0, 60.0, 22.0)
    with c5:
        sys_bp = st.number_input("Systolic BP (mmHg)", 60, 200, 120)
    c6a, c6b = st.columns(2)
    with c6a:
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
    c9, c10 = st.columns(2)
    with c9:
        proteinuria = st.selectbox("Proteinuria", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with c10:
        hiv_status = st.selectbox("HIV Status", [0, 1], format_func=lambda x: "Positive" if x == 1 else "Negative")
    c11a, c11b = st.columns(2)
    with c11a:
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

    # ── Input validation warnings ────────────────────────────────────────────
    _pe_warnings = []
    if sys_bp > 180:
        _pe_warnings.append("⚠️ Systolic BP > 180 mmHg — critically high. Results will reflect this.")
    if dia_bp > 120:
        _pe_warnings.append("⚠️ Diastolic BP > 120 mmHg — critically high.")
    if sys_bp > 140 or dia_bp > 90:
        _pe_warnings.append("⚠️ BP above normal pregnancy range (>140/90). Confirm readings are accurate.")
    if pre_preg_bmi < 15 or pre_preg_bmi > 50:
        _pe_warnings.append(f"⚠️ Pre-pregnancy BMI of {pre_preg_bmi:.1f} is outside the expected range (15–50).")
    if age < 18:
        _pe_warnings.append("⚠️ Patient age is under 18 — elevated risk category.")
    for _w in _pe_warnings:
        st.warning(_w)

    # IMPROVEMENT #4 — Full-width button (handled by CSS)
    if st.button("🔍 Assess Preeclampsia", type="primary"):
        with st.spinner("Analyzing preeclampsia risk..."):
            model = load_model("preeclampsia_model")
            prob  = model.predict_proba(preeclampsia_data)[0][1] * 100
            st.session_state.preeclampsia_result = {"probability": prob, "gestation": gestational_age}
            st.session_state.preeclampsia_timestamp = datetime.datetime.now().strftime("%-d %B %Y at %-I:%M %p")
            st.rerun()

    if st.session_state.preeclampsia_result:
        advice   = get_advice("preeclampsia", st.session_state.preeclampsia_result["probability"])
        prob     = advice["probability"]
        # Use live slider value so trimester tabs update immediately if user adjusts gestational age
        gest_wks = gestational_age
        trimester_key = "1" if gest_wks <= 13 else "2" if gest_wks <= 27 else "3"

        if advice["range"] == "low":
            bg, bc, tc, bar_bg, icon = "#EAF3DE", "#3B6D11", "#173404", "#C0DD97", "✅"
        elif advice["range"] == "moderate":
            bg, bc, tc, bar_bg, icon = "#FAEEDA", "#854F0B", "#412402", "#FAC775", "⚠️"
        else:
            bg, bc, tc, bar_bg, icon = "#FCEBEB", "#A32D2D", "#501313", "#F7C1C1", "🚨"

        urgency = advice.get("urgency", "routine")
        if urgency == "routine":
            urg_bg, urg_tc = "#EAF3DE", "#27500A"
        elif urgency == "monitor":
            urg_bg, urg_tc = "#FAEEDA", "#633806"
        else:
            urg_bg, urg_tc = "#FCEBEB", "#791F1F"

        # ── Result header card ──────────────────────────────────────────────
        st.markdown(f"""<div style='background:{bg};border:1px solid {bc};border-radius:12px;padding:1.5rem;margin:1rem 0;'>
            <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;'>
                <div>
                    <p style='font-size:11px;color:{bc};margin:0 0 2px;text-transform:uppercase;letter-spacing:0.05em;'>Preeclampsia Assessment</p>
                    <p style='font-size:17px;font-weight:600;margin:0;color:{tc};'>{icon} {advice.get("title", "Result")}</p>
                    <span style='display:inline-block;margin-top:6px;background:{urg_bg};color:{urg_tc};border-radius:99px;padding:3px 10px;font-size:11px;font-weight:500;'>{advice.get("urgency_label","")}</span>
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

        # ── Timestamp + Re-assess ───────────────────────────────────────────
        _ts_col2, _ra_col2 = st.columns([3, 1])
        with _ts_col2:
            if st.session_state.preeclampsia_timestamp:
                st.markdown(f"<span style='display:inline-flex;align-items:center;gap:5px;background:#E6F1FB;color:#0C447C;border:0.5px solid #B5D4F4;padding:3px 10px;border-radius:99px;font-size:12px;'>📅 {st.session_state.preeclampsia_timestamp}</span>", unsafe_allow_html=True)
        with _ra_col2:
            if st.button("🔁 Re-assess", key="preeclampsia_reassess", help="Clear this result and re-run with different inputs"):
                st.session_state.confirm_preeclampsia_reassess = True
                st.rerun()
        if st.session_state.get("confirm_preeclampsia_reassess"):
            st.warning("This will clear your preeclampsia result. Are you sure?")
            _pr1, _pr2 = st.columns(2)
            with _pr1:
                if st.button("Yes, clear", key="preeclampsia_reassess_yes", type="primary", use_container_width=True):
                    st.session_state.preeclampsia_result = None
                    st.session_state.preeclampsia_timestamp = None
                    st.session_state.confirm_preeclampsia_reassess = False
                    st.rerun()
            with _pr2:
                if st.button("Cancel", key="preeclampsia_reassess_no", use_container_width=True):
                    st.session_state.confirm_preeclampsia_reassess = False
                    st.rerun()

        # ── Explanation ─────────────────────────────────────────────────────
        if "explanation" in advice:
            st.markdown("##### 📖 What This Means")
            st.markdown(advice["explanation"])

        st.markdown("---")

        # ── Do / Don't ──────────────────────────────────────────────────────
        if "do" in advice or "dont" in advice:
            do_col, dont_col = st.columns(2)
            if "do" in advice:
                do_items = "".join([f"<div style='display:flex;align-items:flex-start;gap:8px;padding:7px 10px;background:#f0faf4;border-radius:6px;margin-bottom:5px;'><span style='color:#1a7a3c;font-size:13px;flex-shrink:0;margin-top:1px;'>✓</span><p style='font-size:13px;color:#27500A;margin:0;line-height:1.4;'>{d}</p></div>" for d in advice["do"]])
                with do_col:
                    st.markdown(f"""<div style='background:#EAF3DE;border:0.5px solid #3B6D11;border-radius:var(--border-radius-lg);padding:1.1rem;'>
                        <p style='font-size:13px;font-weight:500;color:#173404;margin:0 0 10px;'>✅ Do</p>
                        {do_items}
                    </div>""", unsafe_allow_html=True)
            if "dont" in advice:
                dont_items = "".join([f"<div style='display:flex;align-items:flex-start;gap:8px;padding:7px 10px;background:#fff4f4;border-radius:6px;margin-bottom:5px;'><span style='color:#c0392b;font-size:13px;flex-shrink:0;margin-top:1px;'>✕</span><p style='font-size:13px;color:#791F1F;margin:0;line-height:1.4;'>{d}</p></div>" for d in advice["dont"]])
                with dont_col:
                    st.markdown(f"""<div style='background:#FCEBEB;border:0.5px solid #A32D2D;border-radius:var(--border-radius-lg);padding:1.1rem;'>
                        <p style='font-size:13px;font-weight:500;color:#501313;margin:0 0 10px;'>❌ Don't</p>
                        {dont_items}
                    </div>""", unsafe_allow_html=True)
            st.markdown("")

        # ── Diet & Exercise ─────────────────────────────────────────────────
        if "diet" in advice or "exercise" in advice:
            diet_col, ex_col = st.columns(2)
            if "diet" in advice:
                with diet_col:
                    eat_items = "".join([
                        f"""<div style='display:flex;align-items:flex-start;gap:8px;padding:7px 10px;
                            background:#f0faf4;border-radius:6px;margin-bottom:5px;'>
                            <span style='color:#1a7a3c;font-size:13px;flex-shrink:0;margin-top:1px;'>✅</span>
                            <p style='font-size:12px;color:#0d4a24;margin:0;line-height:1.4;'>{item}</p>
                        </div>"""
                        for item in advice["diet"]["eat"]
                    ])
                    avoid_items = "".join([
                        f"""<div style='display:flex;align-items:flex-start;gap:8px;padding:7px 10px;
                            background:#fff4f4;border-radius:6px;margin-bottom:5px;'>
                            <span style='color:#c0392b;font-size:13px;flex-shrink:0;margin-top:1px;'>🚫</span>
                            <p style='font-size:12px;color:#6B0000;margin:0;line-height:1.4;'>{item}</p>
                        </div>"""
                        for item in advice["diet"]["avoid"]
                    ])
                    st.markdown(f"""
                    <div style='background:white;border:1px solid #d4edda;border-radius:12px;overflow:hidden;height:100%;'>
                        <div style='background:linear-gradient(135deg,#0F6E56,#1a9e7a);padding:12px 16px;display:flex;align-items:center;gap:10px;'>
                            <span style='font-size:22px;'>🥗</span>
                            <p style='font-size:14px;font-weight:600;color:white;margin:0;'>Diet Guidance</p>
                        </div>
                        <div style='padding:14px;'>
                            <p style='font-size:10px;font-weight:700;color:#1a7a3c;margin:0 0 8px;
                                text-transform:uppercase;letter-spacing:0.08em;'>✦ Recommended Foods</p>
                            {eat_items}
                            <p style='font-size:10px;font-weight:700;color:#c0392b;margin:12px 0 8px;
                                text-transform:uppercase;letter-spacing:0.08em;'>✦ Foods to Avoid</p>
                            {avoid_items}
                        </div>
                    </div>""", unsafe_allow_html=True)
            if "exercise" in advice:
                with ex_col:
                    rec_items = "".join([
                        f"""<div style='display:flex;align-items:flex-start;gap:8px;padding:7px 10px;
                            background:#f0faf4;border-radius:6px;margin-bottom:5px;'>
                            <span style='color:#1a7a3c;font-size:13px;flex-shrink:0;margin-top:1px;'>✅</span>
                            <p style='font-size:12px;color:#0d4a24;margin:0;line-height:1.4;'>{item}</p>
                        </div>"""
                        for item in advice["exercise"]["recommended"]
                    ])
                    avoid_items = "".join([
                        f"""<div style='display:flex;align-items:flex-start;gap:8px;padding:7px 10px;
                            background:#fff4f4;border-radius:6px;margin-bottom:5px;'>
                            <span style='color:#c0392b;font-size:13px;flex-shrink:0;margin-top:1px;'>🚫</span>
                            <p style='font-size:12px;color:#6B0000;margin:0;line-height:1.4;'>{item}</p>
                        </div>"""
                        for item in advice["exercise"]["avoid"]
                    ])
                    st.markdown(f"""
                    <div style='background:white;border:1px solid #d4edda;border-radius:12px;overflow:hidden;height:100%;'>
                        <div style='background:linear-gradient(135deg,#185FA5,#2980d9);padding:12px 16px;display:flex;align-items:center;gap:10px;'>
                            <span style='font-size:22px;'>🏃</span>
                            <p style='font-size:14px;font-weight:600;color:white;margin:0;'>Exercise Guidance</p>
                        </div>
                        <div style='padding:14px;'>
                            <p style='font-size:10px;font-weight:700;color:#1a7a3c;margin:0 0 8px;
                                text-transform:uppercase;letter-spacing:0.08em;'>✦ Recommended</p>
                            {rec_items}
                            <p style='font-size:10px;font-weight:700;color:#c0392b;margin:12px 0 8px;
                                text-transform:uppercase;letter-spacing:0.08em;'>✦ Avoid</p>
                            {avoid_items}
                        </div>
                    </div>""", unsafe_allow_html=True)
            st.markdown("")

        # ── Immediate Actions ───────────────────────────────────────────────
        if "immediate_actions" in advice:
            ia_items = "".join([
                f"""<div style='display:flex;align-items:flex-start;gap:12px;background:#fff0f0;
                    border-radius:8px;padding:10px 14px;margin-bottom:8px;border:1px solid #ffd0d0;'>
                    <div style='min-width:26px;height:26px;border-radius:50%;background:#A32D2D;color:white;
                        font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:center;
                        flex-shrink:0;'>{i}</div>
                    <p style='font-size:13px;color:#6B0000;margin:0;line-height:1.5;font-weight:450;'>{a}</p>
                </div>"""
                for i, a in enumerate(advice["immediate_actions"], 1)
            ])
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#FCEBEB,#fff5f5);border:1.5px solid #F7C1C1;
                border-radius:12px;padding:1.25rem;margin-bottom:1rem;'>
                <div style='display:flex;align-items:center;gap:10px;margin-bottom:14px;'>
                    <div style='width:36px;height:36px;border-radius:10px;background:#A32D2D;
                        display:flex;align-items:center;justify-content:center;font-size:18px;'>🚨</div>
                    <div>
                        <p style='font-size:14px;font-weight:700;color:#791F1F;margin:0;'>Immediate Actions Required</p>
                        <p style='font-size:11px;color:#A32D2D;margin:0;'>Take these steps without delay</p>
                    </div>
                </div>
                {ia_items}
            </div>""", unsafe_allow_html=True)

        # ── Trimester-specific advice ───────────────────────────────────────
        if "trimester_advice" in advice:
            st.markdown("##### 🗓️ Trimester-Specific Guidance")
            tri_tabs = st.tabs(["🌱 1st Trimester", "🌿 2nd Trimester", "🍀 3rd Trimester"])
            for t_idx, t_key in enumerate(["1", "2", "3"]):
                t = advice["trimester_advice"][t_key]
                active = t_key == trimester_key
                with tri_tabs[t_idx]:
                    focus_bg = bg if active else "#f8fffe"
                    focus_bc = bc if active else "#c8e6de"
                    focus_tc = tc if active else "#2c5f4a"
                    badge = f"<span style='background:{bc};color:white;border-radius:99px;padding:2px 10px;font-size:10px;font-weight:700;margin-left:8px;letter-spacing:0.04em;'>CURRENT</span>" if active else ""
                    tip_items = "".join([
                        f"""<div style='display:flex;align-items:flex-start;gap:10px;
                            background:{'white' if active else '#f9f9f9'};border:1px solid {focus_bc};
                            border-left:3px solid {bc if active else '#9FE1CB'};
                            border-radius:7px;padding:9px 12px;margin-bottom:7px;'>
                            <span style='font-size:14px;flex-shrink:0;color:{bc if active else "#9FE1CB"};margin-top:1px;'>{'◆' if active else '◇'}</span>
                            <p style='font-size:13px;color:{"#1a3a2e" if active else "#4a6b5c"};margin:0;line-height:1.5;'>{tip}</p>
                        </div>"""
                        for tip in t["tips"]
                    ])
                    st.markdown(f"""
                    <div style='background:{focus_bg};border:1.5px solid {focus_bc};border-radius:12px;padding:1.25rem;margin-top:8px;'>
                        <div style='display:flex;align-items:center;gap:10px;margin-bottom:12px;'>
                            <div style='width:38px;height:38px;border-radius:10px;background:{"linear-gradient(135deg,"+bc+","+bc+"aa)" if active else "#e8f5ef"};
                                display:flex;align-items:center;justify-content:center;font-size:20px;'>
                                {'🌱' if t_key=='1' else '🌿' if t_key=='2' else '🍀'}
                            </div>
                            <div>
                                <p style='font-size:14px;font-weight:600;color:{focus_tc};margin:0;'>{t["label"]}{badge}</p>
                                <p style='font-size:11px;color:#5f8a78;margin:0;'>Focus: {t["focus"]}</p>
                            </div>
                        </div>
                        {tip_items}
                    </div>""", unsafe_allow_html=True)

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

    _chat_col1, _chat_col2 = st.columns([5, 1])
    with _chat_col1:
        st.markdown("---")
    with _chat_col2:
        st.markdown("")
        if st.session_state.chat_history:
            if st.button("🗑 Clear", key="clear_chat", help="Clear chat history", use_container_width=True):
                if st.session_state.get("confirm_clear_chat"):
                    st.session_state.chat_history = []
                    st.session_state.confirm_clear_chat = False
                    st.rerun()
                else:
                    st.session_state.confirm_clear_chat = True
                    st.rerun()
        if st.session_state.get("confirm_clear_chat"):
            st.warning("Clear all chat history?")
            _cc1, _cc2 = st.columns(2)
            with _cc1:
                if st.button("Yes, clear", key="confirm_yes", type="primary", use_container_width=True):
                    st.session_state.chat_history = []
                    st.session_state.confirm_clear_chat = False
                    st.rerun()
            with _cc2:
                if st.button("Cancel", key="confirm_no", use_container_width=True):
                    st.session_state.confirm_clear_chat = False
                    st.rerun()

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
        st.markdown("""
        <div style='background:#E1F5EE;border:1px dashed #9FE1CB;border-radius:12px;
             padding:3rem 1.5rem;text-align:center;margin-top:1rem;'>
            <div style='font-size:40px;margin-bottom:12px;'>📊</div>
            <p style='font-size:16px;font-weight:500;color:#085041;margin:0 0 6px;'>No assessment data yet</p>
            <p style='font-size:13px;color:#0F6E56;margin:0 0 18px;'>Complete at least one assessment to see analytics here</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("")
        _ea_c1, _ea_c2, _ea_c3 = st.columns([1,2,1])
        with _ea_c2:
            if st.button("▶ Begin maternal assessment", use_container_width=True, type="primary"):
                st.session_state.menu_index = MENU_OPTIONS.index("👩\u200d⚕️ Maternal Check")
                st.rerun()
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
