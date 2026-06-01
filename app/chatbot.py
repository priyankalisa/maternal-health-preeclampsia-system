import streamlit as st
from google import genai
import time
from cache import get_cached_response, set_cache

# =========================
# API KEY
# =========================
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

# =========================
# MODELS (SMART ORDER)
# =========================
MODEL_PRIORITY = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-pro-latest"
]

# =========================
# EMERGENCY DETECTION (IMPROVED)
# =========================
def check_medical_emergency(text: str) -> bool:
    text = text.lower()

    danger_words = [
        "bleeding", "seizure", "unconscious", "faint",
        "no movement", "fits", "convulsion",
        "severe pain", "chest pain", "emergency",
        "blurred vision", "high bp crisis"
    ]

    return any(word in text for word in danger_words)

# =========================
# SIMPLE RULE-BASED ANSWERS (NO API = SAVE QUOTA)
# =========================
def simple_medical_answers(user_input):
    text = user_input.lower()

    if "what is preeclampsia" in text:
        return (
            "Preeclampsia is a pregnancy condition with high blood pressure "
            "and possible organ damage after 20 weeks of pregnancy. "
            "It needs medical monitoring."
        )

    if "what is hypertension" in text:
        return (
            "Hypertension is high blood pressure. In pregnancy, it must be monitored "
            "to prevent complications like preeclampsia."
        )

    if "nutrition" in text:
        return "Eat iron-rich food, fruits, vegetables, and stay hydrated during pregnancy."

    return None

# =========================
# MAIN CHAT FUNCTION
# =========================
def get_chatbot_response(user_input, chat_history=None):

    # =========================
    # STEP 1: CACHE CHECK
    # =========================
    cached = get_cached_response(user_input)
    if cached:
        return cached

    # =========================
    # STEP 2: RULE-BASED ANSWER
    # =========================
    simple_answer = simple_medical_answers(user_input)
    if simple_answer:
        set_cache(user_input, simple_answer)
        return simple_answer

    # =========================
    # STEP 3: BUILD CONTEXT
    # =========================
    context = ""
    if chat_history:
        for msg in chat_history[-8:]:
            context += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""
You are a certified Maternal Health AI Assistant.

Rules:
- Give safe medical advice
- Keep answers simple and clear
- Always recommend doctor for serious symptoms
- Focus on pregnancy, BP, preeclampsia, nutrition

Conversation:
{context}

User: {user_input}
Assistant:
"""

    # =========================
    # STEP 4: GEMINI CALL (FALLBACK SYSTEM)
    # =========================
    for model in MODEL_PRIORITY:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            if response and hasattr(response, "text"):
                answer = response.text

                # save cache
                set_cache(user_input, answer)

                return answer

        except Exception as e:
            err = str(e)

            # retry logic
            if "503" in err or "UNAVAILABLE" in err:
                time.sleep(2)
                continue

            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                time.sleep(5)
                continue

            return f"⚠️ AI Error: {err}"

    return "⚠️ Service busy. Please try again later."