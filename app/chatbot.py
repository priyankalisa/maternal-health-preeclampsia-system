import streamlit as st
import google.generativeai as genai
import time
from app.cache import get_cached_response, set_cache

# =========================
# API KEY
# =========================
API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=API_KEY)

# =========================
# MODEL (CURRENT STABLE GEMINI)
# =========================
MODEL_NAME = "gemini-1.5-flash"

model = genai.GenerativeModel(MODEL_NAME)

# =========================
# EMERGENCY DETECTION
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
# RULE-BASED ANSWERS
# =========================
def simple_medical_answers(user_input):
    text = user_input.lower()

    if "what is preeclampsia" in text:
        return (
            "Preeclampsia is a pregnancy condition with high blood pressure "
            "after 20 weeks of pregnancy. It can affect organs like liver and kidneys "
            "and needs regular medical monitoring."
        )

    if "what is hypertension" in text:
        return (
            "Hypertension means high blood pressure. During pregnancy, it must be monitored "
            "carefully to avoid complications like preeclampsia."
        )

    if "nutrition" in text:
        return "Eat iron-rich foods, fruits, vegetables, and stay hydrated during pregnancy."

    return None

# =========================
# MAIN CHAT FUNCTION
# =========================
def get_chatbot_response(user_input, chat_history=None):

    # STEP 1: CACHE CHECK
    cached = get_cached_response(user_input)
    if cached:
        return cached

    # STEP 2: RULE-BASED ANSWER
    simple_answer = simple_medical_answers(user_input)
    if simple_answer:
        set_cache(user_input, simple_answer)
        return simple_answer

    # STEP 3: CONTEXT BUILD
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

    # STEP 4: GEMINI CALL
    try:
        response = model.generate_content(prompt)

        if response and hasattr(response, "text"):
            answer = response.text

            set_cache(user_input, answer)
            return answer

        return "⚠️ No response from AI."

    except Exception as e:
        err = str(e)

        if "429" in err:
            return "⚠️ Too many requests. Please try again later."

        if "503" in err:
            return "⚠️ AI service temporarily unavailable. Try again."

        return f"⚠️ AI Error: {err}"