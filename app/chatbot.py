import os
import time

import google.generativeai as genai
from dotenv import load_dotenv

from cache import get_cached_response, set_cache

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# =========================
# GEMINI CONFIG
# =========================
MODEL_NAME = "gemini-2.5-flash-lite"

model = None

if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

# =========================
# EMERGENCY DETECTION
# =========================
def check_medical_emergency(text: str) -> bool:
    text = text.lower()

    danger_words = [
        "bleeding",
        "seizure",
        "unconscious",
        "faint",
        "no movement",
        "fits",
        "convulsion",
        "severe pain",
        "chest pain",
        "emergency",
        "blurred vision",
        "high bp crisis"
    ]

    return any(word in text for word in danger_words)

# =========================
# RULE-BASED RESPONSES
# =========================
def simple_medical_answers(user_input):
    text = user_input.lower()

    if "what is preeclampsia" in text:
        return (
            "Preeclampsia is a pregnancy condition characterized by high blood pressure "
            "after 20 weeks of pregnancy. It may affect organs such as the liver and kidneys "
            "and requires regular medical monitoring."
        )

    if "what is hypertension" in text:
        return (
            "Hypertension means high blood pressure. During pregnancy, it should be monitored "
            "carefully because it may increase the risk of complications such as preeclampsia."
        )

    if "nutrition" in text:
        return (
            "During pregnancy, eat iron-rich foods, fruits, vegetables, whole grains, "
            "protein-rich foods, and drink plenty of water."
        )

    return None

# =========================
# CHATBOT RESPONSE
# =========================
def get_chatbot_response(user_input, chat_history=None):

    # Step 1: Check cache
    cached_response = get_cached_response(user_input)
    if cached_response:
        return cached_response

    # Step 2: Rule-based answers
    simple_answer = simple_medical_answers(user_input)
    if simple_answer:
        set_cache(user_input, simple_answer)
        return simple_answer

    # Step 3: Build conversation context
    context = ""

    if chat_history:
        for msg in chat_history[-8:]:
            context += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""
You are a Maternal Health AI Assistant.

Guidelines:
- Give safe and medically responsible information.
- Keep answers simple and easy to understand.
- Focus on pregnancy, maternal health, blood pressure, nutrition, and preeclampsia.
- Never provide a diagnosis.
- Recommend consulting a healthcare professional for concerning symptoms.

Conversation:
{context}

User: {user_input}

Assistant:
"""

    # Step 4: Gemini response
    if not API_KEY or model is None:
        return "⚠️ GEMINI_API_KEY is not configured. Please add it to your .env file."

    try:
        response = model.generate_content(prompt)

        if response and hasattr(response, "text") and response.text:
            answer = response.text.strip()

            set_cache(user_input, answer)

            return answer

        return "⚠️ No response received from the AI service."

    except Exception as e:
        error_message = str(e)

        if "429" in error_message:
            return "⚠️ Too many requests. Please try again later."

        if "503" in error_message:
            return "⚠️ AI service is temporarily unavailable. Please try again."

        return f"⚠️ AI Error: {error_message}"