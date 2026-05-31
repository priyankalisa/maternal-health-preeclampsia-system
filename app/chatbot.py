import os
import google.generativeai as genai

# Load API key from environment (Render + local safe)
api_key = os.getenv("GEMINI_API_KEY")

# Safety check: prevent crash if API key missing
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in environment variables")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")


# ============================================================================
# 🧠 CHATBOT RESPONSE WITH MEMORY
# ============================================================================
def get_chatbot_response(user_query, chat_history):

    # Convert chat history into readable context
    history_text = ""

    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""
You are a Maternal Health AI Assistant.

Rules:
- Answer only maternal health topics (pregnancy, BP, preeclampsia, nutrition).
- Keep answers under 150 words.
- Be simple and safe.
- If unrelated, politely refuse.
- Never provide medical diagnosis.
- Always encourage consulting a healthcare professional.

Conversation history:
{history_text}

Current user question:
{user_query}
"""

    response = model.generate_content(prompt)

    return response.text


# ============================================================================
# ⚠️ MEDICAL EMERGENCY DETECTION SYSTEM
# ============================================================================
def check_medical_emergency(text):
    text = text.lower()

    danger_keywords = [
        "severe headache",
        "blurred vision",
        "high bp",
        "very high blood pressure",
        "swelling face",
        "face swelling",
        "chest pain",
        "seizure",
        "shortness of breath",
        "breathing difficulty",
        "bleeding",
        "vaginal bleeding",
        "reduced fetal movement",
        "no fetal movement"
    ]

    return any(keyword in text for keyword in danger_keywords)