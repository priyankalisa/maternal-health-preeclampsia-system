import streamlit as st
import google.generativeai as genai

genai.configure(
    api_key=st.secrets["GEMINI_API_KEY"]
)

model = genai.GenerativeModel("gemini-2.5-flash")


def get_chatbot_response(user_query):

    prompt = f"""
You are a Maternal Health AI Assistant.

Rules:
- Answer only maternal health, pregnancy, prenatal care,
  blood pressure, nutrition, and preeclampsia questions.
- Use simple and easy language.
- Keep answers under 150 words.
- Never provide a medical diagnosis.
- Encourage consultation with healthcare professionals.
- If the question is unrelated to maternal health,
  politely refuse.

Question:
{user_query}
"""

    response = model.generate_content(prompt)

    return response.text