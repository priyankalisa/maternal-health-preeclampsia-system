import os
import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")


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
- If unrelated, refuse politely.
- Never diagnose.

Conversation history:
{history_text}

Current user question:
{user_query}
"""

    response = model.generate_content(prompt)

    return response.text