from openai import OpenAI
import streamlit as st

# ⚠️ NOT SECURE - Only for testing!
DEEPSEEK_API_KEY = "sk-da97c4dbc0f84832bbffd1d5057e53c1"
GROQ_API_KEY = "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"


def generate_answer_with_deepseek(prompt, api_key=DEEPSEEK_API_KEY):
    try:
        if not api_key:
            return None

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500,
            timeout=10,
        )

        answer = response.choices[0].message.content.strip()

        sentences = answer.split(". ")
        if len(sentences) > 3:
            answer = ". ".join(sentences[:3]) + "."

        return answer

    except Exception as e:
        st.warning(f"DeepSeek failed: {str(e)}")
        return None


def generate_answer_with_groq(prompt, api_key=GROQ_API_KEY):
    """Free model fallback - Groq"""
    try:
        if not api_key:
            return None

        from groq import Groq

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500,
        )

        answer = response.choices[0].message.content.strip()

        sentences = answer.split(". ")
        if len(sentences) > 3:
            answer = ". ".join(sentences[:3]) + "."

        return answer

    except Exception as e:
        st.warning(f"Groq failed: {str(e)}")
        return None


def generate_answer(prompt, deepseek_key=DEEPSEEK_API_KEY, groq_key=GROQ_API_KEY):
    """
    Try DeepSeek first, then fallback to Groq (free)
    """

    # Try DeepSeek
    if deepseek_key:
        answer = generate_answer_with_deepseek(prompt, deepseek_key)
        if answer:
            return answer, "DeepSeek"

    # Fallback to Groq (free)
    if groq_key:
        answer = generate_answer_with_groq(prompt, groq_key)
        if answer:
            return answer, "Groq (Free)"

    # All failed
    return None, None
