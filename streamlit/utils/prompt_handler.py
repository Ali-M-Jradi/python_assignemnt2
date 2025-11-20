import streamlit as st


def prepare_prompt(query, retrieved_chunks):
    context = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        context += f"\n[Source {i}: {chunk['source']}]\n{chunk['text']}\n"

    prompt = f"""Based ONLY on the provided documents, answer this question directly and concisely.

INSTRUCTIONS:
- Answer the question directly (yes/no/number/name first)
- Be specific and factual
- Keep answer to 1-2 sentences MAX
- If information is not in documents, say "Not found in documents"
- Do NOT provide extra context unless asked
- Do NOT list all available information

QUESTION: {query}

CONTEXT FROM DOCUMENTS:
{context}

ANSWER:"""

    return prompt


def format_answer_with_sources(answer, retrieved_chunks):
    formatted = f"""{answer}

---
**📄 Sources:** {', '.join(set([c['source'] for c in retrieved_chunks]))}"""

    return formatted
