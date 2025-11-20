import streamlit as st
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from utils.document_processor import retrieve_chunks, get_collection_info
from utils.prompt_handler import prepare_prompt
from utils.llm_handler import generate_answer
from utils.session_manager import (
    initialize_session_state,
    are_docs_ready,
    get_docs_status,
)

st.set_page_config(page_title="Chat", layout="wide")
st.title("💬 RAG Chatbot")
st.header("Chat with Your Documents")

initialize_session_state()

status = get_docs_status()
st.sidebar.subheader("📊 Status")
st.sidebar.metric("Documents", status["loaded"])
st.sidebar.metric("Chunks", status["chunks"])
st.sidebar.metric("Ready", "✓ Yes" if status["processed"] else "✗ No")

if not are_docs_ready():
    st.warning("⚠️ No documents processed yet!")
    st.info(
        "📖 Steps:\n1. Go to Documents page\n2. Upload files\n3. Click 'Process for RAG'\n4. Return here to chat"
    )
    st.stop()

collection_info = get_collection_info()
st.success(f"✓ {collection_info['count']} chunks indexed")

st.divider()

st.sidebar.subheader("🔑 LLM Providers")
st.sidebar.write("✅ DeepSeek (Primary)")
st.sidebar.write("✅ Groq (Free Fallback)")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            retrieved_chunks = retrieve_chunks(user_input, top_k=5)

            if not retrieved_chunks:
                response = "❌ No relevant information found in your documents."
                st.markdown(response)
                provider_used = "N/A"
            else:
                with st.spinner("Generating answer..."):
                    prompt = prepare_prompt(user_input, retrieved_chunks)
                    answer, provider_used = generate_answer(
                        prompt
                    )  # No parameters needed!

                    if answer:
                        st.success(f"✓ Using {provider_used}")
                        st.markdown(answer)
                        response = answer
                    else:
                        response = "❌ Failed to generate answer. Please try again."
                        st.error(response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "sources": retrieved_chunks,
                "provider": provider_used,
            }
        )

        if retrieved_chunks:
            with st.expander("📄 Sources Used"):
                for idx, chunk in enumerate(retrieved_chunks, 1):
                    st.markdown(
                        f"**{idx}. {chunk['source']}** (Match: {chunk['relevance']})"
                    )
                    st.markdown(f"```\n{chunk['text'][:300]}\n```")

        st.session_state.chat_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "question": user_input,
                "answer": response,
                "provider": provider_used,
                "chunks_retrieved": len(retrieved_chunks),
            }
        )

st.divider()

if st.session_state.chat_history:
    st.subheader("💾 Chat History")

    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

    for idx, chat in enumerate(st.session_state.chat_history, 1):
        provider = chat.get("provider", "Unknown")
        with st.expander(f"{idx}. {chat['timestamp'][:19]} [{provider}]"):
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"**A:** {chat['answer']}")
            st.caption(f"Sources: {chat['chunks_retrieved']}")
