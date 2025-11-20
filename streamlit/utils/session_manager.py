import streamlit as st


def initialize_session_state():
    if "loaded_documents" not in st.session_state:
        st.session_state.loaded_documents = []

    if "docs_processed" not in st.session_state:
        st.session_state.docs_processed = False

    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "messages" not in st.session_state:
        st.session_state.messages = []


def are_docs_ready():
    return (
        st.session_state.get("docs_processed", False)
        and st.session_state.get("total_chunks", 0) > 0
    )


def get_docs_status():
    return {
        "loaded": len(st.session_state.get("loaded_documents", [])),
        "processed": st.session_state.get("docs_processed", False),
        "chunks": st.session_state.get("total_chunks", 0),
    }


def update_docs_status(docs_processed, total_chunks):
    st.session_state.docs_processed = docs_processed
    st.session_state.total_chunks = total_chunks


def mark_docs_processed(total_chunks):
    st.session_state.docs_processed = True
    st.session_state.total_chunks = total_chunks
