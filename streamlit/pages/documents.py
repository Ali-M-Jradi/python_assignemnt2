import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import os
import csv
import json
import tempfile
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from utils.document_processor import process_and_store_documents, clear_vector_db
from utils.session_manager import (
    initialize_session_state,
    get_docs_status,
    mark_docs_processed,
)

st.set_page_config(page_title="Documents", layout="wide")
st.title("📄 Documents")
st.header("Upload & Process Documents")

initialize_session_state()

SUPPORTED_FORMATS = {".pdf", ".docx", ".txt", ".csv", ".json"}
MAX_FILE_SIZE_MB = 50


def extract_text_from_file(file_path, file_ext):
    try:
        if file_ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        elif file_ext == ".pdf":
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        elif file_ext == ".docx":
            doc = Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs)

        elif file_ext == ".csv":
            rows = []
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    rows.append(dict(zip(header, row)) if header else row)
            return json.dumps(rows, indent=2, ensure_ascii=False)

        elif file_ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                return json.dumps(json.load(f), indent=2, ensure_ascii=False)

        return ""

    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""


def display_document(doc, index):
    """Display document with unique key"""
    with st.expander(f"📄 {doc['name']} ({doc['type'].upper()})"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Size", f"{doc['size_mb']} MB")
        col2.metric("Characters", f"{doc['char_count']:,}")
        col3.metric("Words", f"{doc['word_count']:,}")

        # ✅ Add unique key based on index
        st.text_area(
            "Preview",
            value=doc["content"][:500] + "...",
            height=150,
            disabled=True,
            key=f"preview_{index}",  # ✅ UNIQUE KEY
        )


def clear_all():
    st.session_state.loaded_documents = []
    st.session_state.docs_processed = False
    st.session_state.total_chunks = 0
    clear_vector_db()


status = get_docs_status()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Documents Loaded", status["loaded"])
with col2:
    st.metric("Chunks Indexed", status["chunks"])
with col3:
    st.metric("Status", "✓ Ready" if status["processed"] else "⏳ Not Ready")

uploaded_files = st.file_uploader(
    "Upload your documents", accept_multiple_files=True, type=list(SUPPORTED_FORMATS)
)

if uploaded_files:
    st.session_state.loaded_documents = []
    st.session_state.docs_processed = False
    progress_bar = st.progress(0)

    for idx, file in enumerate(uploaded_files):
        file_ext = f".{file.name.split('.')[-1].lower()}"
        file_size_mb = file.size / (1024 * 1024)

        if file_ext not in SUPPORTED_FORMATS:
            st.error(f"❌ {file.name}: Unsupported format '{file_ext}'")
            continue

        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(
                f"❌ {file.name}: Size ({file_size_mb:.2f}MB) exceeds {MAX_FILE_SIZE_MB}MB"
            )
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name

        content = extract_text_from_file(tmp_path, file_ext)

        if content:
            doc_data = {
                "name": file.name,
                "type": file_ext.replace(".", ""),
                "size_mb": round(file_size_mb, 2),
                "content": content,
                "char_count": len(content),
                "word_count": len(content.split()),
            }
            st.session_state.loaded_documents.append(doc_data)
            st.success(f"✓ {file.name}")

        os.remove(tmp_path)
        progress_bar.progress((idx + 1) / len(uploaded_files))

if st.session_state.loaded_documents:
    st.subheader(f"Loaded Documents ({len(st.session_state.loaded_documents)})")

    col1, col2, col3 = st.columns([0.65, 0.17, 0.18])

    with col2:
        if st.button("⚙️ Process for RAG", use_container_width=True):
            with st.spinner("Creating embeddings..."):
                if process_and_store_documents(st.session_state.loaded_documents):
                    mark_docs_processed(st.session_state.total_chunks)
                    st.success(f"✓ {st.session_state.total_chunks} chunks indexed")
                    st.balloons()

    with col3:
        if st.button("🗑️ Clear All", use_container_width=True):
            clear_all()
            st.rerun()

    if st.session_state.docs_processed:
        st.info(f"✓ Documents ready for chat! Visit Chat page to ask questions.")

    # ✅ Fixed: Pass index to display_document
    for idx, doc in enumerate(st.session_state.loaded_documents):
        display_document(doc, idx)
else:
    st.info("📁 No documents loaded")
