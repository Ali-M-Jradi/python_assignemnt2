import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "c:/Users/hp 15/Desktop/python_university/assignment2/data/chroma_db"

os.makedirs(DB_PATH, exist_ok=True)


# ✅ Use session_state instead of global variables
def get_embedding_model():
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return st.session_state.embedding_model


def get_vector_db():
    if "vector_db_client" not in st.session_state:
        st.session_state.vector_db_client = chromadb.PersistentClient(path=DB_PATH)
    return st.session_state.vector_db_client


def get_collection():
    if "collection" not in st.session_state:
        client = get_vector_db()
        existing_collections = [c.name for c in client.list_collections()]

        if "documents" in existing_collections:
            st.session_state.collection = client.get_collection(name="documents")
        else:
            st.session_state.collection = client.create_collection(
                name="documents", metadata={"hnsw:space": "cosine"}
            )
    return st.session_state.collection


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def embed_texts(texts):
    model = get_embedding_model()
    embeddings = model.encode(
        texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32
    )
    return [vec.astype(np.float32).tolist() for vec in embeddings]


def process_and_store_documents(documents):
    if not documents:
        st.error("No documents to process")
        return False

    try:
        collection = get_collection()
        progress_bar = st.progress(0)
        total_docs = len(documents)
        total_chunks_count = 0

        for idx, doc in enumerate(documents):
            st.write(f"Processing: {doc['name']}")

            chunks = chunk_text(doc["content"])
            total_chunks_count += len(chunks)

            embeddings = embed_texts(chunks)

            ids = []
            metadatas = []
            doc_chunks = []

            for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc_id = f"{doc['name'].replace('.', '_')}_{chunk_idx}"

                ids.append(doc_id)
                metadatas.append(
                    {
                        "source": doc["name"],
                        "type": doc["type"],
                        "chunk_index": str(chunk_idx),
                        "total_chunks": str(len(chunks)),
                    }
                )
                doc_chunks.append(chunk)

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=doc_chunks,
            )

            st.success(f"✓ {doc['name']}: {len(chunks)} chunks indexed")
            progress_bar.progress((idx + 1) / total_docs)

        st.session_state.docs_processed = True
        st.session_state.total_chunks = total_chunks_count

        return True

    except Exception as e:
        st.error(f"Error processing documents: {e}")
        return False


def retrieve_chunks(query, top_k=10):
    try:
        collection = get_collection()

        # ✅ Check if collection has any documents
        count = collection.count()
        if count == 0:
            st.warning(
                "⚠️ No documents in database. Please upload and process documents first."
            )
            return []

        query_embedding = embed_texts([query])[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results["documents"] or len(results["documents"][0]) == 0:
            return []

        retrieved_chunks = []
        for doc, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            similarity = 1 - distance

            if similarity < 0.1:
                continue

            retrieved_chunks.append(
                {
                    "text": doc,
                    "source": metadata["source"],
                    "type": metadata["type"],
                    "chunk_index": metadata["chunk_index"],
                    "relevance": round(max(0, similarity), 3),
                }
            )

        return retrieved_chunks[:top_k]

    except Exception as e:
        st.error(f"Error retrieving chunks: {e}")
        return []


def get_collection_info():
    try:
        collection = get_collection()
        count = collection.count()
        return {
            "name": collection.name,
            "count": count,
            "metadata": collection.metadata,
        }
    except Exception as e:
        st.error(f"Error getting collection info: {e}")
        return None


def clear_vector_db():
    try:
        client = get_vector_db()
        client.delete_collection(name="documents")

        # ✅ Clear from session state
        if "collection" in st.session_state:
            del st.session_state.collection

        st.success("Vector database cleared")
        return True
    except Exception as e:
        st.warning("Database already empty or error clearing")
        return False
