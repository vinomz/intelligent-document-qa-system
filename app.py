import streamlit as st
import requests
import os
from typing import List
from config import settings
from utils.logger import get_logger

logger = get_logger("streamlit_app")

# Ensure upload dir exists
os.makedirs(settings.DEFAULT_DOCS_PATH, exist_ok=True)

st.set_page_config(page_title="Intelligent Document Q&A System", layout="wide")

st.title("📚 Intelligent Document Q&A System")
st.write("Ask questions based on your uploaded documents & internal knowledge base.")


# ---------------------------
# 1. File Upload Section
# ---------------------------
st.subheader("📤 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDFs, TXT, or DOCX files",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Reindex Now"):
        saved_paths = []
        for file in uploaded_files:
            save_path = os.path.join(settings.DEFAULT_DOCS_PATH, file.name)
            saved_paths.append(save_path)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

        logger.info(f"Uploaded {len(uploaded_files)} file(s). Re-indexing the DB...")
        st.success(f"Uploaded {len(uploaded_files)} file(s). Re-indexing the DB...")
        reindex_url = settings.AI_RAG_API_URL + "reindex"
        response = requests.post(reindex_url)
        logger.info(f"Db re-index response ==> {response.json()}")
        st.info(f"{response.json()}")

        for path in saved_paths:
            try:
                os.remove(path)
                logger.info(f"Deleted file after reindex: {path}")
            except Exception as e:
                logger.error(f"Failed to delete {path}: {e}")
        
        st.success("Files cleared from local storage after reindex.")

# ---------------------------
# 2. Ask a Question
# ---------------------------
st.subheader("🔍 Ask a Question")

user_query = st.text_input("Enter your question:")

if st.button("Submit Query"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Getting answer from RAG..."):
            try:
                query_url = settings.AI_RAG_API_URL + "query"
                payload = {"question": user_query}
                response = requests.post(query_url, json=payload)

                if response.status_code != 200:
                    st.error(f"Error {response.status_code}: {response.text}")
                else:
                    data = response.json()

                    # Display answer
                    st.subheader("💡 Answer")
                    st.write(data["answer"])

                    # Display sources
                    if data["sources"]:
                        st.subheader("📄 Source Documents")
                        for src in data["sources"]:
                            st.write(f"• **{src['source']}** (Page: {src['page']})")
                    else:
                        st.info("No source documents used.")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")


# ---------------------------
# Footer
# ---------------------------
st.write("---")
st.caption("Powered by Streamlit + FastAPI + Chroma DB + Gemini API")
