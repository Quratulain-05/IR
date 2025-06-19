import os
import streamlit as st
from rag_pipeline import answer_query, llm_model
from vdb import get_embeddings_model
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Temp directory for uploaded PDFs
pdfs_directory = 'PDFs/'
os.makedirs(pdfs_directory, exist_ok=True)

# Helper functions
def upload_pdf(file):
    file_path = os.path.join(pdfs_directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

# Streamlit UI
st.title("RAG-based PDF QA System")

upload_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)
user_query = st.text_area("Enter your prompt:", height=150, placeholder="Ask anything!")

if st.button("Ask a Question"):
    if upload_file and user_query.strip():
        st.chat_message("user").write(user_query)

        # Check if vectorstore is already cached for this file
        if "faiss_db" not in st.session_state or st.session_state.get("cached_file") != upload_file.name:
            # Save the file and generate vectorstore
            file_path = upload_pdf(upload_file)
            documents = load_pdf(file_path)
            text_chunks = create_chunks(documents)

            embeddings = get_embeddings_model("deepseek-r1:1.5b")
            faiss_db = FAISS.from_documents(text_chunks, embeddings)

            # Cache vector DB and file name
            st.session_state.faiss_db = faiss_db
            st.session_state.cached_file = upload_file.name
        else:
            faiss_db = st.session_state.faiss_db  # Reuse cached one

        # Retrieve and respond
        retrieved_docs = faiss_db.similarity_search(user_query)
        response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)
        st.chat_message("Answer").write(response)
    else:
        st.error("Please upload a PDF and enter a prompt.")
