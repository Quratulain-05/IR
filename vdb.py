from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

#Step1: Upload and load raw pdfs
pdfs_directory='PDFs/'

def upload_pdf(file):
    with open(pdfs_directory+file.name,"wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader=PDFPlumberLoader(file_path)
    documents=loader.load()
    return documents

file_path=r"C:\Users\radeelayesha\Desktop\rag_with_deepseek\udhr.pdf"
documents=load_pdf(file_path)
#print(len(documents))

#Step 2: Create chunks
def create_chunks(documents):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    text_chunks=text_splitter.split_documents(documents)
    return text_chunks

text_chunks=create_chunks(documents)
#print("Chunks count: ",len(text_chunks))

#Step 3:Setup Embeddings Model (Use Deepseek R1 with Ollama)
ollama_model_name="nomic-embed-text"
def get_embeddings_model(ollama_model_name):
    embeddings=OllamaEmbeddings(model=ollama_model_name)
    return embeddings

#Step 4: Index documents Store embeddings in FAISS
FAISS_DB_PATH="vectorestore/db_faiss"
faiss_db=FAISS.from_documents(text_chunks,get_embeddings_model(ollama_model_name))
faiss_db.save_local(FAISS_DB_PATH)