# 📄 PDF Analyzer using RAG (Retrieval-Augmented Generation)

##  Project Overview

This project is an **Information Retrieval system** built to analyze PDF documents using **RAG (Retrieval-Augmented Generation)**. The system allows users to **upload a PDF document**, ask **questions related to its content**, and get **relevant answers** generated using advanced retrieval and generation techniques.

##  Features

-  Upload any PDF document
-  Use RAG to retrieve relevant text chunks from the document
-  Enter natural language queries via an intuitive UI
-  Get accurate, context-aware answers based on the uploaded document
  
##  How It Works

1. The user uploads a PDF document.
2. The document is preprocessed and chunked into text segments.
3. These chunks are embedded and stored for retrieval.
4. When a user enters a question:
   - Relevant chunks are retrieved using vector similarity.
   - The retrieved chunks are passed to a language model for answer generation.
5. The answer is displayed in the UI.

## User Interface

- Built a simple and clean web interface.
- User can:
  - Upload a PDF
  - Enter a question in the query box
  - View the generated answer below

##  Technologies Used

- **Python**
- **LangChain (for RAG)**
- **PDF parsing** (pdfplumber )
- **FAISS** for vector-based retrieval
- **Streamlit** for UI 

##  Academic Context

This is a semester project for the **Information Retrieval** course. The goal is to implement a practical solution that demonstrates:
- Text retrieval
- Document chunking
- Semantic similarity
- Natural Language Question Answering

##  Future Improvements
- Support for multiple PDFs
- Chat history and conversation memory
- Advanced summarization
- Better UI/UX design
##  Team Members
- Quratulain
- Radeel Ayesha Khan


