from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from vdb import faiss_db
#Step 1: Setup LLM (Use Deepseek R1 with Groq)
from dotenv import load_dotenv
load_dotenv()

llm_model=ChatGroq(model="deepseek-r1-distill-llama-70b")

#Step 2: Retrieve docs
def retrieve_docs(query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    context="\n\n".join([doc.page_content for doc in documents])
    return context

#Step 3: Answer Question
custom_prompt_template="""
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Don't provide anything out of the given context
Question: {question} 
Context: {context} 
Answer:
"""

def answer_query(documents,model,query):
    context=get_context(documents)
    prompt=ChatPromptTemplate.from_template(custom_prompt_template)
    chain=prompt|model
    return chain.invoke({"question": query, "context": context})



question="What are the basic human rights?"
retrieved_docs=retrieve_docs(question)
print("Bot: ",answer_query(documents=retrieved_docs, model=llm_model, query=question))