'''
Chunking
Embeddings
Vector Store / Vector Database
Retrieval
'''
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_core.documents import Document 
from langchain_groq import ChatGroq 
import os 
from dotenv import load_dotenv
import streamlit as st  

load_dotenv() 

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# def create_vector_store(text_data):
#     documents = []
#     for item in text_data:
#         documents.append(
#             Document(
#                 page_content = item["content"],
#                 metadata={"source": item["source"]}
#             )
#         )
    
#     # Chunking
#     chunks = text_splitter.split_documents(documents)

#     vector_store = FAISS.from_documents(chunks, embedding_model)
#     return vector_store 

@st.cache_resource
def create_vector_store(text_data):
    documents = []
    for item in text_data:
        documents.append(
            Document(
                page_content=item["content"],
                metadata={"source": item["source"]}
            )
        )
    chunks = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )
    vector_store.save_local("faiss_index")
    return vector_store

def load_vector_store():
    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local(
            "faiss_index",
            embedding_model,
            allow_dangerous_deserialization=True
        )
        return vector_store
    return None

# Retrieval 

def ask_question(vector_store, query, mode="Normal Q&A"):

    llm = ChatGroq(model = "llama-3.1-8b-instant", api_key = os.getenv("GROQ_API_KEY"))

    retrieved_docs = vector_store.similarity_search(query, k = 4)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    sources = list(set(
        [doc.metadata["source"] for doc in retrieved_docs]
    ))

    if mode == "Beginner Explanation":
        instruction = "Explain in very simple beginner-friendly language."

    elif mode == "Interview Questions":
        instruction = "Generate technical interview questions with answers from the context."

    elif mode == "Summarization":
        instruction = "Provide a concise technical summary."

    elif mode == "Comparison":
        instruction = "Compare concepts clearly with differences."
    chat_history = ""

    if "messages" in st.session_state:
        for msg in st.session_state.messages[-4:]:

            chat_history += f"""
            {msg['role']}:
            {msg['content']}
            """

    prompt = f"""
    You are an AI Developer Knowledge Assistant.
    If answer is not explicitly present in context, say you could not find the information. Do not generate assumptions.
    {instruction}
    Previous Conversation History: {chat_history}
    Context: {context}
    Question: {query} 
    """

    response = llm.invoke(prompt)

    source_chunks = []

    for doc in retrieved_docs:

        source_chunks.append({
            "source": doc.metadata["source"],
            "content": doc.page_content[:300]
        })

    return response.content, source_chunks

# Answer ONLY from provided context. If info is missing, say you couldn't find it. Do not hallucinate.
# Answer ONLY using the provided context.

#     Rules:
#     - Give clear and concise answers.
#     - Explain technical concepts properly.
#     - If information is missing, say you could not find it.
#     - Do not hallucinate.
#     - Mention important technical details.
