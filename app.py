import streamlit as st 
from scraper import scraper_website
from rag_pipeline import create_vector_store, ask_question, load_vector_store 
from pdf_loader import load_pdf

st.set_page_config(
    page_title="MultiSource RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

h1, h2, h3 {
    font-weight: 700;
}

.stChatMessage {
    padding: 1rem;
    border-radius: 15px;
    margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.1);
}

.stButton {
    margin-bottom: 0.5rem;
}

.stButton button {
    width: 100%;
    border-radius: 10px;
    height: 3rem;
    font-weight: 600;
}

.stTextArea textarea {
    border-radius: 12px;
}

.stFileUploader {
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.15);
    background-color: rgba(255,255,255,0.02);
    padding: 1rem;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    min-width: 280px;
    max-width: 280px;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: white;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div style='text-align: center; padding: 1rem 0 2rem 0;'>
    <h1 style='font-size: 3rem;'>🤖 MultiSource RAG Assistant</h1>
    <p style='font-size: 1.1rem; color: gray;'>
        AI-Powered Knowledge Retrieval for PDFs and Websites
    </p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("MultiSource RAG")
    st.markdown("""
    ### Features
    - Website scraping
    - PDF ingestion
    - Semantic search
    - Context-aware responses
    - Interview preparation
    - Technical summarization
    - Conversational RAG
    """)

    st.divider()
    st.markdown("Suggested Questions")

    suggestions = [
        "Explain REST APIs",
        "Summarize this document",
        "Generate interview questions",
        "Compare React and Angular",
    ]

    for s in suggestions:
        if st.button(s):
            st.session_state.selected_question = s

    st.divider()

    if st.button("Clear Chat & Reset"):

        st.session_state.messages = []
        st.session_state.vector_store = None

        st.rerun()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

col1, col2, col3 = st.columns(3)

with col1:
    st.info("📄 PDF + Website Support")

with col2:
    st.info("🧠 FAISS Vector Store")

with col3:
    st.info("⚡ Llama 3 via Groq")


col1, col2 = st.columns(2)

with col1:
    mode = st.selectbox(
        "Choose Interaction Mode",
        [
            "Normal Q&A",
            "Beginner Explanation",
            "Interview Questions",
            "Summarization",
            "Comparison"
        ]
    )

with col2:
    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type="pdf",
        accept_multiple_files=True
    )

urls = st.text_area(
    "Enter Website URLs (one per line)",
    height=150,
    placeholder="https://example.com"
)


if st.button("Process Sources", use_container_width=True):

    with st.spinner("Processing sources..."):

        text_data = []

        # Process Websites
        url_list = urls.split("\n")

        for url in url_list:

            if url.strip() != "":

                text = scraper_website(url)

                if text and not text.startswith("Error"):

                    text_data.append({
                        "source": url,
                        "content": text
                    })

                else:
                    st.warning(f"Could not scrape: {url}")

        # Process PDFs
        for uploaded_file in uploaded_files:

            with open(f"data/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())

            docs = load_pdf(f"data/{uploaded_file.name}")

            pdf_text = ""

            for doc in docs:
                pdf_text += doc.page_content

            text_data.append({
                "source": uploaded_file.name,
                "content": pdf_text
            })

        if not text_data:
            st.error("No valid sources found.")
            st.stop()

        vector_store = create_vector_store(text_data)

        st.session_state.vector_store = vector_store

        st.success("Sources processed and vector store created successfully!")


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


default_query = st.session_state.get(
    "selected_question",
    ""
)

query = st.chat_input(
    "Ask a question"
) or default_query


if query:

    if st.session_state.vector_store is None:
        st.error("Please process sources first.")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Generating answer..."):

        answer, sources = ask_question(
            st.session_state.vector_store,
            query,
            mode
        )

    final_response = answer 

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_response
    })

    with st.chat_message("assistant"):
        st.markdown(final_response)

st.markdown("""
<hr>
<center>
Built using Streamlit, LangChain, FAISS, and Groq LLM
</center>
""", unsafe_allow_html=True)

