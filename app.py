import streamlit as st 
from scraper import scraper_website
from rag_pipeline import create_vector_store, ask_question, load_vector_store 
from pdf_loader import load_pdf

st.markdown("""
# DevMind AI

AI-Powered Developer Knowledge Assistant
""") 

st.set_page_config(
    page_title="DevMind AI",
    layout="wide"
)

with st.sidebar:
    st.title("DevMind AI")
    st.markdown("""
    AI-Powered Developer Knowledge Assistant
    Features:
    - Multi-source RAG
    - Website scraping
    - PDF ingestion
    - Developer Q&A
    - Interview prep
    - Summarization
    """)

st.sidebar.markdown("## Suggested Questions")

suggestions = [
    "Explain query parameters",
    "Generate interview questions",
    "Summarize this topic",
    "Explain simply",
]

for s in suggestions:
    if st.sidebar.button(s):
        st.session_state.selected_question = s


if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("RAG Application")

mode = st.selectbox("Choose Mode",
                    ["Normal Q&A",
                     "Beginner Exaplanation",
                     "Interview Questions",
                     "Summarization",
                     "Comparison"])

urls = st.text_area("Enter the URL of the website you want to scrape:(one URL per line)")

uploaded_files = st.file_uploader("Upload PDFs", type = "pdf", accept_multiple_files = True)

if st.button("Process Sources"):
    with st.spinner("Processing sources..."):
        text_data = []

        # Process Websites
        url_list = urls.split("\n")

        for url in url_list:
            if url.strip() != "":
                text = scraper_website(url)
                text_data.append({
                    "source": url,
                    "content": text
                })

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

    final_response = answer + "\n\n### Sources:\n"

    # for source in sources:
    #     final_response += f"- {source}\n"
    for item in sources:
        final_response += f"""
                        ### Source:
                        {item['source']}
                        Snippet:
                        {item['content']}
                        """

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_response
    })

    with st.chat_message("assistant"):
        st.markdown(final_response) 

