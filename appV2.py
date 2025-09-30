import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile

# Load environment variables
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

st.title("📄 Multi-PDF Chatbot")

# ---------- Upload PDFs ----------
uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []

    # Save uploaded PDFs to temporary files and load them
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            loader = PyPDFLoader(tmp_file.name)
            documents.extend(loader.load())

    # ---------- Split documents ----------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    # ---------- Create embeddings ----------
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # ---------- Vector store ----------
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="chroma_store"
    )

    # ---------- Retriever ----------
    retriever = vectordb.as_retriever(search_kwargs={"k":5})

    # ---------- Groq LLM ----------
    llm = ChatGroq(
        api_key=groq_key,
        model="llama-3.1-8b-instant",
        temperature=0
    )

    # ---------- Prompt Template ----------
    template = """Use the following context to answer the question.
    If you don't know, just say you don't know. Be concise.
    {context}
    Question: {question}
    Answer:"""
    QA_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )

    # ---------- Chat Section ----------
    if "history" not in st.session_state:
        st.session_state.history = []

    query = st.text_input("Ask the bot a question about your PDFs:")

    if query:
        result = qa_chain.invoke({"query": query})
        answer = result["result"]

        # Save chat history
        st.session_state.history.append({"user": query, "MyGuy": answer})

    # Display chat history
    for chat in st.session_state.history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**MyGuy:** {chat['MyGuy']}")
else:
    st.info("Please upload at least one PDF to start chatting.")
