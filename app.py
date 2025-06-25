import streamlit as st
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

st.set_page_config(page_title="PDF Question Answering System", layout="centered")
st.title("PDF Question Answering System with OpenAI GPT, LangChain, and Streamlit")

openai_api_key = ""
if not openai_api_key:
    st.warning("Please set your OPENAI_API_KEY environment variable.")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    st.success("PDF loaded successfully!")
    question = st.text_input("Ask a question about the PDF:")
    if question:
        with st.spinner("Generating answer..."):
            # Split the PDF into even smaller chunks
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectorstore = FAISS.from_documents(docs, embeddings)
            # Retrieve fewer relevant chunks
            relevant_docs = vectorstore.similarity_search(question, k=2)
            llm = OpenAI(openai_api_key=openai_api_key, temperature=0, max_tokens=1024)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.invoke({"input_documents": relevant_docs, "question": question})
            st.markdown(f"**Answer:** {answer['output_text']}")
else:
    st.info("Please upload a PDF to get started.")
