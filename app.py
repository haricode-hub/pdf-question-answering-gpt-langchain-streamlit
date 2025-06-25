import streamlit as st
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
import os

st.set_page_config(page_title="PDF Question Answering System", layout="centered")
st.title("PDF Question Answering System with OpenAI GPT, LangChain, and Streamlit")

openai_api_key = os.getenv("OPENAI_API_KEY")
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
            llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=documents, question=question)
            st.markdown(f"**Answer:** {answer}")
else:
    st.info("Please upload a PDF to get started.")
