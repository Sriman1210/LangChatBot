import streamlit as st
from groq import Groq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.schema import LLMResult
from langchain.llms.base import LLM
from typing import List, Optional
import os

# --- Setup GROQ Client ---
groq_client = Groq(api_key="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")

class GroqLLM(LLM):
    model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    temperature: float = 0.7
    max_tokens: int = 1024

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = groq_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            top_p=1,
            stream=False,
            stop=stop,
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "groq-streamlit"

# --- Initialize Vector Store and Chain (cached) ---
@st.cache_resource
def load_chain():
    loader = PyPDFLoader(file_path="data.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)

    retriever = vectordb.as_retriever()
    llm = GroqLLM()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    return chain


# --- Streamlit UI ---
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("🧠 Langchain Powered RAG Chatbot")

query = st.text_input("Ask your question:")

if query:
    with st.spinner("Thinking..."):
        chain = load_chain()
        result = chain({"query": query})
        st.success(result["result"])
