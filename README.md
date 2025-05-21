# 🧙‍♂️ Groq LLaMA-4 RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot powered by LangChain and Groq's blazing-fast LLaMA-4 model. This app allows users to ask questions from a PDF or CSV-based knowledge base and receive accurate, context-aware answers.

## 🚀 Features

* 🧠 Uses LangChain's `RetrievalQA` with HuggingFace embeddings + FAISS
* ⚡ Inference via Groq API using `LLaMA-4 Maverick 17B`
* 📄 Supports PDF and CSV data sources
* 🌐 Clean interactive frontend with Streamlit
* �� Bonus: Streamed LLM response (optional)

---

## 🗂️ Folder Structure

```
.
├── app.py                   # Streamlit UI
├── rag_chatbot_groq.py      # Core RAG logic
├── your_dataset.csv         # CSV file (or use data.pdf instead)
├── data.pdf                 # Optional PDF document
├── requirements.txt
└── README.md
```

---

## 📆 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/groq-rag-chatbot.git
cd groq-rag-chatbot
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Set up your Groq API key:**

Edit `rag_chatbot_groq.py` and add your key:

```python
groq_client = Groq(api_key="your_groq_api_key_here")
```

---

## 🧠 Supported Data Formats

You can use:

* `your_dataset.csv`: A simple CSV file
* `data.pdf`: Any text-based PDF

LangChain will load and chunk these documents to create embeddings and a retriever.

---

## 🏃 Run the Chatbot (Streamlit)

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧹 Example Questions

* What is LangChain used for?
* What topics are covered in this document?
* Give a summary of section 2.

---

## 🛠 requirements.txt

```txt
streamlit
langchain
groq
faiss-cpu
sentence-transformers
pymupdf
```

---

## 📤 Deployment (Bonus)

You can deploy this on:

* [Streamlit Cloud](https://streamlit.io/cloud)

Make sure to securely store your `GROQ_API_KEY`.

---

## 📬 Submission Checklist

* ✅ `rag_chatbot_groq.py`
* ✅ `app.py`
* ✅ `responses.txt` or `.xlsx`
* ✅ Deployed Streamlit app (optional)
* ✅ Uploaded to GitHub repo

---

## 📧 Questions?

Email: [srimanvashishtavemula12@gmail.com](mailto:srimanvashishtavemula12@gmail.com)

---

## 📝 License

MIT License – free to use and modify.
