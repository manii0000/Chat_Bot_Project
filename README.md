
# ChatBot_project
A conversational chatbot web app built using Streamlit and Python.


# 💬 AI Chatbot using LangChain, Google Gemini & ChromaDB

This project is a user-friendly AI-powered chatbot built using **LangChain**, **Google Gemini API**, and **ChromaDB**. It allows users to upload documents (PDF, TXT, DOCX), automatically extracts the content, creates vector embeddings, and enables **Retrieval-Augmented Generation (RAG)** for document-based question answering.

---

## 🚀 Features

- 📄 Upload PDF, DOCX, or TXT files
- 🔍 Automatic document chunking
- 🧠 Embedding with Google Gemini (embedding-001)
- 🧠 LLM responses using Google Gemini (gemini-1.5-pro)
- 🧾 ChromaDB vector storage
- 🤖 RAG-based question-answering system
- 🧠 Creating and Deleting History
- 🧠 Built with LangChain for retrieval pipeline
- 🌐 Streamlit-based web UI

---


---

## 🔧 Tech Stack

- **Python**
- **LangChain**
- **Google Generative AI API (Gemini)**
- **ChromaDB/PineCone** for vector database
- **Streamlit** for frontend UI

---

## 📦 Installation

1. Clone this repo:
   ```bash
   - git clone https://github.com/yourusername/chatbot-project.git
   - cd chatbot-project

2. Create a virtual environment and activate it:
    - python -m venv venv
    - source venv/bin/activate     # On Windows: venv\Scripts\activate

3. Install dependencies:
    - pip install -r requirements.txt

4. Add your Google Gemini API key in a .env file:
    - GEMINI_API_KEY=your_api_key_here

▶️ How to Run
```bash
streamlit run chatbot.py
#Then open the browser link (usually http://localhost:8501) to interact with your chatbot.
