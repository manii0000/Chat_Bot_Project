import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document

load_dotenv()  # Load env vars

# Load from env
API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "models/embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))
LLM_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro-latest")

# Utility Functions
def load_document(file):
    name, ext = os.path.splitext(file)
    loader_map = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.txt': TextLoader
    }
    loader_class = loader_map.get(ext)
    if not loader_class:
        st.error('Unsupported file type.')
        return None
    return loader_class(file).load()

def chunk_data(data, chunk_size=512, chunk_overlap=20):
    if not data:
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    full_text = " ".join([page.page_content for page in data])
    return text_splitter.split_text(full_text)

def create_embedding(chunks):
    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIM, google_api_key=API_KEY)
    return Chroma.from_documents(docs, embedding=embeddings)

def ask_and_get_answer(vector_store, q, k=3, chat_history=""):
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL_NAME"),
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

    # Append previous chat history to the current question
    full_prompt = f"{chat_history}\nUser: {q}\nAssistant:"
    
    return chain.invoke(full_prompt)


def calculate_embedding_cost(texts):
    tokens = sum(len(chunk) for chunk in texts)
    return tokens, (tokens / 1000) * 0.0004

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

# Streamlit UI
st.title("📄 Document QA App")
with st.sidebar:
    api_key_input = st.text_input("🔑 Gemini API Key", type='password', value=API_KEY)
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        API_KEY = api_key_input

    uploaded_file = st.file_uploader("📤 Upload file", type=["pdf", "docx", "txt"])
    chunk_size = st.number_input("📦 Chunk Size", 100, 800, 512,on_change=clear_history)
    k = st.number_input("🔍 Top-K Chunks (k)", 1, 20, 3,on_change=clear_history)
    add_data = st.button("⚙️ Add Data", on_click=clear_history)

if uploaded_file and add_data:
    file_path = f"./temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    data = load_document(file_path)
    if data:
        chunks = chunk_data(data, chunk_size)
        tokens, cost = calculate_embedding_cost(chunks)
        st.write(f"💰 Estimated Cost: ${cost:.6f}, Tokens: {tokens}")
        vs = create_embedding(chunks)
        st.session_state.vs = vs
        st.success("✅ File embedded successfully!")

q = st.text_input("💬 Ask something about the document")
if q and 'vs' in st.session_state:
    chat_history = st.session_state.get("history", "")
    answer = ask_and_get_answer(st.session_state.vs, q, k, chat_history)

    st.text_area("📜 Answer", value=answer)

    history = st.session_state.get("history", "")
    history = f"Q: {q}\nA: {answer}\n{'-'*50}\n" + history
    st.session_state["history"] = history
    st.text_area("📚 Chat History", value=history, height=300)
