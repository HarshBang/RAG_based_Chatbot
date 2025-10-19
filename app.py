# app.py
import os
import uuid
import streamlit as st
from typing import List, Dict
from pdf2image import convert_from_path
import pytesseract
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from chromadb import PersistentClient
from huggingface_hub import login

# ================= CONFIG ==================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-or-your-key")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_db"
PCA_DIM = 128
login("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")
# ============================================

st.set_page_config(page_title="CIC Chatbot", layout="wide")
st.title("🧠 CIC NLP Chatbot")
st.write("Upload PDFs, build a knowledge base, and chat with them!")

# OCR + PDF Text Extraction
def extract_text_pages(pdf_path: str, dpi: int = 200) -> List[Dict]:
    reader = PdfReader(pdf_path)
    pages_text = []
    for i, page in enumerate(reader.pages):
        raw_text = ""
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""
        if len(raw_text.strip()) < 40:
            images = convert_from_path(pdf_path, dpi=dpi, first_page=i+1, last_page=i+1)
            if images:
                ocr_text = pytesseract.image_to_string(images[0])
                pages_text.append({"page_num": i+1, "text": ocr_text})
        else:
            pages_text.append({"page_num": i+1, "text": raw_text})
    return pages_text

def load_multiple_pdfs(paths: List[str]) -> List[Document]:
    docs = []
    for path in paths:
        pages = extract_text_pages(path)
        for p in pages:
            text = p["text"].strip()
            if not text:
                continue
            metadata = {"source": os.path.basename(path), "page": p["page_num"]}
            docs.append(Document(page_content=text, metadata=metadata))
    return docs

def split_documents(docs: List[Document], chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

sbert = SentenceTransformer(SENTENCE_MODEL_NAME)
_pca_model = None

def embed_texts(texts: List[str]) -> np.ndarray:
    embs = sbert.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return embs

def maybe_fit_pca(embeddings: np.ndarray, n_components: int):
    global _pca_model
    if n_components is None:
        return embeddings
    _pca_model = PCA(n_components=n_components, whiten=True, random_state=42)
    return _pca_model.fit_transform(embeddings)

def maybe_transform_pca(embeddings: np.ndarray):
    if _pca_model is None:
        return embeddings
    return _pca_model.transform(embeddings)

client = PersistentClient(path=CHROMA_PERSIST_DIR)

def create_or_get_collection(name="pdf_docs"):
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name)

def persist_documents_to_chroma(documents: List[Document], collection_name="pdf_docs", pca_dim=None):
    texts = [d.page_content for d in documents]
    metadatas = [d.metadata or {} for d in documents]
    ids = [str(uuid.uuid4()) for _ in documents]
    emb = embed_texts(texts)
    if pca_dim is not None:
        emb = maybe_fit_pca(emb, pca_dim)
    coll = create_or_get_collection(collection_name)
    coll.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=emb.tolist())
    return coll

def build_langchain_chain(collection_name="pdf_docs"):
    from langchain.vectorstores import Chroma as LangChainChroma
    from langchain_core.embeddings import Embeddings
    class PCAEmbeddingFunction(Embeddings):
        def embed_documents(self, texts): return maybe_transform_pca(embed_texts(texts)).tolist()
        def embed_query(self, text): return maybe_transform_pca(embed_texts([text]))[0].tolist()
    lc_chroma = LangChainChroma(persist_directory=CHROMA_PERSIST_DIR, collection_name=collection_name, embedding_function=PCAEmbeddingFunction())
    retriever = lc_chroma.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(model="openai/gpt-3.5-turbo", temperature=0.2, openai_api_base=OPENROUTER_API_BASE, openai_api_key=OPENROUTER_API_KEY)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# ================= STREAMLIT UI =================
uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    temp_paths = []
    for f in uploaded_files:
        temp_path = os.path.join("temp_uploads", f.name)
        os.makedirs("temp_uploads", exist_ok=True)
        with open(temp_path, "wb") as tmp:
            tmp.write(f.read())
        temp_paths.append(temp_path)

    st.info("Extracting and embedding PDFs... This may take a few minutes ⏳")
    documents = load_multiple_pdfs(temp_paths)
    docs = split_documents(documents)
    persist_documents_to_chroma(docs, pca_dim=PCA_DIM)
    st.success(f"✅ Loaded {len(documents)} pages and {len(docs)} chunks.")
    chain = build_langchain_chain()

    st.divider()
    st.subheader("💬 Chat with your documents")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question:")
    if query:
        resp = chain({"question": query})
        answer = resp["answer"]
        st.session_state.chat_history.append((query, answer))

    for q, a in st.session_state.chat_history[::-1]:
        with st.chat_message("user"): st.markdown(q)
        with st.chat_message("assistant"): st.markdown(a)
