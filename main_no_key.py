
import os
from pathlib import Path
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import nltk
from nltk.tokenize import sent_tokenize
import math

nltk.download("punkt", quiet=True)

PERSIST_DIR = "chroma_db_local"
DATA_DIR = Path("data")

def get_embeddings():
    
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vectorstore(data_dir=DATA_DIR, persist_directory=PERSIST_DIR):
    texts = []
    metadatas = []
    for p in sorted(data_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        texts.append(text)
        metadatas.append({"source": p.name})
    embeddings = get_embeddings()
    vectordb = Chroma.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb

def load_vectorstore(persist_directory=PERSIST_DIR):
    embeddings = get_embeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb

def retrieve_docs(vectordb, question, k=3):
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(question)
    return docs

def extractive_summary_from_docs(docs, question, max_sentences=3):
    """
    Very simple heuristic extractive summarizer:
    - Split retrieved docs into sentences.
    - Score by how many query words they contain.
    - Return top scoring sentences (deduplicated) joined as a short summary.
    """
    if not docs:
        return "No relevant documents retrieved."

    q_tokens = [t.lower() for t in question.split() if len(t) > 2]
    sentence_scores = []
    seen = set()
    for d in docs:
        sents = sent_tokenize(d.page_content)
        for s in sents:
            key = s.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            low = key.lower()
            score = sum(low.count(qt) for qt in q_tokens)
            sentence_scores.append((score, len(low), key))  

    # Sort by (score desc, shorter length)
    sentence_scores.sort(key=lambda x: ( -x[0], x[1]))
    picked = [s for sc, ln, s in sentence_scores if sc>0][:max_sentences]

    if not picked:
        # take first sentences from docs
        fallback = []
        for d in docs:
            sents = sent_tokenize(d.page_content)
            if sents:
                fallback.append(sents[0])
            if len(fallback) >= max_sentences:
                break
        picked = fallback

    summary = " ".join(picked)
    return summary

def salary_agent(question, vectordb, k=3):
    docs = retrieve_docs(vectordb, question, k=k)
    summary = extractive_summary_from_docs(docs, question)
    return summary, docs

def insurance_agent(question, vectordb, k=3):
    docs = retrieve_docs(vectordb, question, k=k)
    summary = extractive_summary_from_docs(docs, question)
    return summary, docs

# ---------- COORDINATOR ----------
def coordinator(question, vectordb, k=3):
    q = question.lower()
    salary_keywords = ["salary","pay","payslip","deduction","ctc","monthly","annual","hra","pf","bonus"]
    insurance_keywords = ["insurance","policy","premium","claim","coverage","sum insured","benefit","insurer"]
    if any(kword in q for kword in salary_keywords):
        agent_name = "Salary Agent"
        answer, docs = salary_agent(question, vectordb, k=k)
    elif any(kword in q for kword in insurance_keywords):
        agent_name = "Insurance Agent"
        answer, docs = insurance_agent(question, vectordb, k=k)
    else:
        agent_name = "Coordinator"
        answer = "I couldn't confidently assign the question to Salary or Insurance. Please ask a clear question about salary or insurance."
        docs = []
    return agent_name, answer, docs

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Multi-Agent RAG (No Key)", layout="wide")
st.title("Multi-Agent RAG Demo — No API Key (Local embeddings)")

if 'vectordb_ready' not in st.session_state:
    st.session_state.vectordb_ready = False

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Index / Data")
    if st.button("Build (rebuild) vector store from data/*.txt"):
        with st.spinner("Building local vector store (this downloads model files first time)..."):
            try:
                vectordb = build_vectorstore()
                st.session_state.vectordb_ready = True
                st.success("Vector store built and persisted to disk (local).")
            except Exception as e:
                st.error(f"Error building vector store: {e}")

    if st.button("Load existing vector store"):
        try:
            vectordb = load_vectorstore()
            st.session_state.vectordb_ready = True
            st.success("Loaded persisted vector store.")
        except Exception as e:
            st.error(f"No persisted DB found. Build it first. ({e})")

    st.markdown("**Data folder:** `data/` — put `.txt` files here.")
    st.text(f"PERSIST_DIR = {PERSIST_DIR}")

with col2:
    st.header("Ask a question (no external LLM)")
    query = st.text_input("Enter your question (salary or insurance)", "")
    k = st.slider("Retriever: how many relevant chunks to fetch (k)", 1, 5, 3)
    if st.button("Ask") and query.strip():
        if not st.session_state.vectordb_ready:
            st.warning("Vector store not loaded. Click 'Build' or 'Load' first.")
        else:
            try:
                vectordb = load_vectorstore()
                agent_name, answer, docs = coordinator(query, vectordb, k=k)
                st.subheader(f"Answer (by {agent_name})")
                st.write(answer)
                if docs:
                    st.subheader("Retrieved sources")
                    for d in docs:
                        st.markdown(f"**{d.metadata.get('source','unknown')}**")
                        st.write(d.page_content[:1000] + ("..." if len(d.page_content) > 1000 else ""))
            except Exception as e:
                st.error(f"Error during ask: {e}")

st.markdown("---")
st.caption("This demo uses local sentence-transformer embeddings and an extractive summarizer. No OpenAI key required.")
