
# RAGTalk — Audio → RAG (Multi Knowledge Base)

RAGTalk is a local, GitHub-ready project demonstrating an end-to-end
Audio → RAG pipeline with multi-knowledge-base support.

Features:
- Streamlit UI with browser mic recorder (streamlit-mic-recorder)
- Upload audio or record directly in browser
- Transcription using OpenAI Whisper (if OPENAI_API_KEY provided) with a fallback
- Chunking + embeddings (OpenAI embeddings if key present, else sentence-transformers)
- FAISS-based per-KB vector store
- Query KB with RAG-backed answers and sources
- Create / select / delete knowledge bases (KBs)

Quickstart:
1. Create virtualenv & activate it (Python 3.10+ recommended)
2. pip install -r requirements.txt
3. (Optional) export OPENAI_API_KEY="sk-..."
4. streamlit run app/app.py

Files:
- app/: Streamlit app
- services/: transcription, chunking, embeddings, vectorstore, RAG
- data/: sample audio and KB data written at runtime
