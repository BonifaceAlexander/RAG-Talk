# services/embeddings.py
from typing import List, Optional, Union
from services.utils import get_openai_client
import config

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Return list of embeddings for the provided texts.
    Uses OpenAI embeddings when OPENAI_API_KEY is set, otherwise falls back
    to sentence-transformers (local).
    """
    client = get_openai_client()
    if client:
        # Use OpenAI client (v1+)
        try:
            resp = client.embeddings.create(model=config.DEFAULT_EMBEDDING_MODEL, input=texts)
            # The response may be object-like or dict-like. Try both.
            data = getattr(resp, "data", None)
            
            # Fallback dict access
            if data is None and hasattr(resp, "get"):
                 data = resp.get("data", None)

            embeddings = []
            if data:
                for item in data:
                    # item may be an object with .embedding or a dict with ['embedding']
                    emb = getattr(item, "embedding", None)
                    if emb is None and isinstance(item, dict):
                        emb = item.get("embedding")
                    if emb is not None:
                        embeddings.append(emb)
            
            # Fallback if no data shape matched
            if not embeddings:
                 # try raw dict-style access for safety if above failed mysteriously
                 try:
                    # assuming resp might be a dict
                    if isinstance(resp, dict):
                         embeddings = [d["embedding"] for d in resp["data"]]
                 except Exception:
                     # If data extraction failed entirely
                     pass

            if embeddings:
                return embeddings
            else:
                 raise RuntimeError("Unexpected OpenAI embeddings response shape.")
                 
        except Exception as e:
            # If the OpenAI call fails, fall back to local embedding model
            print(f"[embeddings] OpenAI embeddings failed, falling back locally: {e}")
    
    # Local fallback
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
        emb = model.encode(texts, show_progress_bar=False)
        return emb.tolist()
    except ImportError:
        print("[embeddings] sentence-transformers not found. Returning empty list.")
        return []
