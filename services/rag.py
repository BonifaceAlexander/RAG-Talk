# services/rag.py
from typing import Optional, List, Any, Tuple
from services.utils import get_openai_client
import config

def answer_query(query: str, kb: Any, top_k: int = config.DEFAULT_RAG_TOP_K) -> Tuple[str, List[Any]]:
    """
    Answer a query using RAG over the provided Knowledge Base (kb).
    
    Args:
        query: User question.
        kb: Knowledge Base object (must have .query method).
        top_k: Number of chunks to retrieve.
        
    Returns:
        Tuple[str, List[dict]]: (Answer text, List of source documents)
    """
    docs = kb.query(query, top_k=top_k)
    context = "\n\n".join([d['text'] for d in docs])
    prompt = (
        "You are a helpful assistant. Use the context below to answer the question. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{query}\n\n"
        "Answer:"
    )

    client = get_openai_client()

    if client:
        try:
            # new API: client.chat.completions.create(...)
            resp = client.chat.completions.create(
                model=config.DEFAULT_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.DEFAULT_RAG_TEMPERATURE,
                max_tokens=config.DEFAULT_RAG_MAX_TOKENS,
            )
            # resp.choices[0].message.content OR resp.choices if structure is dict-like
            choices = getattr(resp, "choices", None)
            if choices and len(choices) > 0:
                 message = getattr(choices[0], "message", None)
                 answer = getattr(message, "content", "")
            else:
                 # Fallback for dict access if object access fails
                 answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"[LLM call failed: {e}]"
    else:
        # fallback extractive answer when no API key
        if docs:
            answer = "\n\n".join([d['text'] for d in docs[:2]])
        else:
            answer = "I don't know. No documents in the selected KB."
    return answer, docs
