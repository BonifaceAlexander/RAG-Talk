import os
from openai import OpenAI
from typing import Optional

def get_openai_client() -> Optional[OpenAI]:
    """
    Lazily create and return an OpenAI client if OPENAI_API_KEY is set.
    Returns None if key not present.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None
