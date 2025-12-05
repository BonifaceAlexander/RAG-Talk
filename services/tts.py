
from typing import Optional
from services.utils import get_openai_client
import config

def text_to_speech(text: str) -> Optional[bytes]:
    """
    Convert text to speech using OpenAI's TTS API.
    
    Args:
        text: The text to convert to speech.
        
    Returns:
        bytes: The audio data in MP3 format, or None if failed.
    """
    client = get_openai_client()
    if not client:
        return None
        
    try:
        response = client.audio.speech.create(
            model=config.DEFAULT_TTS_MODEL,
            voice=config.DEFAULT_TTS_VOICE,
            input=text
        )
        # response.content gives raw bytes
        return response.content
    except Exception as e:
        print(f"TTS Error: {e}")
        return None
