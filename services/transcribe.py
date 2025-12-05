# services/transcribe.py
import os
import io
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple, Any

from services.utils import get_openai_client

def transcribe_audio(fileobj: Any, filename_hint: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcribe audio BytesIO-like object -> text.
    
    Args:
        fileobj: A file-like object containing audio data.
        filename_hint: Optional filename to help with format detection.
        
    Returns:
        Tuple[str, str]: (transcript_text, error_message).
                         One of them will be None.
    """

    client = get_openai_client()
    if client is None:
        return None, "[No OPENAI_API_KEY set — transcription unavailable. Provide OPENAI_API_KEY for real transcription.]"

    # Ensure we can read from the file-like
    try:
        fileobj.seek(0)
    except Exception:
        pass

    # Try to import pydub for robust format handling
    try:
        from pydub import AudioSegment
    except ImportError as e:
        return None, f"[Transcription failed: missing pydub or ffmpeg. Install pydub and ffmpeg. Error: {e}]"
    except Exception as e:
        return None, f"[Transcription failed: pydub error: {e}]"

    # Load with pydub (auto-detect format if possible)
    audio_segment = None
    raw = fileobj.read()
    bio = io.BytesIO(raw)

    # Try format hint first (if available)
    if filename_hint and "." in filename_hint:
        fmt = filename_hint.split(".")[-1].lower()
        try:
            audio_segment = AudioSegment.from_file(bio, format=fmt)
        except Exception:
            bio.seek(0)
            audio_segment = None

    # If no audio_segment yet, try autodetect
    if audio_segment is None:
        try:
            bio.seek(0)
            audio_segment = AudioSegment.from_file(bio)
        except Exception as e:
            return None, f"[Whisper transcription failed: could not parse uploaded audio ({e})]"

    # Export to a proper WAV temp file (with headers)
    tmp = NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        audio_segment.export(tmp.name, format="wav")
    except Exception as e:
        try:
            tmp.close()
        except Exception:
            pass
        return None, f"[Whisper transcription failed: could not export audio to WAV ({e})]"

    # Call OpenAI Whisper via new client API
    try:
        with open(tmp.name, "rb") as fh:
            resp = client.audio.transcriptions.create(model="whisper-1", file=fh)
        
        # Extract text (handle both object-like and dict-like)
        text = getattr(resp, "text", None)
        if text is None:
            # Fallback for dict-like response
             if hasattr(resp, "get"):
                text = resp.get("text", "")
             else:
                text = ""
        return text, None
    except Exception as e:
        return None, f"Whisper transcription failed: {e}"
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
