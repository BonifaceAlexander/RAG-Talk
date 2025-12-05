import os
from pathlib import Path

# Data Directory
DATA_DIR = Path("data/kbs")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Audio Uploads
ALLOWED_AUDIO_EXTENSIONS = ['wav', 'mp3', 'm4a']

# Models
DEFAULT_CHAT_MODEL = "gpt-3.5-turbo"

# TTS Configuration
DEFAULT_TTS_MODEL = "tts-1"
DEFAULT_TTS_VOICE = "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer

# Embedding Models
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# RAG specific
DEFAULT_RAG_TOP_K = 4
DEFAULT_RAG_TEMPERATURE = 0.0
DEFAULT_RAG_MAX_TOKENS = 512
