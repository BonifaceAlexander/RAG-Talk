
import sys
import os
import pathlib
import shutil
import traceback
from pathlib import Path
import io
import uuid
import base64

# Ensure project root is on sys.path so 'services' imports work when running from /app
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

# Local header image provided by user (developer-provided path)
# Use the uploaded file path from the conversation history
HEADER_IMAGE_PATH = "images/logo.jpg"

import config
# Import project services (must exist)
from services.transcribe import transcribe_audio
from services.vectorstore import KBManager
from services.rag import answer_query

# mic recorder optional
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_AVAILABLE = True
except Exception:
    MIC_AVAILABLE = False

# Page config
st.set_page_config(page_title="RAGTalk", layout="wide", initial_sidebar_state="expanded")

# ---------- Helpers ----------
def show_one_time_message():
    """
    Show a one-time message persisted in session_state under key "_one_time_msg".
    This ensures messages survive a st.rerun() and show exactly once.
    """
    msg = st.session_state.pop("_one_time_msg", None)
    if msg:
        st.success(msg)

def safe_rerun():
    """Call st.rerun() if available, else fallback to experimental_rerun()."""
    try:
        st.rerun()
    except Exception:
        st.rerun()

def get_img_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None

# ---------- CSS & Hero (PLACE HERO AT VERY TOP) ----------
VIBRANT_CSS = r"""
:root{
  --bg-0: #071021;
  --card: rgba(255,255,255,0.03);
  --muted: #9bb0c7;
  --accent-start: #ff4b5c;
  --accent-end: #ff8a65;
  --accent-alt: #6e9cff;
  --glass-border: rgba(255,255,255,0.04);
}

/* Base */
body, .block-container { background: radial-gradient(1200px 400px at 10% 10%, rgba(14,30,60,0.6), transparent), var(--bg-0); color: #fff; }
.stApp { background: transparent; }

/* Hero */
.vhero {
  display:flex;
  align-items:center;
  gap:20px;
  padding:22px;
  border-radius:14px;
  background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border: 1px solid var(--glass-border);
  box-shadow: 0 20px 60px rgba(2,6,23,0.7);
  margin-bottom:14px;
}
.vhero .logo { width:84px; height:84px; border-radius:14px; overflow:hidden; border:1px solid rgba(255,255,255,0.04); }
.vhero h1 { margin:0; font-size:28px; letter-spacing:0.2px; }
.vhero p { margin:0; color: var(--muted); }

/* Particle svg container */
.particles { pointer-events:none; opacity:0.85; }

/* Glass card */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-radius:14px;
  padding:16px;
  border: 1px solid var(--glass-border);
  backdrop-filter: blur(8px);
  transition: transform 220ms ease, box-shadow 220ms ease;
}
.card:hover { transform: translateY(-6px); box-shadow: 0 24px 60px rgba(2,6,23,0.7); }

/* Primary gradient button with glow */
.btn-accent {
  position: relative;
  display:inline-block;
  padding:10px 14px;
  border-radius:12px;
  background: linear-gradient(90deg, var(--accent-start), var(--accent-end));
  color:white;
  border:none;
  cursor:pointer;
  font-weight:600;
  box-shadow: 0 8px 30px rgba(255,75,92,0.12);
  transition: transform 160ms ease, box-shadow 200ms ease;
}
.btn-accent:hover { transform: translateY(-3px); box-shadow: 0 18px 40px rgba(255,75,92,0.18); }

/* Style Streamlit's native button to match */
.stButton>button {
  border-radius:12px !important;
  padding:10px 14px !important;
  background: linear-gradient(90deg, #ff4b5c, #ff8a65) !important;
  color: white !important;
  font-weight:600 !important;
  border: none !important;
  box-shadow: 0 8px 30px rgba(255,75,92,0.12) !important;
}
.stButton>button:hover { transform: translateY(-3px); box-shadow: 0 18px 40px rgba(255,75,92,0.18) !important; }


/* Fancy chip badge */
.chip {
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding:6px 10px;
  border-radius:999px;
  background: rgba(255,255,255,0.02);
  color:var(--muted);
  font-size:0.9rem;
  margin:4px 6px 4px 0;
  border-left: 4px solid rgba(255,255,255,0.03);
}
.chip.kb { border-left-color: var(--accent-alt); color:#eaf2ff; }

/* Skeleton */
@keyframes pulse {
  0% { opacity: 1; transform: translateY(0); }
  50% { opacity: 0.55; transform: translateY(-6px); }
  100% { opacity: 1; transform: translateY(0); }
}
.skeleton {
  background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.04));
  height:14px; border-radius:8px;
  animation: pulse 1.2s ease-in-out infinite;
  margin:8px 0;
}

/* small floating animation for cards */
.floating { animation: floaty 6s ease-in-out infinite; }
@keyframes floaty {
  0% { transform: translateY(0px) }
  50% { transform: translateY(-6px) }
  100% { transform: translateY(0px) }
}

/* compact muted */
.small-muted { color: #9bb0c7; font-size:0.95rem; }
"""

st.markdown(f"<style>{VIBRANT_CSS}</style>", unsafe_allow_html=True)

# Particle SVG (subtle decorative)
PARTICLE_SVG = """
<svg class="particles" viewBox="0 0 160 56" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" fill-rule="evenodd" stroke-opacity="0.12">
    <circle cx="12" cy="14" r="6" fill="#ff8a65" />
    <circle cx="36" cy="34" r="4" fill="#6e9cff" />
    <circle cx="68" cy="18" r="5" fill="#ff6f7a" />
    <circle cx="110" cy="38" r="3" fill="#ffd3c2" />
    <circle cx="140" cy="10" r="4" fill="#a3c9ff" />
  </g>
</svg>
"""
# ---------- HERO (top of page, rendered HTML) ----------
logo_b64 = get_img_as_base64(HEADER_IMAGE_PATH)
if logo_b64:
    img_src = f"data:image/jpeg;base64,{logo_b64}"
else:
    img_src = ""

hero_html = f"""
<div class="vhero card" role="banner" style="display:flex; align-items:center; gap:18px;">
  <div class="logo" style="width:84px;height:84px;border-radius:14px;overflow:hidden;border:1px solid rgba(255,255,255,0.04);">
    <img src="{img_src}" style="width:100%;height:100%;object-fit:cover;" alt="RAGTalk logo"/>
  </div>

  <div style="min-width:320px;">
    <h1 style="margin:0;">RAGTalk — Live Audio → RAG</h1>
    <p class="small-muted" style="margin:6px 0 0 0;">Transcribe, index & query multiple knowledge bases.</p>
  </div>

  <div style="margin-left:12px">{PARTICLE_SVG}</div>

  <div style="margin-left:auto;display:flex;gap:10px;align-items:center">
    <!-- Streamlit button will be rendered below and styled via CSS -->
    <div id="try-demo-button-placeholder"></div>
  </div>
</div>
"""

st.markdown(hero_html, unsafe_allow_html=True)

# Render an actual Streamlit button for interactivity (styled by CSS above)
if st.button("Try Demo", key="hero_try_demo"):
    st.session_state["_one_time_msg"] = "Demo started — demo behavior here."
    safe_rerun()

# ---------- Layout & KB Manager ----------
# Ensure data dir
kb_manager = KBManager(root_dir=str(config.DATA_DIR))

# Show persistent one-time message (if any) EARLY
show_one_time_message()

# Columns layout (now below the hero)
col_sidebar, col_main, col_right = st.columns([1, 2, 3])

# ---------------- Sidebar ----------------
with col_sidebar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("⚙️ Settings")

    existing_key = os.environ.get("OPENAI_API_KEY")
    if existing_key:
        st.markdown("<div class='small-muted'>OpenAI key present: <b>True</b></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='small-muted'>Paste OPENAI_API_KEY (session only)</div>", unsafe_allow_html=True)
        # non-empty label to avoid Streamlit warnings; hidden visually
        key_in = st.text_input("OpenAI API Key", type="password", label_visibility="collapsed",
                               help="Paste your OpenAI API key here for this session.")
        if key_in:
            os.environ["OPENAI_API_KEY"] = key_in.strip()
            st.session_state["_one_time_msg"] = "OPENAI_API_KEY set for this session (not saved)."
            safe_rerun()

    st.write("---")
    st.subheader("Knowledge Bases")
    kbs = kb_manager.list_kbs()
    if not kbs:
        kbs = []
    kb_choice = st.selectbox("Select KB", options=(kbs if kbs else ["<no KBs>"]), key="kb_select")

    with st.expander("Create New KB", expanded=False):
        new_kb_name = st.text_input("New KB name", key="new_kb_name")
        if st.button("Create KB", key="create_kb_btn"):
            if new_kb_name and new_kb_name.strip():
                try:
                    kb_manager.create_kb(new_kb.strip())
                    st.session_state["_one_time_msg"] = f"Created KB '{new_kb_name.strip()}'"
                    st.session_state.pop("new_kb_name", None)
                    safe_rerun()
                except Exception as e:
                    st.error(f"Failed to create KB: {e}")
                    st.text(traceback.format_exc())
            else:
                st.warning("Enter a non-empty KB name.")

    st.write("---")
    st.markdown("### Remove KB")
    if kb_choice and kb_choice != "<no KBs>":
        if st.button("Delete KB", key="delete_kb_btn"):
            try:
                # Resolve the path the app will attempt to delete
                # Use absolute path based on current working dir of the running process
                kb_dir = config.DATA_DIR / kb_choice
                st.write("Resolved KB path:", str(kb_dir.resolve()))
                st.write("Working directory (cwd):", os.getcwd())
                st.write("Exists before delete:", kb_dir.exists())

                # Attempt delete
                shutil.rmtree(kb_dir)
                st.write("rmtree succeeded")

                # Confirm removal
                st.write("Exists after delete:", kb_dir.exists())

                # Clear the selectbox state and any cached list so dropdown updates immediately
                if "kb_select" in st.session_state:
                    del st.session_state["kb_select"]
                if "kb_list" in st.session_state:
                    del st.session_state["kb_list"]

                st.success(f"Deleted KB '{kb_choice}'. Refreshing list...")
                st.rerun()

            except Exception as e:
                st.error("Delete failed: " + str(e))
                st.text(traceback.format_exc()) 

    st.write("---")
    st.markdown("<div class='small-muted'>KB Diagnostics</div>", unsafe_allow_html=True)
    st.write({"existing_kbs": kb_manager.list_kbs()})
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Main center: Recorder / Uploader / Transcript ----------------
with col_main:
    st.markdown("<div class='card floating'>", unsafe_allow_html=True)
    st.subheader("🎙️ Record or Upload")

    # In-browser recorder (optional)
    if MIC_AVAILABLE:
        st.markdown("**Record in your browser**")
        mic_bytes = mic_recorder(start_prompt="🎤 Start recording", stop_prompt="🤐 Stop recording", key="mic_recorder_1")
        if mic_bytes:
            st.audio(mic_bytes["bytes"], format="audio/wav")
            if st.button("Transcribe & Add (recording)"):
                if kb_choice in (None, "<no KBs>"):
                    st.warning("Select or create a KB in the sidebar before adding recordings.")
                else:
                    with st.spinner("Transcribing recording..."):
                        transcript, error = transcribe_audio(io.BytesIO(mic_bytes["bytes"]), filename_hint="recording.wav")
                    
                    if error:
                         st.error(error)
                    else:
                        st.session_state["_one_time_msg"] = "Transcription complete."
                        st.session_state["_last_transcript_preview"] = transcript
                        try:
                            kb_manager.add_transcript(kb_choice, f"recording-{uuid.uuid4().hex[:6]}", transcript)
                            st.session_state["_one_time_msg"] = f"Added to KB '{kb_choice}'"
                        except Exception as e:
                            st.session_state["_one_time_msg"] = f"Failed to add to KB: {e}"
                        safe_rerun()
    else:
        st.info("In-browser recorder not available (install streamlit-mic-recorder). You can upload audio below.")

    st.write("---")

    # File uploader
    uploaded = st.file_uploader("Drag audio here or browse (wav/mp3/m4a/ogg/mp4)", type=["wav", "mp3", "m4a", "ogg", "mp4"])
    if uploaded:
        try:
            st.audio(uploaded)
        except Exception:
            st.write("Audio preview not available for this file.")
        if st.button("Transcribe & Add (upload)", key="transcribe_upload_btn"):
            if kb_choice in (None, "<no KBs>"):
                st.warning("Select or create a KB in the sidebar before adding recordings.")
            else:
                with st.spinner("Transcribing upload..."):
                    transcript, error = transcribe_audio(uploaded, filename_hint=getattr(uploaded, "name", None))
                
                if error:
                    st.error(error)
                else:
                    st.session_state["_last_transcript_preview"] = transcript
                    try:
                        kb_manager.add_transcript(kb_choice, getattr(uploaded, "name", f"upload-{uuid.uuid4().hex[:6]}"), transcript)
                        st.session_state["_one_time_msg"] = f"Added '{getattr(uploaded, 'name', 'upload')}' to KB '{kb_choice}'"
                    except Exception as e:
                        st.session_state["_one_time_msg"] = f"Failed to add to KB: {e}"
                    safe_rerun()

    # Last transcript preview (one-time)
    if "_last_transcript_preview" in st.session_state:
        st.markdown("<div style='margin-top:12px'><b>Last transcript preview</b></div>", unsafe_allow_html=True)
        st.text_area("Preview", value=st.session_state.get("_last_transcript_preview", ""), height=220)
        del st.session_state["_last_transcript_preview"]

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Right: Query & Results ----------------
with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    # Logic to handle "Get Answer" click OR persist previous answer
    st.subheader("🔎 Query KB")
    
    query_text = st.text_input("Ask a question:", key="query_input")
    model_choice = st.selectbox("Model", options=[config.DEFAULT_CHAT_MODEL], index=0, key="model_select")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=config.DEFAULT_RAG_TOP_K, key="topk_slider")

    answer_placeholder = st.empty()
    sources_placeholder = st.empty()
    
    # Layout: Button [Get Answer] | Toggle [Read Aloud]
    col_btn, col_tts = st.columns([0.4, 0.6])
    
    with col_btn:
        run_pressed = st.button("Get Answer", type="primary") # primary for emphasis
    
    with col_tts:
        # Initialize state if not present
        if "tts_enabled" not in st.session_state:
            st.session_state["tts_enabled"] = True
            
        tts_enabled = st.checkbox("🔊 Read Answer", value=st.session_state["tts_enabled"], key="tts_cb")
        st.session_state["tts_enabled"] = tts_enabled

    if run_pressed:
        if kb_choice in (None, "<no KBs>"):
            st.warning("Select or create a KB first.")
        elif not query_text or not query_text.strip():
            st.warning("Type a question to query the KB.")
        else:
            # skeleton loader
            answer_placeholder.markdown('<div class="skeleton" style="width:100%"></div>', unsafe_allow_html=True)
            sources_placeholder.markdown('<div class="skeleton" style="width:70%"></div>', unsafe_allow_html=True)
            
            with st.spinner("Retrieving and generating answer..."):
                kb = kb_manager.get_kb(kb_choice)
                try:
                    answer, sources = answer_query(query_text, kb, top_k=top_k)
                    # Persist in session state
                    st.session_state["last_answer"] = answer
                    st.session_state["last_sources"] = sources
                    # Clear any old audio if new question
                    st.session_state.pop("last_audio", None) 
                except Exception as e:
                    st.session_state["last_answer"] = f"[Error while answering: {e}]"
                    st.session_state["last_sources"] = []

            safe_rerun()

    # Display Result (from session state)
    if "last_answer" in st.session_state:
        answer = st.session_state["last_answer"]
        sources = st.session_state["last_sources"]
        
        answer_placeholder.empty()
        sources_placeholder.empty()

        st.markdown("### Answer")
        st.markdown(f"<div class='card' style='padding:12px'>{answer}</div>", unsafe_allow_html=True)
        
        # TTS Logic
        if st.session_state["tts_enabled"]:
            from services.tts import text_to_speech
            
            # Check if we already generated audio for this EXACT answer
            if "last_audio" not in st.session_state:
                with st.spinner("Generating speech..."):
                    audio_data = text_to_speech(answer)
                    if audio_data:
                        st.session_state["last_audio"] = audio_data
            
            # Play if we have audio
            if "last_audio" in st.session_state:
                st.audio(st.session_state["last_audio"], format="audio/mp3", autoplay=True)

        st.markdown("### Sources")
        if sources:
            for s in sources:
                score = s.get("score", None)
                text = s.get("text", "")
                title = s.get("title", "")
                st.markdown(f"<span class='chip kb'>KB: {title or 'untitled'} • {score:.3f}</span>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top:8px;padding:10px;border-radius:8px;background:rgba(255,255,255,0.02)'>{text[:400]}...</div>", unsafe_allow_html=True)
        else:
            st.write("No sources found.")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer: show any one-time message saved late and tips
show_one_time_message()
st.markdown("<div class='small-muted' style='margin-top:8px'>Tip: For persistent use, export OPENAI_API_KEY in your shell before starting Streamlit. Use the sidebar to paste one for this session.</div>", unsafe_allow_html=True)
