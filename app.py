"""
Streamlit RAG Application
Upload documents, process through RAG pipeline, and chat with your documents
Includes voice input (STT) and read aloud (TTS) features
"""

# Force UTF-8 encoding for Windows console (must be first!)
import os
import sys
if sys.platform == "win32":
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import streamlit as st
import uuid
import base64
from src.pipelines.document_pipeline import DocumentPipeline
from src.services.agent_service import query_rag_agent
from src.services.stt import get_stt_service
from src.services.tts import get_tts_service
from src.logger import logger

# Page configuration
st.set_page_config(
    page_title="Agentx",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - ChatGPT-like interface
st.markdown("""
    <style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main content area */
    .main > div {
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem 1rem 8rem 1rem;
    }
    
    /* Reduce top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 8rem;
    }
    
    /* Style microphone button area - RED */
    .stButton button {
        background-color: #ef4444 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stButton button:hover {
        background-color: #dc2626 !important;
    }
    
    /* Microphone icon with red background */
    .mic-icon {
        background-color: #ef4444;
        color: white;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        cursor: pointer;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
        transition: all 0.3s ease;
    }
    
    .mic-icon:hover {
        background-color: #dc2626;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.5);
        transform: scale(1.05);
    }
    
    /* Input area fixed at bottom - ChatGPT style */
    .input-section {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(to top, rgba(255,255,255,1) 0%, rgba(255,255,255,1) 85%, rgba(255,255,255,0) 100%);
        padding: 1.5rem 1rem 2rem 1rem;
        z-index: 1000;
        display: flex;
        justify-content: center;
    }
    
    /* Center the input container */
    .input-section > div {
        max-width: 900px;
        width: 100%;
        padding: 0 1rem;
    }
    
    /* Input row centered */
    .input-section [data-testid="stHorizontalBlock"] {
        justify-content: center;
    }
    
    /* Hide separator line */
    .input-section hr {
        display: none;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .input-section {
            background: linear-gradient(to top, rgba(31,41,55,1) 0%, rgba(31,41,55,1) 85%, rgba(31,41,55,0) 100%);
        }
    }
    
    /* Remove borders from chat input */
    .stChatInput {
        border: 1px solid #d1d5db !important;
        border-radius: 12px !important;
    }
    
    /* Hide streamlit branding and unnecessary containers */
    div[data-testid="stDecoration"] {
        display: none;
    }
    
    /* Hide any blue outlined boxes */
    div[data-testid="column"] {
        border: none !important;
        outline: none !important;
    }
    
    /* Remove borders from all container elements */
    .element-container {
        border: none !important;
    }
    
    /* Hide any visible container outlines */
    [data-testid="stVerticalBlock"] > div {
        border: none !important;
        outline: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    logger.info(f"New session started with thread_id: {st.session_state.thread_id}")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    logger.info("Chat history initialized")

if "pipeline" not in st.session_state:
    logger.info("Initializing document pipeline...")
    with st.spinner("Initializing pipeline..."):
        st.session_state.pipeline = DocumentPipeline()
    logger.info("Pipeline initialized in session state")

if "stt_service" not in st.session_state:
    logger.info("Initializing Speech-to-Text service...")
    st.session_state.stt_service = get_stt_service()
    logger.info("STT service initialized")

if "tts_service" not in st.session_state:
    logger.info("Initializing Text-to-Speech service...")
    st.session_state.tts_service = get_tts_service()
    logger.info("TTS service initialized")

if "audio_playing" not in st.session_state:
    st.session_state.audio_playing = False
    logger.info("Audio state initialized")

if "recording" not in st.session_state:
    st.session_state.recording = False
    logger.info("Recording state initialized")

# App title - Centered like ChatGPT
st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 40vh; text-align: center;">
        <h1 style="font-size: 3.5rem; font-weight: 700; margin-bottom: 0.5rem;">🤖Agentx</h1>
        <p style="font-size: 1.1rem; color: #6b7280; margin-top: 0.5rem;">Upload a document and start chatting with your knowledge base</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Session Info")
    st.info(f"**Thread ID:** `{st.session_state.thread_id[:8]}...`")
    
    st.markdown("---")
    
    # Compact file upload in sidebar
    st.header("� Upload Document")
    uploaded_file = st.file_uploader(
        "PDF or TXT files",
        type=["pdf", "txt"],
        help="Upload PDF or TXT files only",
        key="file_uploader",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        logger.info(f"📄 File selected: {uploaded_file.name} ({uploaded_file.size} bytes)")
        st.write(f"**{uploaded_file.name}**")
        st.write(f"{uploaded_file.size / 1024:.1f} KB")
        
        if st.button("🚀 Process", type="primary", use_container_width=True):
            logger.info(f"Processing file: {uploaded_file.name}")
            with st.spinner("Processing..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                file_content = uploaded_file.read()
                logger.info(f"File read: {len(file_content)} bytes")
                
                progress_bar.progress(20)
                status_text.text("Uploading...")
                logger.info("Stage 1: Uploading to blob...")
                
                progress_bar.progress(60)
                status_text.text("Processing...")
                logger.info("Stage 2: Processing through pipeline...")
                
                success, message, final_container = st.session_state.pipeline.handle_uploaded_file(
                    uploaded_file, 
                    file_content
                )
                
                progress_bar.progress(100)
                status_text.empty()
                logger.info(f"Processing complete - Success: {success}, Container: {final_container}")
                
                if success:
                    st.success("✅ Processed!")
                    st.balloons()
                    logger.info(f"SUCCESS: File processed successfully - {uploaded_file.name}")
                else:
                    if final_container == "rejected":
                        st.warning("⚠️ Duplicate file")
                        logger.warning(f"DUPLICATE: File rejected - {uploaded_file.name}")
                    else:
                        st.error("❌ Error")
                        logger.error(f"ERROR: Processing failed - {uploaded_file.name} - {message}")
                
                progress_bar.empty()
    
    st.markdown("---")
    
    if st.button("🔄 New Chat", use_container_width=True):
        logger.info(f"New chat button clicked - Old thread: {st.session_state.thread_id[:8]}")
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        logger.info(f"New chat created - New thread: {st.session_state.thread_id[:8]}")
        st.rerun()
    
    if st.button("🗑️ Clear History", use_container_width=True):
        logger.info(f"Clear history button clicked - {len(st.session_state.chat_history)} messages")
        st.session_state.chat_history = []
        logger.info("Chat history cleared")
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    **ℹ️ How it works:**
    1. Upload PDF/TXT
    2. Duplicate check
    3. RAG processing
    4. Chat with docs
    """)

# Main chat area - Display chat history with Read Aloud buttons
if len(st.session_state.chat_history) == 0:
    logger.info("Chat history empty - showing welcome message")
    st.info("👋 Welcome! Upload a document and start chatting with your knowledge base or simply ask anything you like. 🎤 Use voice or type!")
else:
    logger.info(f"Displaying {len(st.session_state.chat_history)} messages")

for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        # Render markdown with unsafe_allow_html to support links
        st.markdown(message["content"], unsafe_allow_html=True)
        
        # Add "Read Aloud" button for assistant messages
        if message["role"] == "assistant":
            col1, col2 = st.columns([1, 10])
            with col1:
                if st.button("🔊", key=f"tts_{idx}", help="Read Aloud"):
                    logger.info(f"TTS button clicked for message {idx}")
                    logger.info(f"Text length: {len(message['content'])} chars")
                    with st.spinner("Generating audio..."):
                        success, audio_bytes, msg = st.session_state.tts_service.synthesize_to_bytes(message["content"])
                        if success and audio_bytes:
                            logger.info(f"TTS audio generated - Size: {len(audio_bytes)} bytes")
                            # Encode audio to base64 for HTML audio player
                            audio_b64 = base64.b64encode(audio_bytes).decode()
                            audio_html = f"""
                                <audio controls autoplay>
                                    <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
                                    Your browser does not support the audio element.
                                </audio>
                            """
                            st.markdown(audio_html, unsafe_allow_html=True)
                            logger.info("Audio player rendered")
                        else:
                            logger.error(f"TTS failed: {msg}")
                            st.error(f"{msg}")

# Input Section at Bottom (ChatGPT-style) - Fixed position
st.markdown('<div class="input-section"><div>', unsafe_allow_html=True)

user_query = None

# Create layout: microphone on left, text input on right
col_mic, col_input = st.columns([1, 11])

with col_mic:
    # Microphone button - toggle recording
    if not st.session_state.recording:
        # Show Start Recording button
        if st.button("🎤 Start", help="Click to start speaking", use_container_width=True, key="start_rec"):
            logger.info("Start recording button clicked")
            success, msg = st.session_state.stt_service.start_continuous_recognition()
            if success:
                st.session_state.recording = True
                logger.info("Recording started")
                st.rerun()
            else:
                st.error(f"Failed to start: {msg}")
    else:
        # Show Stop Recording button (RED)
        if st.button("⏹️ Stop", help="Click to stop recording", use_container_width=True, type="primary", key="stop_rec"):
            logger.info("Stop recording button clicked")
            success, text = st.session_state.stt_service.stop_continuous_recognition()
            st.session_state.recording = False
            
            if success:
                logger.info(f"STT SUCCESS: {text}")
                user_query = text
                st.success(f"You said: {text}")
            else:
                logger.error(f"STT FAILED: {text}")
                st.error(f"{text}")
        
        # Show recording indicator
        st.markdown("🔴 **Recording...**")

with col_input:
    # Text input - only show if no voice input
    if user_query is None:
        text_input = st.chat_input("Type your message or use 🎤 microphone...")
        if text_input:
            user_query = text_input
            logger.info(f"Text input received: '{text_input[:50]}...'")

st.markdown('</div></div>', unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

if user_query:
    logger.info(f"New query received: '{user_query[:100]}...'")
    logger.info(f"Thread ID: {st.session_state.thread_id[:8]}")
    
    # Add user message to history FIRST
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })
    logger.info(f"User message added to history - Total messages: {len(st.session_state.chat_history)}")
    
    # Display the user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Show assistant thinking with spinner
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                logger.info("Calling RAG agent...")
                # Query the RAG agent with thread_id
                result = query_rag_agent(
                    query=user_query,
                    thread_id=st.session_state.thread_id,
                    include_metadata=False
                )
                
                if result["success"]:
                    response = result["response"]
                    logger.info(f"RAG SUCCESS - Response length: {len(response)} chars")
                    logger.info(f"Response preview: '{response[:200]}...'")
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    logger.info(f"Query processed successfully - Total messages: {len(st.session_state.chat_history)}")
                    
                    # Display the response with HTML support for links
                    st.markdown(response, unsafe_allow_html=True)
                else:
                    error_msg = f"Error: {result.get('error', 'Unknown error')}"
                    logger.error(f"RAG FAILED: {result.get('error')}")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    logger.error(f"Query failed: {result.get('error')}")
                    
                    # Display the error
                    st.markdown(error_msg)
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                logger.error(f"EXCEPTION in query: {str(e)}", exc_info=True)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg
                })
                
                # Display the error
                st.markdown(error_msg)
    
    # Rerun to update chat display with Read Aloud buttons
    logger.info("Rerunning app to update chat display")
    st.rerun()