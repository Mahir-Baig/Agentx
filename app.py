"""
Streamlit RAG Application
Upload documents, process through RAG pipeline, and chat with your documents
Includes voice input (STT) and read aloud (TTS) features
"""



import streamlit as st
import uuid
import base64
from src.pipelines.document_pipeline import DocumentPipeline
from src.services.agent_service import stream_rag_agent
from src.services.stt import get_stt_service
from src.services.stt_browser import get_browser_stt_service
from src.services.tts import get_tts_service
from src.logger import logger


st.set_page_config(
    page_title="AgentX",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Main background - dark gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Sidebar - dark theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
        border-right: 1px solid #e94560;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Chat messages styling */
    .stChatMessage {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        background-color: rgba(255, 255, 255, 0.05);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* Make sidebar buttons equal width */
    [data-testid="column"] {
        width: 50% !important;
        flex: 1 1 50% !important;
    }
    
    /* Chat input styling */
    .stChatInput {
        border: 1px solid #e94560 !important;
        border-radius: 12px !important;
    }
    
    /* Typing cursor animation */
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    .typing-cursor {
        animation: blink 1s infinite;
    }
    
    /* Reduce padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Compact Audio Input Bar */
    [data-testid="stAudioInput"] {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        margin-top: 0.5rem !important;
    }
    
    [data-testid="stAudioInput"] > div {
        height: 45px !important; /* Match text input height */
        min-height: 45px !important;
    }
    
    /* Hide the label "Voice Input" to save vertical space */
    [data-testid="stAudioInput"] label {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


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

if "browser_stt_service" not in st.session_state:
    logger.info("Initializing Browser STT service...")
    st.session_state.browser_stt_service = get_browser_stt_service()
    logger.info("Browser STT service initialized")

if "tts_service" not in st.session_state:
    logger.info("Initializing Text-to-Speech service...")
    st.session_state.tts_service = get_tts_service()
    logger.info("TTS service initialized")

if "recording" not in st.session_state:
    st.session_state.recording = False
    logger.info("Recording state initialized")


with st.sidebar:
    st.markdown("## 🤖 AgentX")
    st.markdown("Your AI Knowledge Assistant")
    
    
    st.markdown("### Thread ID")
    st.code(st.session_state.thread_id[:12] + "...", language="text")
    
    st.divider()
    
    
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Drop file here",
        type=["pdf", "txt"],
        help="PDF, TXT • Max 200MB",
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
                    logger.info(f"SUCCESS: File processed successfully - {uploaded_file.name}")
                else:
                    if final_container == "rejected":
                        st.warning("⚠️ Duplicate file")
                        logger.warning(f"DUPLICATE: File rejected - {uploaded_file.name}")
                    else:
                        st.error("❌ Error")
                        logger.error(f"ERROR: Processing failed - {uploaded_file.name} - {message}")
                
                progress_bar.empty()
    
    st.divider()
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            logger.info(f"New chat button clicked - Old thread: {st.session_state.thread_id[:8]}")
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            logger.info(f"New chat created - New thread: {st.session_state.thread_id[:8]}")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            logger.info(f"Clear history button clicked - {len(st.session_state.chat_history)} messages")
            st.session_state.chat_history = []
            logger.info("Chat history cleared")
            st.rerun()
    
    st.divider()
    
   
    st.markdown("### ℹ️ How it works")
    st.markdown("""
    1. 📄 Upload documents
    2. 🔍 AI processes content
    3. ❓ Ask questions
    4. ✨ Get instant answers
    """)


if len(st.session_state.chat_history) == 0:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; color: #fff;">🤖 AgentX</h1>
            <p style="font-size: 1.2rem; color: #888; margin-bottom: 2rem;">
                Upload a document and start chatting with your knowledge base,
                or simply ask anything you'd like.
            </p>
            <div style="display: flex; gap: 2rem; justify-content: center; color: #666; font-size: 0.9rem;">
                <div>🎤 Voice input</div>
                <div>⚡ Real-time answers</div>
                <div>🔊 Text-to-speech</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    logger.info(f"Displaying {len(st.session_state.chat_history)} messages")
    
    
    for idx, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            
            st.markdown(message["content"], unsafe_allow_html=True)
            
            
            if message["role"] == "assistant":
                if st.button("🔊 Read Aloud", key=f"tts_{idx}"):
                    logger.info(f"TTS button clicked for message {idx}")
                    logger.info(f"Text length: {len(message['content'])} chars")
                    with st.spinner("Generating audio..."):
                        success, audio_bytes, msg = st.session_state.tts_service.synthesize_to_bytes(message["content"])
                        if success and audio_bytes:
                            logger.info(f"TTS audio generated - Size: {len(audio_bytes)} bytes")
                            
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


user_query = None

def submit_text():
    st.session_state.query_to_process = st.session_state.widget_input
    st.session_state.widget_input = ""

if "query_to_process" not in st.session_state:
    st.session_state.query_to_process = None


st.text_input("Message", key="widget_input", on_change=submit_text, 
              label_visibility="collapsed", placeholder="Type your message...")


audio_bytes = st.audio_input("Voice Input", key="audio_recorder")


if st.session_state.query_to_process:
    user_query = st.session_state.query_to_process
    st.session_state.query_to_process = None # Reset


if audio_bytes:
    
    if "last_audio_id" not in st.session_state:
        st.session_state.last_audio_id = None
        
   
    current_audio_id = f"{len(audio_bytes.getvalue())}_{audio_bytes.getvalue()[:10]}"
    
    if current_audio_id != st.session_state.last_audio_id:
        st.session_state.last_audio_id = current_audio_id
        logger.info(f"Audio received: {len(audio_bytes.getvalue())} bytes")
        
        with st.spinner("Processing your voice..."):
           
            success, text = st.session_state.browser_stt_service.recognize_from_file(audio_bytes)
            
            if success:
                logger.info(f"STT SUCCESS: {text}")
                user_query = text
                st.success(f"✅ You said: **{text}**")
            else:
                logger.error(f"STT FAILED: {text}")
                st.error(f"❌ {text}")


if user_query:
    logger.info(f"New query received: '{user_query[:100]}...'")
    logger.info(f"Thread ID: {st.session_state.thread_id[:8]}")
    
    
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })
    logger.info(f"User message added to history - Total messages: {len(st.session_state.chat_history)}")
    
    
    with st.chat_message("user"):
        st.markdown(user_query)
    
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            logger.info("Calling RAG agent with streaming...")
            
            
            for chunk in stream_rag_agent(
                query=user_query,
                thread_id=st.session_state.thread_id
            ):
               
                if isinstance(chunk, dict) and "error" in chunk:
                    error_msg = f"❌ Error: {chunk['error']}"
                    message_placeholder.error(error_msg)
                    full_response = error_msg
                    logger.error(f"Streaming error: {chunk['error']}")
                    break
                
                
                if isinstance(chunk, dict):
                    
                    if 'model' in chunk and 'messages' in chunk['model']:
                        messages = chunk['model']['messages']
                        for msg in messages:
                            if hasattr(msg, 'content') and msg.content:
                                
                                if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                                    full_response = msg.content
                                   
                                    message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                    
                    
                    elif 'tools' in chunk and 'messages' in chunk['tools']:
                        
                        logger.info("Tool response received (RAG search complete)")
                else:
                    
                    full_response += str(chunk)
                    message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
            
            if full_response:
                message_placeholder.markdown(full_response, unsafe_allow_html=True)
                logger.info(f"RAG SUCCESS - Response length: {len(full_response)} chars")
                
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                
                logger.info(f"Query processed successfully - Total messages: {len(st.session_state.chat_history)}")
            else:
                error_msg = "No response generated"
                message_placeholder.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg
                })
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"EXCEPTION in query: {str(e)}", exc_info=True)
            message_placeholder.error(error_msg)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg
            })
    

    logger.info("Rerunning app to update chat display")
    st.rerun()


st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.85rem; padding: 1rem;">
    <p>AgentX v0.0.1</p>
    <p>Thread ID: <code>{}</code></p>
</div>
""".format(st.session_state.thread_id[:12] + "..."), unsafe_allow_html=True)