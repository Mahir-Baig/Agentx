# """
# Streamlit RAG Application
# Upload documents, process through RAG pipeline, and chat with your documents
# Includes voice input (STT) and read aloud (TTS) features
# """

# # Force UTF-8 encoding for Windows console (must be first!)
# import os
# import sys
# if sys.platform == "win32":
#     os.environ['PYTHONIOENCODING'] = 'utf-8'

# import streamlit as st
# import uuid
# import base64
# from src.pipelines.document_pipeline import DocumentPipeline
# from src.services.agent_service import stream_rag_agent
# from src.services.stt import get_stt_service
# from src.services.tts import get_tts_service
# from src.logger import logger

# # Page configuration
# st.set_page_config(
#     page_title="AgentX",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for modern dark theme
# st.markdown("""
# <style>
#     /* Main background - dark gradient */
#     [data-testid="stAppViewContainer"] {
#         background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
#     }
    
#     /* Sidebar - dark theme */
#     [data-testid="stSidebar"] {
#         background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
#         border-right: 1px solid #e94560;
#     }
    
#     /* Hide default Streamlit elements */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
    
#     /* Chat messages styling */
#     .stChatMessage {
#         padding: 1.5rem;
#         border-radius: 0.75rem;
#         margin-bottom: 1rem;
#         background-color: rgba(255, 255, 255, 0.05);
#     }
    
#     /* Button styling */
#     .stButton > button {
#         width: 100%;
#         border-radius: 0.5rem;
#         font-weight: 600;
#         transition: all 0.3s ease;
#     }
    
#     /* Make sidebar buttons equal width */
#     [data-testid="column"] {
#         width: 50% !important;
#         flex: 1 1 50% !important;
#     }
    
#     /* Chat input styling */
#     .stChatInput {
#         border: 1px solid #e94560 !important;
#         border-radius: 12px !important;
#     }
    
#     /* Typing cursor animation */
#     @keyframes blink {
#         0%, 50% { opacity: 1; }
#         51%, 100% { opacity: 0; }
#     }
    
#     .typing-cursor {
#         animation: blink 1s infinite;
#     }
    
#     /* Reduce padding */
#     .block-container {
#         padding-top: 2rem;
#         padding-bottom: 2rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if "thread_id" not in st.session_state:
#     st.session_state.thread_id = str(uuid.uuid4())
#     logger.info(f"New session started with thread_id: {st.session_state.thread_id}")

# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
#     logger.info("Chat history initialized")

# if "pipeline" not in st.session_state:
#     logger.info("Initializing document pipeline...")
#     with st.spinner("Initializing pipeline..."):
#         st.session_state.pipeline = DocumentPipeline()
#     logger.info("Pipeline initialized in session state")

# if "stt_service" not in st.session_state:
#     logger.info("Initializing Speech-to-Text service...")
#     st.session_state.stt_service = get_stt_service()
#     logger.info("STT service initialized")

# if "tts_service" not in st.session_state:
#     logger.info("Initializing Text-to-Speech service...")
#     st.session_state.tts_service = get_tts_service()
#     logger.info("TTS service initialized")

# if "recording" not in st.session_state:
#     st.session_state.recording = False
#     logger.info("Recording state initialized")

# # Sidebar
# with st.sidebar:
#     st.markdown("## 🤖 AgentX")
#     st.markdown("Your AI Knowledge Assistant")
    
#     # Session info
#     st.markdown("### 📋 Session")
#     st.code(st.session_state.thread_id[:12] + "...", language="text")
    
#     st.divider()
    
#     # Document upload
#     st.markdown("### 📄 Upload Document")
#     uploaded_file = st.file_uploader(
#         "Drop file here",
#         type=["pdf", "txt"],
#         help="PDF, TXT • Max 200MB",
#         label_visibility="collapsed"
#     )
    
#     if uploaded_file is not None:
#         logger.info(f"📄 File selected: {uploaded_file.name} ({uploaded_file.size} bytes)")
#         st.write(f"**{uploaded_file.name}**")
#         st.write(f"{uploaded_file.size / 1024:.1f} KB")
        
#         if st.button("🚀 Process", type="primary", use_container_width=True):
#             logger.info(f"Processing file: {uploaded_file.name}")
#             with st.spinner("Processing..."):
#                 progress_bar = st.progress(0)
#                 status_text = st.empty()
                
#                 file_content = uploaded_file.read()
#                 logger.info(f"File read: {len(file_content)} bytes")
                
#                 progress_bar.progress(20)
#                 status_text.text("Uploading...")
#                 logger.info("Stage 1: Uploading to blob...")
                
#                 progress_bar.progress(60)
#                 status_text.text("Processing...")
#                 logger.info("Stage 2: Processing through pipeline...")
                
#                 success, message, final_container = st.session_state.pipeline.handle_uploaded_file(
#                     uploaded_file, 
#                     file_content
#                 )
                
#                 progress_bar.progress(100)
#                 status_text.empty()
#                 logger.info(f"Processing complete - Success: {success}, Container: {final_container}")
                
#                 if success:
#                     st.success("✅ Processed!")
#                     logger.info(f"SUCCESS: File processed successfully - {uploaded_file.name}")
#                 else:
#                     if final_container == "rejected":
#                         st.warning("⚠️ Duplicate file")
#                         logger.warning(f"DUPLICATE: File rejected - {uploaded_file.name}")
#                     else:
#                         st.error("❌ Error")
#                         logger.error(f"ERROR: Processing failed - {uploaded_file.name} - {message}")
                
#                 progress_bar.empty()
    
#     st.divider()
    
#     # Actions - Equal width buttons
#     col1, col2 = st.columns(2)
    
#     with col1:
#         if st.button("➕ New Chat", use_container_width=True, type="primary"):
#             logger.info(f"New chat button clicked - Old thread: {st.session_state.thread_id[:8]}")
#             st.session_state.thread_id = str(uuid.uuid4())
#             st.session_state.chat_history = []
#             logger.info(f"New chat created - New thread: {st.session_state.thread_id[:8]}")
#             st.rerun()
    
#     with col2:
#         if st.button("🗑️ Clear", use_container_width=True):
#             logger.info(f"Clear history button clicked - {len(st.session_state.chat_history)} messages")
#             st.session_state.chat_history = []
#             logger.info("Chat history cleared")
#             st.rerun()
    
#     st.divider()
    
#     # Info footer
#     st.markdown("### ℹ️ How it works")
#     st.markdown("""
#     1. 📄 Upload documents
#     2. 🔍 AI processes content
#     3. ❓ Ask questions
#     4. ✨ Get instant answers
#     """)

# # Main chat area
# # Display welcome message if no messages
# if len(st.session_state.chat_history) == 0:
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         st.markdown("""
#         <div style="text-align: center; padding: 3rem;">
#             <h1 style="font-size: 3rem; margin-bottom: 1rem; color: #fff;">🤖 AgentX</h1>
#             <p style="font-size: 1.2rem; color: #888; margin-bottom: 2rem;">
#                 Upload a document and start chatting with your knowledge base,
#                 or simply ask anything you'd like.
#             </p>
#             <div style="display: flex; gap: 2rem; justify-content: center; color: #666; font-size: 0.9rem;">
#                 <div>🎤 Voice input</div>
#                 <div>🔊 Text-to-speech</div>
#                 <div>⚡ Real-time answers</div>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
# else:
#     logger.info(f"Displaying {len(st.session_state.chat_history)} messages")
    
#     # Display chat messages
#     for idx, message in enumerate(st.session_state.chat_history):
#         with st.chat_message(message["role"]):
#             # Render markdown with unsafe_allow_html to support links
#             st.markdown(message["content"], unsafe_allow_html=True)
            
#             # Add "Read Aloud" button for assistant messages
#             if message["role"] == "assistant":
#                 if st.button("🔊 Read Aloud", key=f"tts_{idx}"):
#                     logger.info(f"TTS button clicked for message {idx}")
#                     logger.info(f"Text length: {len(message['content'])} chars")
#                     with st.spinner("Generating audio..."):
#                         success, audio_bytes, msg = st.session_state.tts_service.synthesize_to_bytes(message["content"])
#                         if success and audio_bytes:
#                             logger.info(f"TTS audio generated - Size: {len(audio_bytes)} bytes")
#                             # Encode audio to base64 for HTML audio player
#                             audio_b64 = base64.b64encode(audio_bytes).decode()
#                             audio_html = f"""
#                                 <audio controls autoplay>
#                                     <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
#                                     Your browser does not support the audio element.
#                                 </audio>
#                             """
#                             st.markdown(audio_html, unsafe_allow_html=True)
#                             logger.info("Audio player rendered")
#                         else:
#                             logger.error(f"TTS failed: {msg}")
#                             st.error(f"{msg}")

# # Voice input section
# st.divider()

# user_query = None

# # Create layout: microphone on left, text input on right
# col_mic, col_input = st.columns([1, 11])

# with col_mic:
#     # Microphone button - toggle recording
#     if not st.session_state.recording:
#         # Show Start Recording button
#         if st.button("🎤 Start", help="Click to start speaking", use_container_width=True, key="start_rec"):
#             logger.info("Start recording button clicked")
#             success, msg = st.session_state.stt_service.start_continuous_recognition()
#             if success:
#                 st.session_state.recording = True
#                 logger.info("Recording started")
#                 st.rerun()
#             else:
#                 st.error(f"Failed to start: {msg}")
#     else:
#         # Show Stop Recording button (RED)
#         if st.button("⏹️ Stop", help="Click to stop recording", use_container_width=True, type="primary", key="stop_rec"):
#             logger.info("Stop recording button clicked")
#             success, text = st.session_state.stt_service.stop_continuous_recognition()
#             st.session_state.recording = False
            
#             if success:
#                 logger.info(f"STT SUCCESS: {text}")
#                 user_query = text
#                 st.success(f"You said: {text}")
#             else:
#                 logger.error(f"STT FAILED: {text}")
#                 st.error(f"{text}")
        
#         # Show recording indicator
#         st.markdown("🔴 **Recording...**")

# with col_input:
#     # Text input - only show if no voice input
#     if user_query is None:
#         text_input = st.chat_input("Type your message or use 🎤 microphone...")
#         if text_input:
#             user_query = text_input
#             logger.info(f"Text input received: '{text_input[:50]}...'")

# # Process user query with STREAMING - FIXED
# if user_query:
#     logger.info(f"New query received: '{user_query[:100]}...'")
#     logger.info(f"Thread ID: {st.session_state.thread_id[:8]}")
    
#     # Add user message to history FIRST
#     st.session_state.chat_history.append({
#         "role": "user",
#         "content": user_query
#     })
#     logger.info(f"User message added to history - Total messages: {len(st.session_state.chat_history)}")
    
#     # Display the user message immediately
#     with st.chat_message("user"):
#         st.markdown(user_query)
    
#     # Show assistant thinking with streaming
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""
        
#         try:
#             logger.info("Calling RAG agent with streaming...")
            
#             # Stream response from RAG agent
#             for chunk in stream_rag_agent(
#                 query=user_query,
#                 thread_id=st.session_state.thread_id
#             ):
#                 # Check for errors
#                 if isinstance(chunk, dict) and "error" in chunk:
#                     error_msg = f"❌ Error: {chunk['error']}"
#                     message_placeholder.error(error_msg)
#                     full_response = error_msg
#                     logger.error(f"Streaming error: {chunk['error']}")
#                     break
                
#                 # Extract text from LangGraph chunk
#                 if isinstance(chunk, dict):
#                     # Check for 'model' key (AI response)
#                     if 'model' in chunk and 'messages' in chunk['model']:
#                         messages = chunk['model']['messages']
#                         for msg in messages:
#                             if hasattr(msg, 'content') and msg.content:
#                                 # Check if this is the final answer (not tool call)
#                                 if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
#                                     full_response = msg.content
#                                     # Update with typing cursor
#                                     message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                    
#                     # Check for 'tools' key (tool output) - optional display
#                     elif 'tools' in chunk and 'messages' in chunk['tools']:
#                         # This is the RAG tool response, you can optionally show it
#                         logger.info("Tool response received (RAG search complete)")
#                 else:
#                     # If it's a plain string chunk (shouldn't happen with LangGraph but just in case)
#                     full_response += str(chunk)
#                     message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
#             # Final update without cursor
#             if full_response:
#                 message_placeholder.markdown(full_response, unsafe_allow_html=True)
#                 logger.info(f"RAG SUCCESS - Response length: {len(full_response)} chars")
                
#                 # Add assistant response to history
#                 st.session_state.chat_history.append({
#                     "role": "assistant",
#                     "content": full_response
#                 })
                
#                 logger.info(f"Query processed successfully - Total messages: {len(st.session_state.chat_history)}")
#             else:
#                 error_msg = "No response generated"
#                 message_placeholder.error(error_msg)
#                 st.session_state.chat_history.append({
#                     "role": "assistant",
#                     "content": error_msg
#                 })
            
#         except Exception as e:
#             error_msg = f"Error: {str(e)}"
#             logger.error(f"EXCEPTION in query: {str(e)}", exc_info=True)
#             message_placeholder.error(error_msg)
#             st.session_state.chat_history.append({
#                 "role": "assistant",
#                 "content": error_msg
#             })
    
#     # Rerun to update chat display with Read Aloud buttons
#     logger.info("Rerunning app to update chat display")
#     st.rerun()

# # Footer - Below input
# st.divider()
# st.markdown("""
# <div style="text-align: center; color: #666; font-size: 0.85rem; padding: 1rem;">
#     <p>Made with ❤️ by Mahir Baig | AgentX v0.0.1</p>
#     <p>Session ID: <code>{}</code></p>
# </div>
# """.format(st.session_state.thread_id[:12] + "..."), unsafe_allow_html=True)



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
from src.services.agent_service import stream_rag_agent
from src.services.stt import get_stt_service
from src.services.stt_browser import get_browser_stt_service
from src.services.tts import get_tts_service
from src.logger import logger

# Page configuration
st.set_page_config(
    page_title="AgentX",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
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

# Sidebar
with st.sidebar:
    st.markdown("## 🤖 AgentX")
    st.markdown("Your AI Knowledge Assistant")
    
    # Session info
    st.markdown("### 📋 Session")
    st.code(st.session_state.thread_id[:12] + "...", language="text")
    
    st.divider()
    
    # Document upload
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
    
    # Actions - Equal width buttons
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
    
    # Info footer
    st.markdown("### ℹ️ How it works")
    st.markdown("""
    1. 📄 Upload documents
    2. 🔍 AI processes content
    3. ❓ Ask questions
    4. ✨ Get instant answers
    """)

# Main chat area
# Display welcome message if no messages
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
                <div>🔊 Text-to-speech</div>
                <div>⚡ Real-time answers</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    logger.info(f"Displaying {len(st.session_state.chat_history)} messages")
    
    # Display chat messages
    for idx, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            # Render markdown with unsafe_allow_html to support links
            st.markdown(message["content"], unsafe_allow_html=True)
            
            # Add "Read Aloud" button for assistant messages
            if message["role"] == "assistant":
                if st.button("🔊 Read Aloud", key=f"tts_{idx}"):
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

# 1. Text Input Section - Rendered first
user_query = None

def submit_text():
    st.session_state.query_to_process = st.session_state.widget_input
    st.session_state.widget_input = ""

if "query_to_process" not in st.session_state:
    st.session_state.query_to_process = None

# Text input bar
st.text_input("Message", key="widget_input", on_change=submit_text, 
              label_visibility="collapsed", placeholder="Type your message...")

# 2. Voice Input Section - Rendered below text input
audio_bytes = st.audio_input("Voice Input", key="audio_recorder")

# Check for text submission
if st.session_state.query_to_process:
    user_query = st.session_state.query_to_process
    st.session_state.query_to_process = None # Reset

# Check for audio submission
if audio_bytes:
    # Prevent re-processing the same audio
    if "last_audio_id" not in st.session_state:
        st.session_state.last_audio_id = None
        
    # Create unique ID for this audio
    current_audio_id = f"{len(audio_bytes.getvalue())}_{audio_bytes.getvalue()[:10]}"
    
    if current_audio_id != st.session_state.last_audio_id:
        st.session_state.last_audio_id = current_audio_id
        logger.info(f"Audio received: {len(audio_bytes.getvalue())} bytes")
        
        with st.spinner("Processing your voice..."):
            # Use browser STT service
            success, text = st.session_state.browser_stt_service.recognize_from_file(audio_bytes)
            
            if success:
                logger.info(f"STT SUCCESS: {text}")
                user_query = text
                st.success(f"✅ You said: **{text}**")
            else:
                logger.error(f"STT FAILED: {text}")
                st.error(f"❌ {text}")

# Process user query with STREAMING - FIXED
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
    
    # Show assistant thinking with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            logger.info("Calling RAG agent with streaming...")
            
            # Stream response from RAG agent
            for chunk in stream_rag_agent(
                query=user_query,
                thread_id=st.session_state.thread_id
            ):
                # Check for errors
                if isinstance(chunk, dict) and "error" in chunk:
                    error_msg = f"❌ Error: {chunk['error']}"
                    message_placeholder.error(error_msg)
                    full_response = error_msg
                    logger.error(f"Streaming error: {chunk['error']}")
                    break
                
                # Extract text from LangGraph chunk
                if isinstance(chunk, dict):
                    # Check for 'model' key (AI response)
                    if 'model' in chunk and 'messages' in chunk['model']:
                        messages = chunk['model']['messages']
                        for msg in messages:
                            if hasattr(msg, 'content') and msg.content:
                                # Check if this is the final answer (not tool call)
                                if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                                    full_response = msg.content
                                    # Update with typing cursor
                                    message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                    
                    # Check for 'tools' key (tool output) - optional display
                    elif 'tools' in chunk and 'messages' in chunk['tools']:
                        # This is the RAG tool response, you can optionally show it
                        logger.info("Tool response received (RAG search complete)")
                else:
                    # If it's a plain string chunk (shouldn't happen with LangGraph but just in case)
                    full_response += str(chunk)
                    message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
            # Final update without cursor
            if full_response:
                message_placeholder.markdown(full_response, unsafe_allow_html=True)
                logger.info(f"RAG SUCCESS - Response length: {len(full_response)} chars")
                
                # Add assistant response to history
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
    
    # Rerun to update chat display with Read Aloud buttons
    logger.info("Rerunning app to update chat display")
    st.rerun()

# Footer - Below input
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.85rem; padding: 1rem;">
    <p>Made with ❤️ by Mahir Baig | AgentX v0.0.1</p>
    <p>Session ID: <code>{}</code></p>
</div>
""".format(st.session_state.thread_id[:12] + "..."), unsafe_allow_html=True)