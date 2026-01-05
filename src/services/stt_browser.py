"""
Speech-to-Text Service using Azure Cognitive Services
Browser-compatible version that works with deployed Streamlit apps
Uses audio stream from browser instead of server-side microphone
"""

import os
import io
from typing import Tuple, Optional
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from src.logger import logger

load_dotenv()


class BrowserSTTService:
    """Azure Speech-to-Text service for browser audio"""
    
    def __init__(self):
        """Initialize Azure Speech configuration"""
        self.speech_key = os.getenv('SPEECH_KEY')
        self.speech_endpoint = os.getenv('SPEECH_ENDPOINT')
        
        if not self.speech_key or not self.speech_endpoint:
            logger.warning("Azure Speech credentials not configured")
            self.speech_config = None
        else:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                endpoint=self.speech_endpoint
            )
            self.speech_config.speech_recognition_language = "en-US"
            logger.info("Browser STT initialized")
    
    def recognize_from_audio_bytes(self, audio_bytes: bytes) -> Tuple[bool, str]:
        """
        Recognize speech from audio bytes (from browser recording)
        
        Args:
            audio_bytes: Audio data in WAV format from browser
            
        Returns:
            Tuple of (success: bool, text: str)
        """
        try:
            if not self.speech_config:
                return False, "Set SPEECH_KEY and SPEECH_ENDPOINT"
            
            if not audio_bytes:
                return False, "No audio data provided"
            
            logger.info(f"Processing audio bytes: {len(audio_bytes)} bytes")
            
            # Create audio stream from bytes
            audio_stream = speechsdk.audio.PushAudioInputStream()
            audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
            
            # Create recognizer
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # Push audio data to stream
            audio_stream.write(audio_bytes)
            audio_stream.close()
            
            # Recognize
            logger.info("Starting speech recognition...")
            result = speech_recognizer.recognize_once_async().get()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                logger.info(f"Recognized: {result.text}")
                return True, result.text
            
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning(f"No speech: {result.no_match_details}")
                return False, "No speech detected in audio"
            
            elif result.reason == speechsdk.ResultReason.Canceled:
                details = result.cancellation_details
                logger.error(f"Canceled: {details.reason}")
                if details.reason == speechsdk.CancellationReason.Error:
                    logger.error(f"Error: {details.error_details}")
                    return False, f"Recognition error: {details.error_details}"
                return False, "Recognition canceled"
            
            return False, "Unknown error"
            
        except Exception as e:
            logger.error(f"STT error: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def recognize_from_file(self, audio_file) -> Tuple[bool, str]:
        """
        Recognize speech from uploaded audio file
        
        Args:
            audio_file: File-like object (e.g., from st.file_uploader)
            
        Returns:
            Tuple of (success: bool, text: str)
        """
        try:
            # Read file content
            audio_bytes = audio_file.read()
            
            # Reset file pointer if needed
            if hasattr(audio_file, 'seek'):
                audio_file.seek(0)
            
            return self.recognize_from_audio_bytes(audio_bytes)
            
        except Exception as e:
            logger.error(f"Error reading audio file: {str(e)}")
            return False, f"Error: {str(e)}"


# Singleton
_browser_stt_service = None

def get_browser_stt_service() -> BrowserSTTService:
    """Get browser STT service instance"""
    global _browser_stt_service
    if _browser_stt_service is None:
        _browser_stt_service = BrowserSTTService()
    return _browser_stt_service
