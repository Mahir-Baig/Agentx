"""
Text-to-Speech Service using Azure Cognitive Services
Converts text to audio for read aloud functionality in the RAG assistant
"""

import os
import tempfile
from typing import Optional, Tuple
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from src.logger import logger

load_dotenv()


class TextToSpeechService:
    """
    Azure Text-to-Speech service for converting text to audio
    """
    
    def __init__(self, voice_name: str = "en-US-AvaMultilingualNeural"):
        """
        Initialize Azure Speech configuration
        
        Args:
            voice_name: Azure neural voice name (default: en-US-AvaMultilingualNeural)
        """
        self.speech_key = os.getenv('SPEECH_KEY')
        self.speech_endpoint = os.getenv('SPEECH_ENDPOINT')
        self.voice_name = voice_name
        
        if not self.speech_key or not self.speech_endpoint:
            logger.warning("Azure Speech credentials not configured")
            self.speech_config = None
        else:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                endpoint=self.speech_endpoint
            )
            self.speech_config.speech_synthesis_voice_name = self.voice_name
            logger.info(f"Text-to-Speech service initialized with voice: {self.voice_name}")
    
    def synthesize_to_file(self, text: str, output_path: str) -> Tuple[bool, str]:
        """
        Synthesize text to audio file
        
        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file (WAV format)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if not self.speech_config:
                return False, "Speech service not configured. Please set SPEECH_KEY and SPEECH_ENDPOINT."
            
            logger.info(f"Synthesizing text to file: {output_path}")
            
            audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            result = synthesizer.speak_text_async(text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info(f"Speech synthesized successfully")
                return True, "Audio generated successfully"
            
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                logger.error(f"Speech synthesis canceled: {cancellation.reason}")
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    logger.error(f"Error details: {cancellation.error_details}")
                    return False, f"Error: {cancellation.error_details}"
                return False, "Speech synthesis was canceled."
            
            return False, "Unknown error occurred."
            
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def synthesize_to_bytes(self, text: str) -> Tuple[bool, Optional[bytes], str]:
        """
        Synthesize text to audio bytes
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Tuple of (success: bool, audio_bytes: Optional[bytes], message: str)
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_path = tmp_file.name
            
            # Synthesize to file
            success, message = self.synthesize_to_file(text, tmp_path)
            
            if not success:
                return False, None, message
            
            # Read the audio bytes
            with open(tmp_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Clean up
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            return True, audio_bytes, "Audio generated successfully"
            
        except Exception as e:
            logger.error(f"Error generating audio bytes: {str(e)}")
            return False, None, f"Error: {str(e)}"


# Singleton instance
_tts_service = None

def get_tts_service() -> TextToSpeechService:
    """Get or create TTS service instance"""
    global _tts_service
    if _tts_service is None:
        _tts_service = TextToSpeechService()
    return _tts_service