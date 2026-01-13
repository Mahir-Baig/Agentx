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
            import wave
            
            if not self.speech_config:
                logger.error("Speech config not initialized - missing credentials")
                return False, "Azure Speech Service not configured"
            
            if not audio_bytes:
                logger.error("No audio data provided")
                return False, "No audio data provided"
            
            logger.info(f"Processing audio bytes: {len(audio_bytes)} bytes")
            
            raw_data = None
            sample_rate = 16000
            channels = 1
            bits_per_sample = 16
            
            try:
                with io.BytesIO(audio_bytes) as wav_buffer:
                    with wave.open(wav_buffer, 'rb') as wav_file:
                        channels = wav_file.getnchannels()
                        sample_rate = wav_file.getframerate()
                        bits_per_sample = wav_file.getsampwidth() * 8
                        num_frames = wav_file.getnframes()
                        raw_data = wav_file.readframes(num_frames)
                        duration = num_frames / sample_rate
                        
                logger.info(f"WAV Format: {sample_rate}Hz, {channels}ch, {bits_per_sample}bit, {num_frames} frames, {duration:.2f}s duration")
                logger.info(f"Raw audio data size: {len(raw_data)} bytes")
                
            except Exception as e:
                logger.warning(f"Failed to parse WAV header: {e}. Trying raw upload.")
                raw_data = audio_bytes
                
            if not raw_data or len(raw_data) == 0:
                logger.error("No audio data after processing")
                return False, "No valid audio data"
                
            stream_format = speechsdk.audio.AudioStreamFormat(
                samples_per_second=sample_rate,
                bits_per_sample=bits_per_sample,
                channels=channels
            )
            
            audio_stream = speechsdk.audio.PushAudioInputStream(stream_format=stream_format)
            audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
            
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            logger.info(f"Writing {len(raw_data)} bytes to audio stream...")
            audio_stream.write(raw_data)
            audio_stream.close()
            
            logger.info("Starting speech recognition...")
            result = speech_recognizer.recognize_once_async().get()
            logger.info(f"Recognition result reason: {result.reason}")
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                logger.info(f"✅ Recognized: {result.text}")
                return True, result.text
            
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning(f"❌ No speech detected: {result.no_match_details}")
                return False, "No speech detected in audio. Please speak clearly and try again."
            
            elif result.reason == speechsdk.ResultReason.Canceled:
                details = result.cancellation_details
                logger.error(f"❌ Recognition canceled: {details.reason}")
                if details.reason == speechsdk.CancellationReason.Error:
                    logger.error(f"Error details: {details.error_details}")
                    return False, f"{details.error_details}"
                return False, "Recognition was canceled. Please try again."
            
            return False, "Unknown recognition error"
            
        except Exception as e:
            logger.error(f"❌ STT Exception: {str(e)}", exc_info=True)
            return False, f"Error processing audio: {str(e)}"
    
    def recognize_from_file(self, audio_file) -> Tuple[bool, str]:
        """
        Recognize speech from uploaded audio file
        
        Args:
            audio_file: File-like object (e.g., from st.file_uploader)
            
        Returns:
            Tuple of (success: bool, text: str)
        """
        try:
            audio_bytes = audio_file.read()
            
            if hasattr(audio_file, 'seek'):
                audio_file.seek(0)
            
            return self.recognize_from_audio_bytes(audio_bytes)
            
        except Exception as e:
            logger.error(f"Error reading audio file: {str(e)}")
            return False, f"Error: {str(e)}"


_browser_stt_service = None

def get_browser_stt_service() -> BrowserSTTService:
    """Get browser STT service instance"""
    global _browser_stt_service
    if _browser_stt_service is None:
        _browser_stt_service = BrowserSTTService()
    return _browser_stt_service
