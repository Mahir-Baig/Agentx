# """
# Speech-to-Text Service using Azure Cognitive Services
# Simple direct microphone approach with continuous recognition support
# """

# import os
# import threading
# from typing import Tuple, Optional
# import azure.cognitiveservices.speech as speechsdk
# from dotenv import load_dotenv
# from src.logger import logger

# load_dotenv()

# class SpeechToTextService:
#     """Azure Speech-to-Text service"""
    
#     def __init__(self):
#         """Initialize Azure Speech configuration"""
#         self.speech_key = os.getenv('SPEECH_KEY')
#         self.speech_endpoint = os.getenv('SPEECH_ENDPOINT')
        
#         if not self.speech_key or not self.speech_endpoint:
#             logger.warning("Azure Speech credentials not configured")
#             self.speech_config = None
#         else:
#             self.speech_config = speechsdk.SpeechConfig(
#                 subscription=self.speech_key,
#                 endpoint=self.speech_endpoint
#             )
#             self.speech_config.speech_recognition_language = "en-US"
#             logger.info("Speech-to-Text initialized")
        
#         # For continuous recognition
#         self.recognizer: Optional[speechsdk.SpeechRecognizer] = None
#         self.recognized_text = []
#         self.is_recognizing = False
#         self.recognition_done = threading.Event()
    
#     def recognize_from_microphone(self) -> Tuple[bool, str]:
#         """
#         Direct microphone recognition - exactly like Azure example
        
#         Returns:
#             Tuple of (success: bool, text: str)
#         """
#         try:
#             if not self.speech_config:
#                 return False, "Set SPEECH_KEY and SPEECH_ENDPOINT"
            
#             # Use default microphone
#             audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
#             speech_recognizer = speechsdk.SpeechRecognizer(
#                 speech_config=self.speech_config,
#                 audio_config=audio_config
#             )
            
#             logger.info("Speak into your microphone...")
#             result = speech_recognizer.recognize_once_async().get()
            
#             if result.reason == speechsdk.ResultReason.RecognizedSpeech:
#                 logger.info(f"Recognized: {result.text}")
#                 return True, result.text
            
#             elif result.reason == speechsdk.ResultReason.NoMatch:
#                 logger.warning(f"No speech: {result.no_match_details}")
#                 return False, "No speech detected"
            
#             elif result.reason == speechsdk.ResultReason.Canceled:
#                 details = result.cancellation_details
#                 logger.error(f"Canceled: {details.reason}")
#                 if details.reason == speechsdk.CancellationReason.Error:
#                     logger.error(f"Error: {details.error_details}")
#                     return False, f"Error: {details.error_details}"
#                 return False, "Recognition canceled"
            
#             return False, "Unknown error"
            
#         except Exception as e:
#             logger.error(f"STT error: {str(e)}")
#             return False, f"Error: {str(e)}"
    
#     def start_continuous_recognition(self) -> Tuple[bool, str]:
#         """
#         Start continuous speech recognition (non-blocking)
        
#         Returns:
#             Tuple of (success: bool, message: str)
#         """
#         try:
#             if not self.speech_config:
#                 return False, "Set SPEECH_KEY and SPEECH_ENDPOINT"
            
#             if self.is_recognizing:
#                 return False, "Already recording"
            
#             # Reset state
#             self.recognized_text = []
#             self.recognition_done.clear()
            
#             # Use default microphone
#             audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
#             self.recognizer = speechsdk.SpeechRecognizer(
#                 speech_config=self.speech_config,
#                 audio_config=audio_config
#             )
            
#             # Connect callbacks
#             def recognized_cb(evt):
#                 if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
#                     self.recognized_text.append(evt.result.text)
#                     logger.info(f"Recognized: {evt.result.text}")
            
#             def stopped_cb(evt):
#                 logger.info("Recognition stopped")
#                 self.is_recognizing = False
#                 self.recognition_done.set()
            
#             self.recognizer.recognized.connect(recognized_cb)
#             self.recognizer.session_stopped.connect(stopped_cb)
#             self.recognizer.canceled.connect(stopped_cb)
            
#             # Start continuous recognition
#             self.recognizer.start_continuous_recognition_async()
#             self.is_recognizing = True
            
#             logger.info("Started continuous recognition")
#             return True, "Recording started"
            
#         except Exception as e:
#             logger.error(f"Error starting recognition: {str(e)}")
#             self.is_recognizing = False
#             return False, f"Error: {str(e)}"
    
#     def stop_continuous_recognition(self) -> Tuple[bool, str]:
#         """
#         Stop continuous speech recognition and return recognized text
        
#         Returns:
#             Tuple of (success: bool, text: str)
#         """
#         try:
#             if not self.is_recognizing or not self.recognizer:
#                 return False, "Not recording"
            
#             # Stop recognition
#             self.recognizer.stop_continuous_recognition_async()
            
#             # Wait for it to complete (with timeout)
#             self.recognition_done.wait(timeout=2.0)
            
#             # Combine all recognized text
#             full_text = " ".join(self.recognized_text).strip()
            
#             if full_text:
#                 logger.info(f"Final recognized text: {full_text}")
#                 return True, full_text
#             else:
#                 return False, "No speech detected"
            
#         except Exception as e:
#             logger.error(f"Error stopping recognition: {str(e)}")
#             self.is_recognizing = False
#             return False, f"Error: {str(e)}"


# # Singleton
# _stt_service = None

# def get_stt_service() -> SpeechToTextService:
#     """Get STT service instance"""
#     global _stt_service
#     if _stt_service is None:
#         _stt_service = SpeechToTextService()
#     return _stt_service





"""
Speech-to-Text Service using Azure Cognitive Services
Simple direct microphone approach with continuous recognition support
"""

import os
import threading
from typing import Tuple, Optional
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from src.logger import logger

load_dotenv()


class SpeechToTextService:
    """Azure Speech-to-Text service"""
    
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
            logger.info("Speech-to-Text initialized")
        
        # For continuous recognition
        self.recognizer: Optional[speechsdk.SpeechRecognizer] = None
        self.recognized_text = []
        self.is_recognizing = False
        self.recognition_done = threading.Event()
    
    def recognize_from_microphone(self) -> Tuple[bool, str]:
        """
        Direct microphone recognition - exactly like Azure example
        
        Returns:
            Tuple of (success: bool, text: str)
        """
        try:
            if not self.speech_config:
                return False, "Set SPEECH_KEY and SPEECH_ENDPOINT"
            
            # Use default microphone
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            logger.info("Speak into your microphone...")
            result = speech_recognizer.recognize_once_async().get()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                logger.info(f"Recognized: {result.text}")
                return True, result.text
            
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning(f"No speech: {result.no_match_details}")
                return False, "No speech detected"
            
            elif result.reason == speechsdk.ResultReason.Canceled:
                details = result.cancellation_details
                logger.error(f"Canceled: {details.reason}")
                if details.reason == speechsdk.CancellationReason.Error:
                    logger.error(f"Error: {details.error_details}")
                    return False, f"Error: {details.error_details}"
                return False, "Recognition canceled"
            
            return False, "Unknown error"
            
        except Exception as e:
            logger.error(f"STT error: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def start_continuous_recognition(self) -> Tuple[bool, str]:
        """
        Start continuous speech recognition (non-blocking)
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if not self.speech_config:
                return False, "Set SPEECH_KEY and SPEECH_ENDPOINT"
            
            if self.is_recognizing:
                return False, "Already recording"
            
            # Reset state
            self.recognized_text = []
            self.recognition_done.clear()
            
            # Use default microphone
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            self.recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # Connect callbacks
            def recognized_cb(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    self.recognized_text.append(evt.result.text)
                    logger.info(f"Recognized: {evt.result.text}")
            
            def stopped_cb(evt):
                logger.info("Recognition stopped")
                self.is_recognizing = False
                self.recognition_done.set()
            
            self.recognizer.recognized.connect(recognized_cb)
            self.recognizer.session_stopped.connect(stopped_cb)
            self.recognizer.canceled.connect(stopped_cb)
            
            # Start continuous recognition
            self.recognizer.start_continuous_recognition_async()
            self.is_recognizing = True
            
            logger.info("Started continuous recognition")
            return True, "Recording started"
            
        except Exception as e:
            logger.error(f"Error starting recognition: {str(e)}")
            self.is_recognizing = False
            return False, f"Error: {str(e)}"
    
    def stop_continuous_recognition(self) -> Tuple[bool, str]:
        """
        Stop continuous speech recognition and return recognized text
        
        Returns:
            Tuple of (success: bool, text: str)
        """
        try:
            if not self.is_recognizing or not self.recognizer:
                return False, "Not recording"
            
            # Stop recognition
            self.recognizer.stop_continuous_recognition_async()
            
            # Wait for it to complete (with timeout)
            self.recognition_done.wait(timeout=2.0)
            
            # Combine all recognized text
            full_text = " ".join(self.recognized_text).strip()
            
            if full_text:
                logger.info(f"Final recognized text: {full_text}")
                return True, full_text
            else:
                return False, "No speech detected"
            
        except Exception as e:
            logger.error(f"Error stopping recognition: {str(e)}")
            self.is_recognizing = False
            return False, f"Error: {str(e)}"


# Singleton
_stt_service = None

def get_stt_service() -> SpeechToTextService:
    """Get STT service instance"""
    global _stt_service
    if _stt_service is None:
        _stt_service = SpeechToTextService()
    return _stt_service

