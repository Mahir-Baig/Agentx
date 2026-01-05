"""
Speech-to-Text Service using Azure Cognitive Services REST API
Browser-compatible version that works with deployed Streamlit apps
Uses REST API instead of SDK to avoid audio library dependencies
"""

import os
import requests
from typing import Tuple
from dotenv import load_dotenv
from src.logger import logger

load_dotenv()


class BrowserSTTService:
    """Azure Speech-to-Text service for browser audio using REST API"""
    
    def __init__(self):
        """Initialize Azure Speech configuration"""
        self.speech_key = os.getenv('SPEECH_KEY')
        self.speech_region = os.getenv('SPEECH_REGION', 'eastus')  # Default to eastus
        
        # Extract region from endpoint if available
        speech_endpoint = os.getenv('SPEECH_ENDPOINT')
        if speech_endpoint and not self.speech_region:
            # Extract region from endpoint like https://eastus.api.cognitive.microsoft.com/
            try:
                self.speech_region = speech_endpoint.split('//')[1].split('.')[0]
            except:
                pass
        
        if not self.speech_key:
            logger.warning("Azure Speech credentials not configured (SPEECH_KEY missing)")
            self.configured = False
        else:
            self.configured = True
            logger.info(f"Browser STT initialized with region: {self.speech_region}")
    
    def recognize_from_audio_bytes(self, audio_bytes: bytes) -> Tuple[bool, str]:
        """
        Recognize speech from audio bytes using Azure REST API
        
        Args:
            audio_bytes: Audio data in WAV format from browser
            
        Returns:
            Tuple of (success: bool, text: str)
        """
        try:
            if not self.configured:
                return False, "Set SPEECH_KEY in environment variables"
            
            if not audio_bytes:
                return False, "No audio data provided"
            
            logger.info(f"Processing audio bytes: {len(audio_bytes)} bytes")
            
            # Azure Speech-to-Text REST API endpoint
            url = f"https://{self.speech_region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
            
            # Request parameters
            params = {
                'language': 'en-US',
                'format': 'detailed'
            }
            
            # Request headers
            headers = {
                'Ocp-Apim-Subscription-Key': self.speech_key,
                'Content-Type': 'audio/wav; codec=audio/pcm; samplerate=16000',
                'Accept': 'application/json'
            }
            
            logger.info("Sending audio to Azure Speech REST API...")
            
            # Make the request
            response = requests.post(
                url,
                params=params,
                headers=headers,
                data=audio_bytes,
                timeout=30
            )
            
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"API Response: {result}")
                
                # Check recognition status
                if result.get('RecognitionStatus') == 'Success':
                    text = result.get('DisplayText', '')
                    if text:
                        logger.info(f"Recognized: {text}")
                        return True, text
                    else:
                        return False, "No speech detected in audio"
                else:
                    status = result.get('RecognitionStatus', 'Unknown')
                    return False, f"Recognition failed: {status}"
            else:
                error_msg = f"API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return False, error_msg
            
        except requests.exceptions.Timeout:
            logger.error("Request timeout")
            return False, "Request timeout - please try again"
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return False, f"Network error: {str(e)}"
        except Exception as e:
            logger.error(f"STT error: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def recognize_from_file(self, audio_file) -> Tuple[bool, str]:
        """
        Recognize speech from uploaded audio file
        
        Args:
            audio_file: File-like object (e.g., from st.audio_input)
            
        Returns:
            Tuple of (success: bool, text: str)
        """
        try:
            # Read file content
            if hasattr(audio_file, 'getvalue'):
                # BytesIO object from st.audio_input
                audio_bytes = audio_file.getvalue()
            else:
                # Regular file object
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
