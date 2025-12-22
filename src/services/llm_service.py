from langchain_openai import AzureChatOpenAI
import warnings
# Suppress the Google Generative AI deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
import google.genai as genai
from groq import Groq
from langsmith import traceable
from src.logger import logger
from src.exceptions import RagException
import sys
import os
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

load_dotenv()


class LLMService:
    """
    LLM Service supporting Azure OpenAI, Groq (Meta LLaMA), and Google Gemini Pro models.
    """
    
    def __init__(self):
        """Initialize LLM Service with all providers."""
        logger.info("="*80)
        logger.info("Initializing LLM Service")
        logger.info("="*80)
        
        # Azure OpenAI configuration
        self.azure_client = None
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
        
        # Groq configuration
        self.groq_client = None
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Gemini configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Initialize clients
        self._init_azure_client()
        self._init_groq_client()
        # self._init_gemini_client()
        
        logger.info("✓ LLM Service initialization completed")
        logger.info("="*80)
    
    def _init_azure_client(self):
        """Initialize Azure OpenAI client."""
        try:
            if self.azure_endpoint and self.azure_api_key:
                self.azure_client = AzureChatOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    api_key=self.azure_api_key,
                    api_version=self.azure_api_version,
                    azure_deployment=self.azure_deployment
                )
                logger.info(f"✓ Azure OpenAI client initialized (deployment: {self.azure_deployment})")
            else:
                logger.warning("⚠ Azure OpenAI credentials not found")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
    
    def _init_groq_client(self):
        """Initialize Groq client."""
        try:
            if self.groq_api_key:
                self.groq_client = Groq(api_key=self.groq_api_key)
                logger.info("✓ Groq client initialized")
            else:
                logger.warning("⚠ Groq API key not found")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
    
    def _init_gemini_client(self):
        """Initialize Google Gemini client."""
        try:
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                logger.info("✓ Google Gemini client initialized")
            else:
                logger.warning("⚠ Gemini API key not found")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
    
    @traceable(name="azure_chat_completion", tags=["llm", "azure", "chat"])
    def azure_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate chat completion using Azure OpenAI.
        
        :param messages: List of message dictionaries with 'role' and 'content'.
        :param model: Azure deployment name (e.g., 'gpt-35-turbo', 'gpt-4').
        :param temperature: Sampling temperature (0-2).
        :param max_tokens: Maximum tokens in response.
        :param stream: Whether to stream the response.
        :return: Chat completion response.
        """
        try:
            if not self.azure_client:
                raise ValueError("Azure OpenAI client not initialized")
            
            # Use deployment from environment if model not specified
            if model is None:
                model = self.azure_deployment
            
            logger.info("="*60)
            logger.info("Azure OpenAI Chat Completion")
            logger.info("="*60)
            logger.info(f"Model: {model}")
            logger.info(f"Temperature: {temperature}")
            logger.info(f"Max tokens: {max_tokens}")
            logger.info(f"Messages count: {len(messages)}")
            logger.info(f"Stream: {stream}")
            
            # Log messages (truncated for privacy)
            for i, msg in enumerate(messages):
                content_preview = msg.get('content', '')[:100]
                logger.info(f"Message {i+1} ({msg.get('role', 'unknown')}): {content_preview}...")
            
            # Convert messages to LangChain format
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            
            langchain_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role == 'system':
                    langchain_messages.append(SystemMessage(content=content))
                elif role == 'user':
                    langchain_messages.append(HumanMessage(content=content))
                elif role == 'assistant':
                    langchain_messages.append(AIMessage(content=content))
            
            # Create a new client instance with the specified model if different from default
            if model != self.azure_deployment:
                client = AzureChatOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    api_key=self.azure_api_key,
                    api_version=self.azure_api_version,
                    azure_deployment=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                # Use existing client with updated parameters
                client = AzureChatOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    api_key=self.azure_api_key,
                    api_version=self.azure_api_version,
                    azure_deployment=self.azure_deployment,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            # Generate response
            if stream:
                response = client.stream(langchain_messages)
                logger.info("✓ Streaming response initiated")
                return {"response": response, "streaming": True}
            else:
                response = client.invoke(langchain_messages)
                content = response.content
                
                logger.info(f"✓ Response generated successfully")
                logger.info(f"✓ Response length: {len(content)} characters")
                logger.info("="*60)
                
                return {
                    "content": content,
                    "usage": {
                        "prompt_tokens": getattr(response, 'usage_metadata', {}).get('input_tokens', 0),
                        "completion_tokens": getattr(response, 'usage_metadata', {}).get('output_tokens', 0),
                        "total_tokens": getattr(response, 'usage_metadata', {}).get('total_tokens', 0)
                    },
                    "model": model,
                    "provider": "azure_openai"
                }
        
        except Exception as e:
            logger.error(f"Error in Azure chat completion: {e}")
            raise RagException(f"Error in Azure chat completion: {e}", sys)
    
    def groq_llama_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama3-8b-8192",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate chat completion using Groq with Meta LLaMA models.
        
        :param messages: List of message dictionaries with 'role' and 'content'.
        :param model: Groq model name (e.g., 'llama3-8b-8192', 'llama3-70b-8192', 'mixtral-8x7b-32768').
        :param temperature: Sampling temperature (0-2).
        :param max_tokens: Maximum tokens in response.
        :param stream: Whether to stream the response.
        :return: Chat completion response.
        """
        try:
            if not self.groq_client:
                raise ValueError("Groq client not initialized")
            
            logger.info("="*60)
            logger.info("Groq Meta LLaMA Chat Completion")
            logger.info("="*60)
            logger.info(f"Model: {model}")
            logger.info(f"Temperature: {temperature}")
            logger.info(f"Max tokens: {max_tokens}")
            logger.info(f"Messages count: {len(messages)}")
            logger.info(f"Stream: {stream}")
            
            # Log messages (truncated for privacy)
            for i, msg in enumerate(messages):
                content_preview = msg.get('content', '')[:100]
                logger.info(f"Message {i+1} ({msg.get('role', 'unknown')}): {content_preview}...")
            
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                logger.info("✓ Streaming response initiated")
                return {"response": response, "streaming": True}
            else:
                content = response.choices[0].message.content
                usage = response.usage
                
                logger.info(f"✓ Response generated successfully")
                logger.info(f"✓ Response length: {len(content)} characters")
                logger.info(f"✓ Tokens used - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")
                logger.info("="*60)
                
                return {
                    "content": content,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens
                    },
                    "model": model,
                    "provider": "groq_llama"
                }
        
        except Exception as e:
            logger.error(f"Error in Groq LLaMA completion: {e}")
            raise RagException(f"Error in Groq LLaMA completion: {e}", sys)
    
    def gemini_pro_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemini-pro",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate chat completion using Google Gemini Pro models.
        
        :param messages: List of message dictionaries with 'role' and 'content'.
        :param model: Gemini model name (e.g., 'gemini-pro', 'gemini-pro-vision').
        :param temperature: Sampling temperature (0-1).
        :param max_tokens: Maximum tokens in response.
        :return: Chat completion response.
        """
        try:
            if not self.gemini_api_key:
                raise ValueError("Gemini client not initialized")
            
            logger.info("="*60)
            logger.info("Google Gemini Pro Chat Completion")
            logger.info("="*60)
            logger.info(f"Model: {model}")
            logger.info(f"Temperature: {temperature}")
            logger.info(f"Max tokens: {max_tokens}")
            logger.info(f"Messages count: {len(messages)}")
            
            # Log messages (truncated for privacy)
            for i, msg in enumerate(messages):
                content_preview = msg.get('content', '')[:100]
                logger.info(f"Message {i+1} ({msg.get('role', 'unknown')}): {content_preview}...")
            
            # Convert messages to Gemini format
            gemini_messages = self._convert_to_gemini_format(messages)
            
            # Initialize model
            gemini_model = genai.GenerativeModel(model)
            
            # Configure generation
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            # Generate response
            response = gemini_model.generate_content(
                gemini_messages,
                generation_config=generation_config
            )
            
            content = response.text
            
            # Extract usage information if available
            usage_metadata = getattr(response, 'usage_metadata', None)
            usage = {
                "prompt_tokens": getattr(usage_metadata, 'prompt_token_count', 0) if usage_metadata else 0,
                "completion_tokens": getattr(usage_metadata, 'candidates_token_count', 0) if usage_metadata else 0,
                "total_tokens": getattr(usage_metadata, 'total_token_count', 0) if usage_metadata else 0
            }
            
            logger.info(f"✓ Response generated successfully")
            logger.info(f"✓ Response length: {len(content)} characters")
            if usage['total_tokens'] > 0:
                logger.info(f"✓ Tokens used - Prompt: {usage['prompt_tokens']}, Completion: {usage['completion_tokens']}, Total: {usage['total_tokens']}")
            logger.info("="*60)
            
            return {
                "content": content,
                "usage": usage,
                "model": model,
                "provider": "google_gemini"
            }
        
        except Exception as e:
            logger.error(f"Error in Gemini Pro completion: {e}")
            raise RagException(f"Error in Gemini Pro completion: {e}", sys)
    
    def _convert_to_gemini_format(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI-style messages to Gemini format.
        
        :param messages: List of message dictionaries.
        :return: Formatted string for Gemini.
        """
        gemini_prompt = ""
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                gemini_prompt += f"System: {content}\n\n"
            elif role == 'user':
                gemini_prompt += f"User: {content}\n\n"
            elif role == 'assistant':
                gemini_prompt += f"Assistant: {content}\n\n"
        
        return gemini_prompt.strip()
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get list of available models for each provider.
        
        :return: Dictionary of available models by provider.
        """
        models = {
            "azure_openai": [
                "gpt-35-turbo",
                "gpt-35-turbo-16k", 
                "gpt-4",
                "gpt-4-32k",
                "gpt-4-turbo",
                "gpt-4o"
            ],
            "groq_llama": [
                "llama3-8b-8192",
                "llama3-70b-8192",
                "mixtral-8x7b-32768",
                "gemma-7b-it"
            ],
            "google_gemini": [
                "gemini-pro",
                "gemini-pro-vision"
            ]
        }
        
        logger.info("Available models:")
        for provider, provider_models in models.items():
            logger.info(f"  {provider}: {provider_models}")
        
        return models
    
    @traceable(name="chat_completion", tags=["llm", "chat"])
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        provider: str = "azure_openai",
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Universal chat completion method that routes to appropriate provider.
        
        :param messages: List of message dictionaries.
        :param provider: Provider name ('azure_openai', 'groq_llama', 'google_gemini').
        :param model: Model name (provider-specific).
        :param temperature: Sampling temperature.
        :param max_tokens: Maximum tokens in response.
        :param stream: Whether to stream (not supported by all providers).
        :return: Chat completion response.
        """
        logger.info(f"Universal chat completion - Provider: {provider}")
        
        if provider == "azure_openai":
            model = model or "gpt-35-turbo"
            return self.azure_chat_completion(messages, model, temperature, max_tokens, stream)
        
        elif provider == "groq_llama":
            model = model or "llama3-8b-8192"
            return self.groq_llama_completion(messages, model, temperature, max_tokens, stream)
        
        elif provider == "google_gemini":
            model = model or "gemini-pro"
            return self.gemini_pro_completion(messages, model, temperature, max_tokens)
        
        else:
            raise ValueError(f"Unsupported provider: {provider}. Choose from: azure_openai, groq_llama, google_gemini")
    
    def health_check(self) -> Dict[str, bool]:
        """
        Check health status of all LLM providers.
        
        :return: Dictionary with provider availability status.
        """
        logger.info("Performing LLM service health check...")
        
        health = {
            "azure_openai": self.azure_client is not None,
            "groq_llama": self.groq_client is not None,
            "google_gemini": self.gemini_api_key is not None
        }
        
        logger.info("Health check results:")
        for provider, status in health.items():
            status_text = "✓ Available" if status else "✗ Not available"
            logger.info(f"  {provider}: {status_text}")
        
        return health