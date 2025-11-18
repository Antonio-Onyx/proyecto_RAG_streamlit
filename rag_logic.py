import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# by default we use groq
from groq import Groq

# additional import for another clients can be added here
try:
    from google import genai
except ImportError:
    genai = None

class BaseLLMClient(ABC):
    @abstractmethod
    def generate_content(self, prompt: str, model_name: str):
        """
        must be a generator (use 'yield) that returns content chunks
        """ 
        pass

class GroqClient(BaseLLMClient):
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        print("GroqClient initialized")

    def generate_content(self, prompt: str, model_name: str):
        message = [
            {
                "role":"user",
                "content": prompt
            }
        ]

        try:
            response = self.client.chat.completions.create(
                messages=message,
                model=model_name,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Error generating content with Groq Client: {e}")
            yield f"Error: {e}"


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str):
        if genai is None:
            raise ImportError("Please install the gemini genai package: pip install -q -U google-genai")
        genai.configure(api_key=api_key)
        print("Gemini Client initialized.")

    def generate_content(self, prompt: str, model_name: str):
        client = genai.Client()
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                stream=True
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            print(f"Error generation content with Gemini client: {e}")
            yield f"Error: {e}"


def get_llm_client() -> BaseLLMClient:
    """
    read variables from .env and return instance of corresponging LLM client
    """
    api_host = os.getenv("API_HOST", "groq").lower()

    if api_host == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        return GroqClient(api_key=api_key)
    elif api_host == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        return GeminiClient(api_key=api_key)
    else:
        raise ValueError(f"Unsupported API_HOST: {api_host}")