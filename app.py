import os
from dotenv import load_dotenv
from rag_logic import get_llm_client

load_dotenv()

API_HOST = os.getenv("API_HOST", "groq").lower()

if API_HOST == "groq":
    MODEL_TO_USE = "openai/gpt-oss-120b"
elif API_HOST == "gemini":
    MODEL_TO_USE = "gemini-2.5-flash"

PROMPT = "Escribe un historia de amor entre una pareja de distintas clases sociales en la epoca de los años 2000 en mexico"

if __name__ == "__main__":
    try:
        client = get_llm_client()
        print(f"Using {API_HOST} client with model: {MODEL_TO_USE}")
        response = client.generate_content(PROMPT, MODEL_TO_USE)

        full_story = ""
        for token in response:
            full_story += token
            print(token, end="", flush=True)
        print("\n\n--- End of story ---")
    except Exception as e:
        print(f"\n[ERROR] an error occured: {e}")