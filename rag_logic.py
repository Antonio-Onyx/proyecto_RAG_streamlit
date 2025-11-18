import os
from dotenv import load_dotenv

from groq import Groq

# load enviroment variables from .env file
load_dotenv()
API_HOST = os.getenv("API_HOST", "groq")

if API_HOST == "groq":
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY"),
    )
elif API_HOST == "gemini":
    #first we check if we have the gemini genai package installed
    try:
        from google import genai
    except ImportError:
        raise ImportError("Please install the gemini genai package: pip install -q -U google-genai")
    
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Escribe un historia de amor entre una pareja de distintas clases sociales en la epoca de los años 2000 en mexico"
    )

response = client.chat.completions.create(
    messages=[
        {
            "role":"user",
            "content":"Escribe un historia de amor entre una pareja de distintas clases sociales en la epoca de los años 2000 en mexico"
        }
    ],
    model="openai/gpt-oss-120b",
    stream=True
)

if __name__ == "__main__":
    print("Generando historia...")
    full_story = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_story += token
            print(token, end="", flush=True)