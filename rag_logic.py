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

response = client.chat.completions.create(
    messages=[
        {
            "role":"user",
            "content":"Escribe un historia de amor entre una pareja de distintas clases sociales en la epoca de los años 2000 en mexico"
        }
    ],
    model="openai/gpt-oss-120b"
)

if __name__ == "__main__":
    print(response.choices[0].message.content)